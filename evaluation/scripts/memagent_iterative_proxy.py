#!/usr/bin/env python3

import os
import time
import uuid
from typing import Any, Dict, List

import requests
from flask import Flask, jsonify, request
from transformers import AutoTokenizer


UPDATE_MEMORY_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.
<problem>
{prompt}
</problem>
<memory>
{memory}
</memory>
<section>
{chunk}
</section>
Updated memory:
"""


FINAL_ANSWER_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory.
<problem>
{prompt}
</problem>
<memory>
{memory}
</memory>
Your answer:
"""


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


BACKEND_MODEL = os.environ.get("MEMAGENT_BACKEND_MODEL", "BytedTsinghua-SIA/RL-MemoryAgent-7B")
PROXY_MODEL = os.environ.get("MEMAGENT_PROXY_MODEL", "memagent-iterative-proxy")
BACKEND_API_URL = os.environ.get("MEMAGENT_BACKEND_API_URL", "http://localhost:9002/v1/chat/completions")
SUMMARY_TASK_PROMPT = os.environ.get(
    "MEMAGENT_SUMMARY_TASK_PROMPT",
    "Create a detailed summary of the conversation history that preserves entities, actions, tool calls, observations, errors, and outcomes. Do not include recommendations for future actions.",
)
CHUNK_SIZE = int(os.environ.get("MEMORY_CHUNK_SIZE", "5000"))
DEFAULT_TEMPERATURE = float(os.environ.get("MEMAGENT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.environ.get("MEMAGENT_MAX_TOKENS", "1536"))
REQUEST_TIMEOUT = int(os.environ.get("MEMAGENT_TIMEOUT_SEC", "120"))
FORCE_ITERATIVE = _to_bool(os.environ.get("MEMAGENT_FORCE_ITERATIVE", "true"), default=True)

print(f"Loading tokenizer: {BACKEND_MODEL}")
TOKENIZER = AutoTokenizer.from_pretrained(BACKEND_MODEL, trust_remote_code=True)

app = Flask(__name__)


def _extract_prompt(messages: List[Dict[str, Any]]) -> str:
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)
    return str(content)


def _extract_conversation(summary_prompt: str) -> str:
    marker = "### Conversation:\n"
    tail = "\n\nPlease provide a detailed summary"

    if marker in summary_prompt:
        after = summary_prompt.split(marker, 1)[1]
        if tail in after:
            return after.split(tail, 1)[0]
        return after

    return summary_prompt


def _backend_chat(prompt_text: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": BACKEND_MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(BACKEND_API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    body = response.json()
    return body["choices"][0]["message"]["content"]


def _iterative_summary(conversation: str, temperature: float, max_tokens: int) -> str:
    input_ids = TOKENIZER.encode(conversation)
    memory = "No previous memory"

    for start in range(0, len(input_ids), CHUNK_SIZE):
        chunk_ids = input_ids[start : start + CHUNK_SIZE]
        chunk_text = TOKENIZER.decode(chunk_ids)
        update_prompt = UPDATE_MEMORY_TEMPLATE.format(
            prompt=SUMMARY_TASK_PROMPT,
            memory=memory,
            chunk=chunk_text,
        )
        memory = _backend_chat(update_prompt, temperature=temperature, max_tokens=max_tokens)

    final_prompt = FINAL_ANSWER_TEMPLATE.format(prompt=SUMMARY_TASK_PROMPT, memory=memory)
    final_summary = _backend_chat(final_prompt, temperature=temperature, max_tokens=max_tokens)
    return final_summary


def _chat_response(content: str, model: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "backend_model": BACKEND_MODEL,
            "backend_api_url": BACKEND_API_URL,
            "chunk_size": CHUNK_SIZE,
            "force_iterative": FORCE_ITERATIVE,
        }
    )


@app.route("/v1/models", methods=["GET"])
def models() -> Any:
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": PROXY_MODEL,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "memagent-proxy",
                }
            ],
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions() -> Any:
    payload = request.get_json(force=True, silent=True) or {}
    model = payload.get("model", PROXY_MODEL)
    messages = payload.get("messages", [])
    temperature = float(payload.get("temperature", DEFAULT_TEMPERATURE))
    max_tokens = int(payload.get("max_tokens", DEFAULT_MAX_TOKENS))

    try:
        prompt_text = _extract_prompt(messages)
        conversation = _extract_conversation(prompt_text)

        if FORCE_ITERATIVE:
            response_text = _iterative_summary(conversation, temperature=temperature, max_tokens=max_tokens)
        else:
            response_text = _backend_chat(prompt_text, temperature=temperature, max_tokens=max_tokens)

        return jsonify(_chat_response(response_text, model=model))
    except Exception as exc:
        return (
            jsonify(
                {
                    "error": {
                        "message": f"memagent iterative proxy error: {exc}",
                        "type": "proxy_error",
                        "code": "memagent_proxy_error",
                    }
                }
            ),
            500,
        )


if __name__ == "__main__":
    host = os.environ.get("MEMAGENT_PROXY_HOST", "0.0.0.0")
    port = int(os.environ.get("MEMAGENT_PROXY_PORT", "9001"))
    print(f"Starting memagent iterative proxy on {host}:{port}")
    print(f"Backend: {BACKEND_API_URL} | model: {BACKEND_MODEL}")
    app.run(host=host, port=port, threaded=True)
