"""
Benchmark agent for full-context latency measurement.

Thin subclass of Agent from agent_vllm.py that adds:
1. Input token padding (via model_query override — pads messages before querying)
2. Output token control (via min_tokens/max_tokens on the vLLM call)

The run() loop is NOT overridden — the real full-context pipeline
(growing conversation history, timing) runs unmodified.
"""

import copy
import time
import requests
from typing import Any, Dict, List, Optional

import litellm
from openai import OpenAI

from r2egym.agenthub.agent.agent_vllm import Agent, AgentArgs
from r2egym.agenthub.environment.dummy_env import generate_text_with_n_tokens
from transformers import AutoTokenizer


class BenchmarkFullContextAgent(Agent):
    """Full-context agent with controlled input/output token counts for benchmarking.

    Overrides only model_query to:
    - Pad messages to input_token_target
    - Use min_tokens/max_tokens for controlled output length

    Everything else (run loop, history management, timing, action parsing)
    runs unmodified from Agent.
    """

    def __init__(
        self,
        name: str,
        args: AgentArgs,
        input_token_target: int = 4096,
        output_token_target: int = 2048,
        logger=None,
    ):
        super().__init__(name, args, logger)
        self.input_token_target = input_token_target
        self.output_token_target = output_token_target

        # Load the agent model's tokenizer for padding generation
        model_name = "/".join(args.llm_name.split("/")[1:])  # strip "hosted_vllm/" prefix
        self.padding_tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _pad_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Pad messages to input_token_target tokens."""
        current_tokens = self._count_tokens(messages)
        if current_tokens < self.input_token_target:
            deficit = self.input_token_target - current_tokens
            padding = generate_text_with_n_tokens(deficit, self.padding_tokenizer)
            messages[-1]["content"] += "\n\n[BENCHMARK CONTEXT PADDING]\n" + padding
            self.logger.info(
                f"Padded input from {current_tokens} to ~{self.input_token_target} tokens "
                f"(added {deficit} tokens of padding)"
            )
        else:
            self.logger.info(
                f"Input already at {current_tokens} tokens (target: {self.input_token_target})"
            )
        return messages

    def parse_response(self, response_text: str):
        """Override to keep the FULL raw model output as the 'thought'.

        The parent parse_response extracts only <think>/<function> tags,
        discarding most of the output. For benchmarking, we need the full
        output stored in history so prompt tokens grow realistically.
        """
        from r2egym.agenthub.action import Action
        # Keep all output as thought; use an empty action
        thought = response_text.strip() if response_text else ""
        action = Action.from_string("")
        return thought, action

    def model_query(
        self, messages: List[Dict[str, str]], temperature: float = 0,
    ) -> Any:
        """Query the LLM with controlled input/output token count.

        Simplified version of the parent model_query — no tools or function
        calling since the benchmark uses random input/output tokens.
        Skips the MAX_CONTEXT_TOKENS check since we're deliberately padding.
        """
        response = None
        retries = 0

        messages_ = copy.deepcopy(messages)
        # Pad messages to target
        messages_ = self._pad_messages(messages_)

        vllm_llm_name = "/".join(self.llm_name.split("/")[1:])  # remove hosted_vllm/ prefix

        start_time = time.perf_counter()
        while retries < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=vllm_llm_name,
                    messages=messages_,
                    timeout=self.llm_timeout,
                    temperature=temperature,
                    max_tokens=self.output_token_target,
                    extra_body={"min_tokens": self.output_token_target},
                )
                break
            except Exception as e:
                self.logger.error(f"LLM query failed @ {retries}: {e}")
                retries += 1
                if "RateLimitError" in str(e):
                    time.sleep(60)
                if retries >= self.max_retries:
                    raise e

        end_time = time.perf_counter()
        exec_time = end_time - start_time
        vllm_metrics = requests.get(
            self.llm_base_url.replace("/v1", "/metrics")
        ).text
        return response, exec_time, start_time, end_time, vllm_metrics
