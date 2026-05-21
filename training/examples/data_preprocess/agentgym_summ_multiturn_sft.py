"""
This script collects summarizations of 
"""

import json
import os
import re
import torch
import numpy as np
import pandas as pd
import logging
from copy import deepcopy
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from tqdm import tqdm
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import Any, List, Dict, Tuple
from openai import OpenAI, AsyncOpenAI
import asyncio
import itertools
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset, Dataset, concatenate_datasets

from argparse import ArgumentParser

# seed
np.random.seed(42)

logger = logging.getLogger(__name__)

compressor_prompt_v1 = """You are a helpful assistant that summarizes conversations.
Given the following conversation between a user and an assistant, provide a concise summary of the key points discussed.

### Conversation:
{conversation}

Please provide a summary of the above conversation."""

compressor_prompt_v2 = """You are a helpful assistant that summarizes conversations for another model to use. You need to make sure that the summary captures all important details from the conversation so that the other model can understand the context just like reading the full conversation.

### Conversation:
{conversation}

Please provide a detailed summary of the above conversation. Make sure to include all necessary details for the other model to understand the context, especially entities, actions taken, and outcomes."""

compressor_prompt_v3 = """You are a helpful assistant that summarizes conversations for another model to use. You need to make sure that the summary captures all important details from the conversation so that the other model can understand the context just like reading the full conversation.

### Conversation:
{conversation}

Please provide a detailed summary of the above conversation. Make sure to include the following in your summary:
- What is the task or goal discussed?
- What is the specific initial context or setting?
- A summary of action traces or steps taken. What are the feedbacks? Any failures or successes?
- Make sure your output is clear and structured, markdown format is preferred."""

compressor_prompt_v4 = """You are a helpful assistant that summarizes conversations for another model to use. You need to make sure that the summary captures all important details from the conversation so that the other model can understand the context just like reading the full conversation. You may refer to the following example output:

### Example:
In this ALFRED Embodied Environment episode, the agent begins in a bathroom-like room featuring countertop 1, drawers 1–8, garbagecan 1, handtowelholder 1, sinkbasins 1–2, toilet 1, toiletpaperhanger 1, and towelholder 1; the stated objective is to clean a soapbar and place it in the garbagecan. From the initial admissible actions list that includes navigating to each object, the agent first executes <action>go to countertop 1</action>. Upon arrival, the observation specifies that countertop 1 holds candle 1, candle 2, soapbar 1, soapbottle 1, soapbottle 2, and spraybottle 1, and updates the admissible actions to include multiple “take” options for each item in addition to standard navigation/look/inventory actions. The agent then executes <action>take soapbar 1 from countertop 1</action>, and the environment confirms that soapbar 1 is now in hand (inventory implicitly includes soapbar 1), with admissible actions updated accordingly (e.g., examine soapbar 1, move soapbar 1 to countertop 1, navigate to other fixtures). Next, the agent executes <action>go to sinkbasin 1</action>; the observation states that sinkbasin 1 contains sink 1 and enables the cleaning affordance “clean soapbar 1 with sinkbasin 1,” alongside standard examination, movement, and navigation actions. The agent proceeds with <action>clean soapbar 1 with sinkbasin 1</action>, and the environment confirms that soapbar 1 is now cleaned; admissible actions continue to list cleaning/examine/move/navigation options. Finally, the agent executes <action>go to garbagecan 1</action>; the observation reports garbagecan 1 is present with no contents and lists admissible actions including “move soapbar 1 to garbagecan 1,” as well as examination and navigation options. At the end of the dialogue, the agent’s location is garbagecan 1; it is holding soapbar 1 (now cleaned); the environment has affirmed each transition (countertop → possession of soapbar → sinkbasin cleaning → garbagecan location); and the task goal state requires the cleaned soapbar to be placed into garbagecan 1, which is currently empty and reported as an available receptacle by the admissible action set.

### Conversation:
{conversation}

Please provide a detailed summary of the above conversation. Make sure to include all necessary details for the other model to understand the context so that it can make moves just like reading the full conversation. Do not provide any suggestions for the model to take actions, only summarize the conversation."""

compressor_prompt_v5 = """You are a helpful assistant that summarizes conversations for another model to use. You need to make sure that the summary captures all important details from the conversation so that the other model can understand the context just like reading the full conversation.

### Conversation:
{conversation}

Please provide a detailed summary of the above conversation. Make sure to include all necessary details for the other model to understand the context, especially entities, actions taken, and outcomes. But NEVER include any suggestions or recommendations for future actions—only summarize what has already occurred."""

compressor_prompt_v6 = """
        Your only task is to summarize the interaction history in detail. Include any task descriptions, previous attempts and thinkings. Do not add your own judgments, take actions, or answer the current query.

        **Interaction History:**
        {conversation_history}

        Summarize the interaction history in detail and output in the following format. Do not add your own judgments.
        <context>[brief summary]</context>
        """

class DataCollector:
    def __init__(self, args):
        self.config = args
        self.max_concurrency = self.config.max_concurrency
        self.max_executor_threads = self.config.max_executor_threads

        self._setup_endpoints()
    
    def _setup_endpoints(self):
        """
        Build a client pool from config.env.response_agent.address.
        """
        timeout_s = getattr(self.config, "timeout_s", 60)
        self._per_endpoint_concurrency = getattr(self.config, "per_endpoint_concurrency", None)

        # Use AsyncOpenAI clients pointed at each base_url
        self._clients = [AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))]

        self._timeout_s = timeout_s
        self._rr = itertools.cycle(range(len(self._clients)))  # round-robin cursor
        
    def get_batch_response_from_memory(
        self,
        memory_contexts: List[Dict[str, str]],
        extract_action_func=None,
    ):
        """
        get responses from the memory agent for a batch of inputs
        """

        async def process_all():
            batch_size = len(memory_contexts)
            # Global cap across ALL endpoints (keeps your original behavior)
            sem_global = asyncio.Semaphore(self.max_concurrency)

            # Optional per-endpoint caps (parallelize across gateways)
            ep_sems = None
            if self._per_endpoint_concurrency is not None:
                ep_sems = [asyncio.Semaphore(self._per_endpoint_concurrency) for _ in self._clients]
        
            async def _invoke_with_failover(order, invoker):
                """
                order: list of endpoint indices to try in sequence
                invoker: async function(client) -> response
                """
                last_exc = None
                for i in order:
                    client = self._clients[i]
                    try:
                        if ep_sems:
                            async with ep_sems[i]:
                                return await asyncio.wait_for(invoker(client), timeout=self._timeout_s)
                        else:
                            return await asyncio.wait_for(invoker(client), timeout=self._timeout_s)
                    except Exception as e:
                        last_exc = e
                        continue
                # If we get here, all endpoints failed
                raise last_exc or RuntimeError("All endpoints failed")
        
            async def process_item(idx: int):
                chat = []
                turns_to_summarize = memory_contexts[idx][:-2] if self.config.use_last_assistant_response else memory_contexts[idx]
                turn_len = len(turns_to_summarize)
                # randomly sample in the middle
                if turn_len > 2:
                    end_idx = np.random.randint(2, turn_len)
                    turns_to_summarize = turns_to_summarize[:end_idx]
                # for context in turns_to_summarize:
                #     if context["assistant"] is not None and context["assistant"] != "":
                #         # extract action from assistant response if needed
                #         if extract_action_func is not None:
                #             action = extract_action_func(context["assistant"])
                #         else:
                #             action = context["assistant"]
                #         chat.append({"role": "assistant", "content": action})
                #     if context["user"] is not None and context["user"] != "":
                #         chat.append({"role": "user", "content": context["user"]})
                chat = turns_to_summarize
                conv_string = "\n".join(
                        [f"{item['role']}: {item['content']}" for item in chat]
                    )

                compressor_prompt_version = self.config.compressor_prompt_version
                if compressor_prompt_version == 'v1':
                    format_prompt = compressor_prompt_v1
                elif compressor_prompt_version == 'v2':
                    format_prompt = compressor_prompt_v2
                elif compressor_prompt_version == 'v3':
                    format_prompt = compressor_prompt_v3
                elif compressor_prompt_version == 'v4':
                    format_prompt = compressor_prompt_v4
                elif compressor_prompt_version == 'v5':
                    format_prompt = compressor_prompt_v5
                elif compressor_prompt_version == 'v6':
                    format_prompt = compressor_prompt_v6
                else:
                    raise ValueError(f"Unsupported compressor_prompt_version: {compressor_prompt_version}")
                prompt_with_conv = format_prompt.format(conversation=conv_string) + "\nExpected to be less than 1000 words."
            
                async with sem_global:
                    # Choose where to start (round-robin), then try all for failover
                    n = len(self._clients)
                    start = next(self._rr)
                    order = [(start + k) % n for k in range(n)]

                    async def _call(client):
                        return await client.chat.completions.create(
                            model=self.config.model_name,
                            messages=[{"role": "user", "content": prompt_with_conv}],
                            n=1,
                        )

                    response = await _invoke_with_failover(order, _call)
                
                response_text = response.choices[0].message.content.strip()
                logging.info(f"Response for item {idx}: {response}")

                return idx, response_text, prompt_with_conv

            results = await asyncio.gather(*(process_item(i) for i in range(batch_size)))
            results.sort(key=lambda x: x[0])
            responses = [r[1] for r in results]
            prompts = [r[2] for r in results]
            return responses, prompts
    
        # Create or fetch the loop, and (optionally) cap its thread pool
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.max_executor_threads is not None:
            loop.set_default_executor(ThreadPoolExecutor(max_workers=self.max_executor_threads))

        return loop.run_until_complete(process_all())


def add_dataset_name(example):
    iid = example.get("item_id", "")
    name = iid.split("_", 1)[0] if isinstance(iid, str) and "_" in iid else ("unknown" if isinstance(iid, str) else "unknown")
    return {"dataset_name": name}

def take_group(g):
    n = min(targets.get(g.name, 0), len(g))
    return g.sample(n=n, random_state=42) if n > 0 else g.iloc[0:0]

def downsample_by_instance(df, target=2000, seed=42):
    """
    Proportional downsampling within each instance_id group to total `target`.
    - Allocates per-group quotas via largest-remainder method.
    - Samples that many rows from each group.
    - Returns a HF Dataset with exactly `target` rows (or the dataset size if smaller).
    """
    rng = np.random.RandomState(seed)

    N = len(df)
    if N <= target:
        # Nothing to downsample
        return Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)

    sizes = df.groupby("instance_id").size().rename("size")
    total = sizes.sum()  # == N

    # Ideal proportional shares
    ideal = sizes * (target / total)

    # Initial floor allocation, capped by group size
    alloc = np.floor(ideal).astype(int)
    alloc = np.minimum(alloc, sizes)

    # Distribute remaining using largest remainders, only to groups with capacity left
    remaining = target - int(alloc.sum())
    if remaining > 0:
        remainder = (ideal - alloc).astype(float)

        # Groups with spare capacity
        capacity_left = (sizes - alloc) > 0
        # Stable, deterministic tie-breaking: sort by (remainder desc, instance_id)
        order = (
            pd.DataFrame({
                "remainder": remainder,
                "iid": remainder.index,
                "cap": capacity_left
            })
            .query("cap == True")
            .sort_values(["remainder", "iid"], ascending=[False, True])
            .iid.values
        )

        give = min(remaining, capacity_left.sum())
        bump_iids = order[:give]
        alloc.loc[bump_iids] += 1

        # If still short (e.g., many groups capped), keep cycling through capacity
        remaining = target - int(alloc.sum())
        if remaining > 0:
            # Build a pool of rows not yet allocated
            extra_pool = sizes - alloc
            while remaining > 0 and (extra_pool > 0).any():
                # deterministic round-robin over available groups
                for iid in extra_pool[extra_pool > 0].index:
                    alloc.loc[iid] += 1
                    extra_pool.loc[iid] -= 1
                    remaining -= 1
                    if remaining == 0:
                        break

    # Final guard: if overshot due to edge-casing, trim back (rare)
    overshoot = int(alloc.sum()) - target
    if overshoot > 0:
        # remove from smallest remainders first
        take_back_order = (
            pd.DataFrame({"rem": (ideal - (alloc - 1)).clip(lower=0), "iid": alloc.index})
            .sort_values(["rem", "iid"], ascending=[True, True])
            .iid.values
        )
        for iid in take_back_order:
            if overshoot == 0:
                break
            if alloc.loc[iid] > 0:
                alloc.loc[iid] -= 1
                overshoot -= 1

    # Sample within each group according to alloc
    groups = dict(tuple(df.groupby("instance_id", sort=False)))
    parts = []
    for iid, k in alloc.items():
        if k <= 0: 
            continue
        g = groups[iid]
        # deterministic per-group seed
        rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
        parts.append(g.sample(n=int(k), random_state=rs))

    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # Ensure exact size (belt-and-suspenders)
    if len(out) > target:
        out = out.iloc[:target].copy()
    elif len(out) < target:
        # should not happen, but fallback: top up from remaining rows proportionally
        remaining_df = df.drop(out.index)
        need = target - len(out)
        out = pd.concat(
            [out, remaining_df.sample(n=need, random_state=seed)],
            axis=0
        ).reset_index(drop=True)

    return Dataset.from_pandas(out, preserve_index=False)


# =========================
# Normalization map helpers
# =========================
def _normalize_message_role(role: str) -> str:
    if not isinstance(role, str):
        return "assistant"
    r = role.strip().lower()
    if r in {"user", "human"}:
        return "user"
    if r in {"assistant", "gpt", "ai", "model"}:
        return "assistant"
    if r == "system":
        return "system"
    return r or "assistant"


def _first_nonempty(d: dict, keys: list[str]) -> str:
    for k in keys:
        v = d.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def preprocess_agentgym_conversations(example: dict) -> dict:
    """
    Normalize AgentGym `conversations` to a list of {role, content}.
    Handles variations like {from/value} or {role/content/text} and stringified JSON.
    Rewrites the `conversations` field in-place to the normalized schema.
    """
    conv = example.get("conversations")
    cleaned = []

    # Try to decode stringified JSON
    if isinstance(conv, str):
        try:
            loaded = json.loads(conv)
            conv = loaded
        except Exception:
            # treat as a single message
            conv = [{"role": "user", "content": conv}]

    if isinstance(conv, list):
        for item in conv:
            if not isinstance(item, dict):
                # fallback to string repr
                msg = str(item)
                if msg.strip():
                    cleaned.append({"role": "assistant", "content": msg.strip()})
                continue

            role = _normalize_message_role(
                _first_nonempty(item, ["role", "from"]) or "assistant"
            )
            content = _first_nonempty(item, ["content", "value", "text", "message"]) or ""
            if content:
                cleaned.append({"role": role, "content": content})

    elif isinstance(conv, dict):
        # Some datasets store a single turn as dict
        role = _normalize_message_role(_first_nonempty(conv, ["role", "from"]))
        content = _first_nonempty(conv, ["content", "value", "text", "message"]) or ""
        if content:
            cleaned.append({"role": role or "assistant", "content": content})

    # If still empty, provide a minimal placeholder to avoid empty lists downstream
    if not cleaned:
        cleaned = [{"role": "system", "content": ""}]

    return {"conversations": cleaned}


def preprocess_sweagent_trajectories(example: dict) -> dict:
    """
    Normalize SWE-agent `trajectories` to a list of {role, content} messages.
    Heuristics: for each step, try common user-like and assistant-like keys.
    Rewrites the `trajectories` field in-place to the normalized schema.
    """
    # SWE-agent dataset uses "trajectory" (singular), not "trajectories" (plural)
    traj = example.get("trajectory") or example.get("trajectories")
    messages: list[dict] = []

    # If it's a string, try to parse JSON
    if isinstance(traj, str):
        try:
            traj = json.loads(traj)
        except Exception:
            # store as a single assistant message
            return {"trajectories": [{"role": "assistant", "content": traj.strip()}]}

    # If it's already a message list with role/content
    if isinstance(traj, list) and traj and isinstance(traj[0], dict) and {"role", "content"}.issubset(traj[0].keys()):
        # ensure normalized roles and stripped content
        for m in traj:
            role = _normalize_message_role(m.get("role"))
            content = (m.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
        return {"trajectories": messages or [{"role": "system", "content": ""}]}

    # Common SWE-agent variant: list of dicts with role + text/system_prompt
    if isinstance(traj, list) and traj and isinstance(traj[0], dict) and ("role" in traj[0]):
        for step in traj:
            if not isinstance(step, dict):
                continue
            role = _normalize_message_role(step.get("role"))
            # Prefer normal content keys; for system messages, fall back to system_prompt if text is empty
            content = _first_nonempty(step, ["content", "text", "message", "value"]) or ""
            if role == "system" and not content:
                content = (step.get("system_prompt") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
        if not messages:
            messages = [{"role": "system", "content": ""}]
        return {"trajectories": messages}

    # Generic step-wise extraction
    if isinstance(traj, list):
        user_keys = [
            "user", "question", "prompt", "input", "query", "instruction",
            "task", "issue", "human"
        ]
        assistant_keys = [
            "assistant", "response", "output", "final_answer", "answer",
            "thoughts", "message", "tool_output", "ai", "gpt"
        ]

        for step in traj:
            if not isinstance(step, dict):
                # fallback: treat as assistant text
                msg = str(step)
                if msg.strip():
                    messages.append({"role": "assistant", "content": msg.strip()})
                continue

            # extract user-like
            user_text = _first_nonempty(step, user_keys)
            if user_text:
                messages.append({"role": "user", "content": user_text})

            # extract assistant-like
            asst_text = _first_nonempty(step, assistant_keys)
            if asst_text:
                messages.append({"role": "assistant", "content": asst_text})

        if not messages:
            messages = [{"role": "system", "content": ""}]
        return {"trajectories": messages}

    if isinstance(traj, dict):
        # Some SWE-agent variants keep a conversation-like structure under 'messages'
        msgs = traj.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = _normalize_message_role(m.get("role"))
                content = _first_nonempty(m, ["content", "text", "message", "value"]) or ""
                if content:
                    messages.append({"role": role or "assistant", "content": content})
        if not messages:
            messages = [{"role": "system", "content": ""}]
        return {"trajectories": messages}

    # Unknown structure fallback
    return {"trajectories": [{"role": "system", "content": ""}]}

def train_test_split(dataset: Dataset, test_size: float = 0.1, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split a HuggingFace Dataset into train and test sets.
    """
    # Shuffle the dataset
    shuffled_dataset = dataset.shuffle(seed=seed)
    # Calculate split index
    split_index = int(len(shuffled_dataset) * (1 - test_size))
    # Create train and test datasets
    train_dataset = shuffled_dataset.select(range(split_index))
    test_dataset = shuffled_dataset.select(range(split_index, len(shuffled_dataset)))
    return train_dataset, test_dataset

def save_parquet(dataset: Dataset, out_path: str):
    """
    Save a HuggingFace Dataset to Parquet format.
    """
    dataset.to_parquet(out_path)
    print(f"Saved dataset to {out_path}")

def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of records."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def parse_turns(raw: str) -> List[Dict[str, str]]:
    """
    Return a list of {"role": "user"|"assistant", "content": "<block>"}.
    - Tolerates inputs wrapped like: `"prompt": "user\\n...\\nassistant\\n..."`
    - Splits on lines that are exactly `user` or `assistant` (case-insensitive).
    """
    text = raw

    # If wrapped in a JSON string field "prompt": "<escaped>", unescape it
    m = re.search(r'"prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.S)
    if m:
        try:
            text = json.loads(f'"{m.group(1)}"')
        except Exception:
            text = m.group(1).replace("\\n", "\n")

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Split into [preamble, role1, block1, role2, block2, ...]
    parts = re.split(r'(?mi)^\s*(user|assistant)\s*$', text)
    if len(parts) < 3:
        return []

    turns = []
    # Skip parts[0] (preamble before first role marker)
    for i in range(1, len(parts) - 1, 2):
        role = parts[i].strip().lower()
        block = parts[i + 1].strip()
        if role in ("user", "assistant"):
            turns.append({"role": role, "content": block})
    if turns[-1]["content"] == "": # Remove trailing empty block
        turns = turns[:-1]
    return turns

if __name__ == "__main__":
    # ========= Setup DataCollector =========
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-5-mini", help="Model name for summarization")
    parser.add_argument("--max_concurrency", type=int, default=32, help="Max concurrency across all endpoints")
    parser.add_argument("--compressor_prompt_version", type=str, default="v5", help="Compressor prompt version to use")
    parser.add_argument("--use_last_assistant_response", action="store_true", help="Whether to exclude the last assistant response from the summary input")
    parser.add_argument("--max_executor_threads", type=int, default=16, help="Max threads for async executor")
    parser.add_argument("--timeout_s", type=int, default=60, help="Timeout per request in seconds")
    parser.add_argument("--per_endpoint_concurrency", type=int, default=16, help="Max concurrency per endpoint")
    parser.add_argument("--save_parquet", action="store_true", help="Whether to save the final dataset as parquet")
    args = parser.parse_args()

    # ========= Load AgentGym/AgentTraj-L dataset =========
    print("Loading AgentGym/AgentTraj-L dataset...")
    agentgym_dataset = load_dataset("AgentGym/AgentTraj-L", split="train")
    agentgym_dataset = agentgym_dataset.map(add_dataset_name, num_proc=4, desc="Adding dataset names")
    agentgym_dataset = agentgym_dataset.to_pandas()

    counts = pd.Series(agentgym_dataset["dataset_name"]).value_counts().to_dict()
    print("Counts per dataset_name:", counts)

    targets = {
        "webshop": 2170,
        "sqlgym": 1657,
        "pick": 1228,
        "sciworld": 1171,
        "lmrlgym": 646,
        "babyai": 447,
        "textcraft": 207,
        "weather": 172,
        "movie": 119,
        "look": 108,
        "todo": 75,
    }

    down_df = (
        agentgym_dataset.groupby("dataset_name", group_keys=False)
        .apply(take_group)
        .reset_index(drop=True)
    )

    assert len(down_df) == 8000, f"Got {len(down_df)} rows, expected 8000"
    print(down_df["dataset_name"].value_counts())

    agentgym_dataset = Dataset.from_pandas(down_df, preserve_index=False)
    # Normalize conversations
    agentgym_dataset = agentgym_dataset.map(
        preprocess_agentgym_conversations,
        desc="Normalizing AgentGym conversations",
        num_proc=4,
    )


    # ========= Load SWE-agent-trajectories dataset =========
    print("Loading SWE-agent-trajectories dataset...")
    sweagent_dataset = load_dataset("nebius/SWE-agent-trajectories", split="train")
    df = sweagent_dataset.to_pandas()
    sweagent_dataset = downsample_by_instance(df, target=800, seed=42)
    # Normalize trajectories
    sweagent_dataset = sweagent_dataset.map(
        preprocess_sweagent_trajectories,
        desc="Normalizing SWE-agent trajectories",
        num_proc=4,
    )

    # ======== Load MiroVerse-v0.1 dataset =========
    print("Loading MiroVerse-v0.1 dataset...")
    deepresearch = load_dataset("miromind-ai/MiroVerse-v0.1", "MiroVerse-WebWalkerQA-Silver", split="train")
    # remove the split column
    deepresearch = deepresearch.remove_columns(['split'])
    # add item_id column
    deepresearch = deepresearch.map(
        lambda ex, idx: {"item_id": f"miroverse_webwalkerqa_{idx}"},
        with_indices=True,
        desc="Adding item_id to MiroVerse rows",
    )
    # unify schema: rename messages -> conversations
    deepresearch = deepresearch.rename_column("messages", "conversations")
    # downsample to 2000 examples
    deepresearch = deepresearch.shuffle(seed=42).select(range(2000))

    # # ======== Load some of alfworld, webshop and sciworld datasets from Qwen3-32B =========
    # rollouts = []
    # for folder in ["alfworld_29steps_16bs", "sciworld_25steps_32bs", "webshop_small_20steps_16bs"]:
    #     data = load_jsonl_file(f"outputs/rollouts/qwen3-32b/{folder}/rollouts_00000.jsonl")
    #     for idx, item in enumerate(data):
    #         item_id = f"{folder}_{idx:05d}"
    #         conversations = parse_turns(item.get("prompt", ""))
    #         rollouts.append({"item_id": item_id, "conversations": conversations})

    # ========= Prepare column subsets =========
    # Keep only conversations (and item_id) from AgentGym
    agentgym_keep_cols = [c for c in ["item_id", "conversations"] if c in agentgym_dataset.column_names]
    agentgym_dataset = agentgym_dataset.select_columns(agentgym_keep_cols)

    # Keep only trajectories from SWE-agent and add item_id like dataset_idx
    # Assumption: We standardize dataset prefix as "sweagent" and use the row index as idx starting at 0.
    sweagent_dataset = sweagent_dataset.map(
        lambda ex, idx: {"item_id": f"sweagent_{idx}"},
        with_indices=True,
        desc="Adding item_id to SWE-agent rows",
    )
    # Unify schema: rename trajectories -> conversations, then keep same columns as AgentGym
    if "trajectories" in sweagent_dataset.column_names:
        sweagent_dataset = sweagent_dataset.rename_column("trajectories", "conversations")
    swe_keep_cols = [c for c in ["item_id", "conversations"] if c in sweagent_dataset.column_names]
    sweagent_dataset = sweagent_dataset.select_columns(swe_keep_cols)

    # ========= Combine datasets =========
    combined = concatenate_datasets([agentgym_dataset, sweagent_dataset])
    # Ensure unified column naming
    assert "conversations" in combined.column_names, "Unified 'conversations' column missing in combined dataset"
    assert "trajectories" not in combined.column_names, "Found unexpected 'trajectories' column after renaming"

    # Basic stats and schema preview
    print({
        "agentgym_rows": len(agentgym_dataset),
        "sweagent_rows": len(sweagent_dataset),
        "combined_rows": len(combined),
        "combined_columns": combined.column_names,
    })

    # Optional: save to disk (commented out by default)
    # out_dir = os.path.join("outputs", "data_preprocess", "agentgym_sweagent_combined")
    # os.makedirs(out_dir, exist_ok=True)
    # combined.to_json(os.path.join(out_dir, "combined.jsonl"), lines=True, orient="records", force_ascii=False)
    # print(f"Saved combined dataset to {out_dir}")

    # downsample to 4000 examples for SFT training
    combined_sampled = combined.shuffle(seed=42).select(range(3000))

    # ========= Generate summaries with checkpoint support =========
    def generate_summaries_with_checkpoints(
        dataset,
        collector: DataCollector,
        checkpoint_dir="outputs/checkpoints",
        batch_size=50,
        output_file="outputs/data_with_summaries.jsonl"
    ):
        """
        Generate summaries for conversations with checkpoint support.

        Args:
            dataset: HuggingFace Dataset with 'item_id' and 'conversations' columns
            collector: DataCollector instance for generating summaries
            checkpoint_dir: Directory to save intermediate checkpoints
            batch_size: Number of items to process before saving checkpoint
            output_file: Final output file path

        Returns:
            List of dicts with {item_id, conversations, summaries}
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

        checkpoint_file = os.path.join(checkpoint_dir, "processed_items.jsonl")

        # Load existing checkpoint if available
        processed_items = {}
        if os.path.exists(checkpoint_file):
            print(f"Loading checkpoint from {checkpoint_file}...")
            with open(checkpoint_file, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    processed_items[item['item_id']] = item
            print(f"Loaded {len(processed_items)} previously processed items")

        # Prepare items to process
        all_items = []
        items_to_process = []

        for idx, example in enumerate(dataset):
            item_id = example.get('item_id', f'item_{idx}')
            conversations = example.get('conversations', [])

            if item_id in processed_items:
                # Already processed, use cached result
                all_items.append(processed_items[item_id])
            else:
                # Need to process
                items_to_process.append({
                    'index': len(all_items),
                    'item_id': item_id,
                    'conversations': conversations
                })
                all_items.append(None)  # placeholder

        print(f"Total items: {len(dataset)}")
        print(f"Already processed: {len(processed_items)}")
        print(f"Items to process: {len(items_to_process)}")

        if len(items_to_process) == 0:
            print("All items already processed!")
            # Write final output
            with open(output_file, 'w') as f:
                for item in all_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Final output saved to {output_file}")
            train_dataset, test_dataset = train_test_split(Dataset.from_list(all_items), test_size=0.05, seed=42)
            print(f"Saving train/test split: {len(train_dataset)} train, {len(test_dataset)} test")
            parquet_path = output_file.replace(".jsonl", ".parquet")
            save_parquet(train_dataset, parquet_path.replace(".parquet", "_train.parquet"))
            save_parquet(test_dataset, parquet_path.replace(".parquet", "_test.parquet"))
            print(f"✓ Parquet files saved.")
            return all_items

        # Process in batches
        num_batches = (len(items_to_process) + batch_size - 1) // batch_size

        with open(checkpoint_file, 'a') as checkpoint_f:
            for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(items_to_process))
                batch = items_to_process[start_idx:end_idx]

                print(f"\nProcessing batch {batch_idx + 1}/{num_batches} ({len(batch)} items)...")

                # Convert conversations to memory context format
                memory_contexts = [item['conversations'] for item in batch]

                # Generate summaries
                try:
                    summaries, prompts = collector.get_batch_response_from_memory(memory_contexts)

                    # Save results
                    for i, item in enumerate(batch):
                        result = {
                            'item_id': item['item_id'],
                            'conversations': item['conversations'],
                            'summary': summaries[i],
                            'prompt': prompts[i],
                            'model_name': args.model_name,
                        }

                        # Update in all_items
                        all_items[item['index']] = result

                        # Append to checkpoint file immediately
                        checkpoint_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        checkpoint_f.flush()  # Ensure it's written to disk

                    print(f"Batch {batch_idx + 1} completed and saved to checkpoint")

                except Exception as e:
                    print(f"Error processing batch {batch_idx + 1}: {e}")
                    print("Progress has been saved. You can resume by running the script again.")
                    raise

        # Write final output
        print(f"\nWriting final output to {output_file}...")
        with open(output_file, 'w') as f:
            for item in all_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"✓ All {len(all_items)} items processed successfully!")
        print(f"✓ Output saved to {output_file}")

        train_dataset, test_dataset = train_test_split(Dataset.from_list(all_items), test_size=0.05, seed=42)
        print(f"Saving train/test split: {len(train_dataset)} train, {len(test_dataset)} test")
        parquet_path = output_file.replace(".jsonl", ".parquet")
        save_parquet(train_dataset, parquet_path.replace(".parquet", "_train.parquet"))
        save_parquet(test_dataset, parquet_path.replace(".parquet", "_test.parquet"))
        print(f"✓ Parquet files saved.")

        return all_items
    
    # Example usage (commented out - uncomment to run):
    collector = DataCollector(args)
    #
    results = generate_summaries_with_checkpoints(
        dataset=combined_sampled,
        collector=collector,
        checkpoint_dir="outputs/summary_checkpoints",
        batch_size=32,  # Process 50 items at a time
        output_file="outputs/data_with_summaries.jsonl"
    )
