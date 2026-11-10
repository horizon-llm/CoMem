import asyncio
import itertools
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import litellm
import numpy as np
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from agent_system.environments import EnvironmentManagerBase
from agent_system.environments.env_package.tools import (
    execute_bash_tool,
    file_editor,
    finish_tool,
    r2egym_bash_execute_tool,
    search_tool,
    str_replace_editor_tool,
    submit_tool,
)
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.multi_turn_rollout.swe_function_match_v2 import (
    Action,
    action_consistency_reward_detailed,
    parse_action,
)
from agent_system.multi_turn_rollout.utils import to_list_of_dict
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


class SumAPIRespRecentLiteLLMTrajectoryCollector(TrajectoryCollector):
    """Collects summarization rollouts from the memory actor, queries the frozen
    policy agent via async LiteLLM endpoints with round-robin failover, and
    evaluates functional action-consistency rewards.
    """

    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_concurrency = config.env.response_agent.max_concurrency
        self.max_executor_threads = config.env.response_agent.max_executor_threads
        self._setup_endpoints()

    def _setup_endpoints(self):
        """Configure LiteLLM and round-robin endpoint pool for policy agent calls."""
        model_name = self.config.env.response_agent.model_name
        self.model_name = model_name
        self._timeout_s = getattr(self.config.env.response_agent, "timeout_s", 60)
        self._per_endpoint_concurrency = getattr(
            self.config.env.response_agent, "per_endpoint_concurrency", None
        )

        # Disable LiteLLM's async logging & callbacks to prevent worker timeouts
        litellm.drop_params = True
        litellm.set_verbose = False
        litellm.success_callback = []
        litellm.failure_callback = []

        self.using_local = "openai/" in model_name or "hosted_vllm" in model_name
        if self.using_local:
            addrs = getattr(self.config.env.response_agent, "address", None) or []
            if not addrs:
                raise ValueError(
                    f"Model '{model_name}' appears to be vLLM-hosted, but no addresses "
                    "provided in config.env.response_agent.address"
                )
            self._endpoints = addrs
            self._rr = itertools.cycle(range(len(self._endpoints)))
            litellm.api_key = None
            logger.info(
                f"Using vLLM endpoints with model: {model_name}, addresses: {addrs}"
            )
        else:
            self._endpoints = [None]
            self._rr = itertools.cycle([0])
            logger.info(f"Using cloud API with model: {model_name}")

    # ------------------------------------------------------------------
    # Prompt Preprocessing
    # ------------------------------------------------------------------
    def _get_retain_turns(self) -> int:
        """Return how many recent turns should be kept verbatim alongside the summary."""
        if not getattr(self.config.env.rollout, "use_last_assistant_response", False):
            return 0
        retain_turns = getattr(self.config.env.rollout, "retain_most_turns", 1)
        try:
            retain_turns = int(retain_turns)
        except (TypeError, ValueError):
            retain_turns = 1
        return max(1, retain_turns)

    @staticmethod
    def _format_steps(steps: List[Dict[str, str]], leading_obs_newline: bool = False) -> str:
        """Format a list of trajectory step dicts into THOUGHT / ACTION / OBSERVATION text."""
        obs_prefix = "\nOBSERVATION:" if leading_obs_newline else "OBSERVATION:"
        chunks = []
        for step in steps:
            chunks.append(
                f"THOUGHT:\n```\n{step['thought']}\n```\n"
                f"ACTION:\n```\n{step['action']}\n```\n"
                f"{obs_prefix}\n```\n{step['observation']}\n```\n"
            )
        return "".join(chunks)

    def _prepare_history_message(
        self,
        trajectory_steps: List[Dict[str, str]],
        system_prompt: str,
        user_prompt: str,
        retain_most_turns: int = 1,
    ) -> Tuple[str, str]:
        """Split trajectory steps into (history_to_compress, history_to_append)."""
        if len(trajectory_steps) <= retain_most_turns - 1:
            return "", ""

        header = (
            f"SYSTEM PROMPT:\n```\n{system_prompt}\n```\n"
            f"INITIAL USER PROMPT:\n```\n{user_prompt}\n```\n"
        )
        num_compress = len(trajectory_steps) - retain_most_turns
        compress_steps = (
            trajectory_steps[:num_compress] if num_compress > 0 else trajectory_steps
        )
        history_to_compress = header + self._format_steps(
            compress_steps, leading_obs_newline=True
        )
        history_to_append = self._format_steps(
            trajectory_steps[-retain_most_turns:], leading_obs_newline=False
        )
        return history_to_compress, history_to_append

    def preprocess_single_sample(self, item: int, gen_batch: DataProto) -> Dict[str, Any]:
        """Slice a single trajectory sample into a compressor prompt and target action."""
        data_source = gen_batch.non_tensor_batch["data_source"][item]
        tools_kwargs = gen_batch.non_tensor_batch["tools_kwargs"][item]
        trajectory_steps = tools_kwargs["trajectory_steps"]
        system_prompt = tools_kwargs["system_prompt"]
        user_prompt = tools_kwargs["user_prompt"]

        retain_turns = self._get_retain_turns()
        history_to_compress, history_to_append = self._prepare_history_message(
            trajectory_steps[:-1], system_prompt, user_prompt, retain_turns
        )
        expected_action = trajectory_steps[-1]["action"]

        if history_to_compress == "":
            raise ValueError("No history to compress for summarization.")

        prompt_with_conv = (
            "You are a helpful assistant that summarizes conversations for another model to use. "
            "You need to make sure that the summary captures all important details from the conversation "
            "so that the other model can understand the context just like reading the full conversation.\n\n"
            "### Conversation:\n"
            f"{history_to_compress}\n\n"
            "Please provide a detailed summary of the above conversation. Make sure to include all necessary details "
            "for the other model to understand the context, especially entities, actions taken, and outcomes. "
            "But NEVER include any suggestions or recommendations for future actions—only summarize what has already occurred."
        )
        chat = np.array([{"role": "user", "content": prompt_with_conv}])
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict = {
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": self.tokenizer.encode(
                prompt_with_chat_template, add_special_tokens=False
            ),
            "index": item,
            "data_source": data_source,
            "turns": trajectory_steps,
            "history_to_append": history_to_append,
            "expected_actions": expected_action,
            "system_prompt": system_prompt,
        }
        if self.config.data.get("return_raw_chat", False):
            row_dict["raw_prompt"] = chat.tolist()

        return row_dict

    def preprocess_batch(self, gen_batch: DataProto) -> DataProto:
        """Convert a batch of trajectory prefixes into tokenized compressor prompts."""
        batch_size = len(gen_batch.batch["input_ids"])
        processed_samples = [
            self.preprocess_single_sample(item=i, gen_batch=gen_batch)
            for i in range(batch_size)
        ]
        return DataProto.from_single_dict(
            data=collate_fn(processed_samples), meta_info=gen_batch.meta_info
        )

    # ------------------------------------------------------------------
    # Policy Agent Querying
    # ------------------------------------------------------------------
    def _build_agent_query_prompt(
        self,
        compression_text: str,
        history_to_append: str,
        turns: List[Dict[str, Any]],
        retain_turns: int,
    ) -> str:
        """Construct the compact user prompt combining summary and recent turns."""
        if retain_turns > 0:
            formatted_prompt = (
                "You are a helpful assistant. Based on the summarized conversation history and recent detailed turns, "
                "respond to the latest user message in the recent detailed turns.\n"
                "## Conversation Summary:\n"
                f"{compression_text}\n"
                "## Recent Detailed Turns:\n"
                f"{history_to_append}\n"
            )
        else:
            obs_text = turns[-2]["observation"]
            formatted_prompt = (
                "You are a helpful assistant. Based on the summarized conversation history, "
                "generate a response to the user's query.\n"
                "## Conversation History:\n"
                f"{compression_text}\n"
                "## Current User Query:\n"
                f"{obs_text}"
            )

        max_steps = self.config.env.swebench.get("max_steps", 40)
        steps_remaining = max_steps - len(turns) + 1
        stepcount_message = (
            f"Steps Remaining: {steps_remaining}"
            if steps_remaining > 0
            else "You have reached the maximum number of steps. Please submit your answer NOW."
        )
        return f"{formatted_prompt}\n{stepcount_message}\n"

    @staticmethod
    def _prepare_tools_and_caching(
        messages: List[Dict[str, str]],
        use_tool_call: bool,
        scaffold: str,
        llm_name: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Select scaffold tool definitions and attach Anthropic cache_control markers."""
        if not use_tool_call:
            return None

        if scaffold == "r2egym":
            tools = [search_tool, file_editor, r2egym_bash_execute_tool, finish_tool]
        elif scaffold in ("openhands", "sweagent"):
            tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]
        else:
            tools = None

        if tools and "vertex" not in llm_name.lower():
            tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
            breakpoints_remaining = 3
            for message in reversed(messages):
                if message["role"] in ("user", "tool"):
                    if breakpoints_remaining > 0:
                        message["cache_control"] = {"type": "ephemeral"}
                        breakpoints_remaining -= 1
                    else:
                        break
        return tools

    async def _call_agent_with_failover(
        self,
        system_prompt: str,
        formatted_prompt: str,
        use_tool_call: bool,
        scaffold: str,
        ep_sems: Optional[List[asyncio.Semaphore]],
    ) -> Any:
        """Dispatch a single query across the round-robin endpoint ring with failover."""
        n = len(self._endpoints)
        start = next(self._rr) % n
        order = [(start + k) % n for k in range(n)]

        last_exc = None
        for endpoint_idx in order:
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_prompt},
                ]
                tools = self._prepare_tools_and_caching(
                    messages, use_tool_call, scaffold, self.model_name
                )
                call_params: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.config.env.response_agent.temperature,
                    "max_tokens": 4000,
                    "n": 1,
                    "timeout": self._timeout_s,
                    "tools": tools,
                }
                if not use_tool_call:
                    call_params["tool_choice"] = "none"
                    call_params["function_call"] = None
                if self.using_local and self._endpoints[endpoint_idx] is not None:
                    call_params["api_base"] = self._endpoints[endpoint_idx]

                coro = litellm.acompletion(**call_params)
                if ep_sems:
                    async with ep_sems[endpoint_idx]:
                        return await asyncio.wait_for(coro, timeout=self._timeout_s)
                return await asyncio.wait_for(coro, timeout=self._timeout_s)
            except Exception as e:
                last_exc = e
                continue

        raise last_exc or RuntimeError("All endpoints failed")

    def get_batch_response_from_agent(
        self, compressions: DataProto
    ) -> Tuple[List[Any], List[str], List[int], List[str]]:
        """Query the frozen policy agent concurrently for all compressed summaries in the batch."""
        use_tool_call = self.config.env.swebench.get("use_tool_call", False)
        scaffold = self.config.env.swebench.get("scaffold", "")
        retain_turns = self._get_retain_turns()
        batch_size = len(compressions.batch["input_ids"])

        async def process_all():
            sem_global = asyncio.Semaphore(self.max_concurrency)
            ep_sems = (
                [
                    asyncio.Semaphore(self._per_endpoint_concurrency)
                    for _ in self._endpoints
                ]
                if self._per_endpoint_concurrency is not None
                else None
            )

            async def process_item(idx: int):
                compression_ids = compressions.batch["responses"][idx]
                compression_text = self.tokenizer.decode(
                    compression_ids, skip_special_tokens=True
                )
                turns = compressions.non_tensor_batch.get("turns", [[]])[idx]
                history_to_append = compressions.non_tensor_batch.get(
                    "history_to_append", [""]
                )[idx]
                system_prompt = compressions.non_tensor_batch.get(
                    "system_prompt", [""]
                )[idx]

                formatted_prompt = self._build_agent_query_prompt(
                    compression_text, history_to_append, turns, retain_turns
                )
                try:
                    async with sem_global:
                        response = await self._call_agent_with_failover(
                            system_prompt=system_prompt,
                            formatted_prompt=formatted_prompt,
                            use_tool_call=use_tool_call,
                            scaffold=scaffold,
                            ep_sems=ep_sems,
                        )
                    completion_tokens = (
                        response.usage.completion_tokens
                        if hasattr(response, "usage")
                        else -1
                    )
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    response = None
                    completion_tokens = -1
                    formatted_prompt = ""

                return (
                    idx,
                    response,
                    compression_text,
                    completion_tokens,
                    formatted_prompt,
                )

            results = await asyncio.gather(
                *[process_item(i) for i in range(batch_size)]
            )
            results.sort(key=lambda x: x[0])
            responses = [r[1] for r in results]
            compression_texts = [r[2] for r in results]
            completion_tokens = [r[3] for r in results]
            formatted_prompts = [r[4] for r in results]
            return responses, compression_texts, completion_tokens, formatted_prompts

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.max_executor_threads is not None:
            loop.set_default_executor(
                ThreadPoolExecutor(max_workers=self.max_executor_threads)
            )

        return loop.run_until_complete(process_all())

    # ------------------------------------------------------------------
    # Reward Computation & Rollout Packaging
    # ------------------------------------------------------------------
    def _compute_batch_rewards(
        self,
        text_actions: List[Any],
        expected_actions: np.ndarray,
        completion_tokens: List[int],
    ) -> Tuple[np.ndarray, List[float], List[Action], List[str]]:
        """Evaluate action-consistency reward and length penalty for each rollout sample."""
        use_tool_call = self.config.env.swebench.get("use_tool_call", False)
        length_penalty_coef = self.config.env.swebench.get(
            "length_penalty_coefficient", 0.0
        )

        rewards: List[float] = []
        length_penalties: List[float] = []
        parsed_actions: List[Action] = []
        reward_details_list: List[str] = []

        for i in range(len(text_actions)):
            _, pred_action = parse_action(
                text_actions[i], use_tool_call=use_tool_call
            )
            gt_action = Action.from_string(expected_actions[i])
            reward, reward_details = action_consistency_reward_detailed(
                pred_action, gt_action
            )
            length_penalty = length_penalty_coef * completion_tokens[i]
            reward = max(reward - length_penalty, 0.0)

            rewards.append(reward)
            length_penalties.append(length_penalty)
            parsed_actions.append(pred_action)
            reward_details_list.append(json.dumps(reward_details))

        return (
            np.array(rewards, dtype=np.float32),
            length_penalties,
            parsed_actions,
            reward_details_list,
        )

    def gather_rollout_data(
        self,
        batch_list: List[Dict],
        rewards: np.ndarray,
        length_penalties: np.ndarray,
        traj_uid: np.ndarray,
    ) -> DataProto:
        """Package rollout trajectories and reward statistics into a DataProto."""
        episode_rewards_mean = np.mean(rewards)
        episode_rewards_min = np.min(rewards)
        episode_rewards_max = np.max(rewards)
        length_penalty_mean = np.mean(length_penalties)

        effective_batch = []
        for bs, data in enumerate(batch_list):
            assert (
                traj_uid[bs] == data["traj_uid"]
            ), "data is not from the same trajectory"
            data["episode_rewards"] = rewards[bs]
            data["episode_rewards_mean"] = episode_rewards_mean
            data["episode_rewards_min"] = episode_rewards_min
            data["episode_rewards_max"] = episode_rewards_max
            data["episode_lengths"] = np.int64(1)
            data["episode_lengths_mean"] = np.int64(1)
            data["episode_lengths_min"] = np.int64(1)
            data["episode_lengths_max"] = np.int64(1)
            data["length_penalties"] = length_penalties[bs]
            data["length_penalties_mean"] = length_penalty_mean
            effective_batch.append(data)

        return DataProto.from_single_dict(data=collate_fn(effective_batch))

    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        is_train: bool = True,
    ) -> Tuple[List[Dict], np.ndarray, np.ndarray, List[float]]:
        """Execute a full GRPO rollout step:
        1. Group-repeat prompts and generate candidate summaries from actor_rollout_wg.
        2. Query the frozen policy agent on each summary.
        3. Compute action-consistency rewards.
        """
        if is_train:
            gen_batch = gen_batch.repeat(
                repeat_times=self.config.env.rollout.n, interleave=True
            )

        batch_size = len(gen_batch.batch["input_ids"])
        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range( len(gen_batch.batch))], dtype=object)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)

        batch = self.preprocess_batch(gen_batch=gen_batch)
        expected_actions = batch.non_tensor_batch["expected_actions"]

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        for optional_key in ("raw_prompt", "tool_kwargs"):
            if optional_key in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append(optional_key)

        batch_input = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        batch_input.meta_info = gen_batch.meta_info

        # 1. Generate summary from memory model
        batch_output = actor_rollout_wg.generate_sequences(batch_input)
        batch = batch.union(batch_output)

        # 2. Generate induced action from frozen policy agent
        (
            text_actions,
            compression_texts,
            completion_tokens,
            formatted_prompts,
        ) = self.get_batch_response_from_agent(compressions=batch)

        # 3. Compute action-consistency rewards
        (
            rewards,
            length_penalties,
            parsed_actions,
            reward_details_list,
        ) = self._compute_batch_rewards(
            text_actions, expected_actions, completion_tokens
        )

        batch.non_tensor_batch["uid"] = uid_batch
        batch.non_tensor_batch["traj_uid"] = traj_uid
        batch.non_tensor_batch["rewards"] = rewards
        batch.non_tensor_batch["length_penalties"] = length_penalties
        batch.non_tensor_batch["reward_details"] = np.array(
            reward_details_list, dtype=object
        )
        batch.non_tensor_batch["text_actions"] = np.array(
            [a.to_xml_string() for a in parsed_actions], dtype=object
        )
        batch.non_tensor_batch["expected_actions"] = np.array(
            expected_actions, dtype=object
        )
        batch.non_tensor_batch["formatted_prompts"] = np.array(
            formatted_prompts, dtype=object
        )

        return to_list_of_dict(batch), rewards, traj_uid, length_penalties

    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        is_train: bool = True,
    ) -> DataProto:
        """Entry point called by RayPPOTrainer to collect summarized rollout batch."""
        batch_list, rewards, traj_uid, length_penalties = self.vanilla_multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            is_train=is_train,
        )
        return self.gather_rollout_data(
            batch_list=batch_list,
            rewards=rewards,
            traj_uid=traj_uid,
            length_penalties=length_penalties,
        )
