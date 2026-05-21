"""
This module implements the SumTrajectoryCollector class for collecting summarization trajectories
with support for retain_most_turns feature using litellm for LLM calls
"""

import json
import os
import torch
import numpy as np
import logging
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
from typing import List, Dict, Tuple
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
import asyncio
import itertools
from concurrent.futures import ThreadPoolExecutor
import litellm
# from agent_system.multi_turn_rollout.swe_function_match import action_consistency_reward
from agent_system.multi_turn_rollout.swe_function_match_v2 import action_consistency_reward, action_consistency_reward_detailed, parse_action, Action
from agent_system.environments.env_package.tools import (
    r2egym_bash_execute_tool,
    search_tool,
    file_editor,
    finish_tool,
    str_replace_editor_tool,
    execute_bash_tool,
    submit_tool,
)

logger = logging.getLogger(__name__)

# Predefined list of OpenAI model names
OPENAI_MODELS = ['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'gpt-5', 'gpt-5-mini']

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
In this ALFRED Embodied Environment episode, the agent begins in a bathroom-like room featuring countertop 1, drawers 1–8, garbagecan 1, handtowelholder 1, sinkbasins 1–2, toilet 1, toiletpaperhanger 1, and towelholder 1; the stated objective is to clean a soapbar and place it in the garbagecan. From the initial admissible actions list that includes navigating to each object, the agent first executes <action>go to countertop 1</action>. Upon arrival, the observation specifies that countertop 1 holds candle 1, candle 2, soapbar 1, soapbottle 1, soapbottle 2, and spraybottle 1, and updates the admissible actions to include multiple "take" options for each item in addition to standard navigation/look/inventory actions. The agent then executes <action>take soapbar 1 from countertop 1</action>, and the environment confirms that soapbar 1 is now in hand (inventory implicitly includes soapbar 1), with admissible actions updated accordingly (e.g., examine soapbar 1, move soapbar 1 to countertop 1, navigate to other fixtures). Next, the agent executes <action>go to sinkbasin 1</action>; the observation states that sinkbasin 1 contains sink 1 and enables the cleaning affordance "clean soapbar 1 with sinkbasin 1," alongside standard examination, movement, and navigation actions. The agent proceeds with <action>clean soapbar 1 with sinkbasin 1</action>, and the environment confirms that soapbar 1 is now cleaned; admissible actions continue to list cleaning/examine/move/navigation options. Finally, the agent executes <action>go to garbagecan 1</action>; the observation reports garbagecan 1 is present with no contents and lists admissible actions including "move soapbar 1 to garbagecan 1," as well as examination and navigation options. At the end of the dialogue, the agent's location is garbagecan 1; it is holding soapbar 1 (now cleaned); the environment has affirmed each transition (countertop → possession of soapbar → sinkbasin cleaning → garbagecan location); and the task goal state requires the cleaned soapbar to be placed into garbagecan 1, which is currently empty and reported as an available receptacle by the admissible action set.

### Conversation:
{conversation}

Please provide a detailed summary of the above conversation. Make sure to include all necessary details for the other model to understand the context so that it can make moves just like reading the full conversation. Do not provide any suggestions for the model to take actions, only summarize the conversation."""

compressor_prompt_v5 = """You are a helpful assistant that summarizes conversations for another model to use. You need to make sure that the summary captures all important details from the conversation so that the other model can understand the context just like reading the full conversation.

### Conversation:
{conversation}

Please provide a detailed summary of the above conversation. Make sure to include all necessary details for the other model to understand the context, especially entities, actions taken, and outcomes. But NEVER include any suggestions or recommendations for future actions—only summarize what has already occurred."""


class SumAPIRespRecentLiteLLMTrajectoryCollector(TrajectoryCollector):
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the SumAPIRespRecentLiteLLMTrajectoryCollector class using litellm for API calls.

        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_concurrency = config.env.response_agent.max_concurrency
        self.max_executor_threads = config.env.response_agent.max_executor_threads

        self._setup_endpoints()

    def _setup_endpoints(self):
        """
        Setup litellm configuration for API calls.
        """
        model_name = self.config.env.response_agent.model_name
        timeout_s = getattr(self.config.env.response_agent, "timeout_s", 60)
        self._per_endpoint_concurrency = getattr(self.config.env.response_agent, "per_endpoint_concurrency", None)

        # Store model configuration
        self.model_name = model_name
        self._timeout_s = timeout_s

        # Disable LiteLLM's async logging to prevent logging worker timeouts
        litellm.drop_params = True  # Drop unsupported params instead of raising errors
        litellm.set_verbose = False  # Disable verbose logging

        # Optionally, disable callbacks that can cause timeouts
        litellm.success_callback = []
        litellm.failure_callback = []

        # Check if using locally hosted models (vLLM)
        self.using_local = "openai/" in model_name or "hosted_vllm" in model_name

        if self.using_local:
            # For vLLM, get addresses and setup round-robin
            addrs = getattr(self.config.env.response_agent, "address", None) or []
            if not addrs:
                raise ValueError(f"Model '{model_name}' appears to be vLLM-hosted, but no addresses provided in config.env.response_agent.address")
            self._endpoints = addrs
            self._rr = itertools.cycle(range(len(self._endpoints)))  # round-robin cursor
            litellm.api_key = None  # Disable API key for local models
            logger.info(f"Using vLLM endpoints with model: {model_name}, addresses: {addrs}")
        else:
            # For OpenAI or other cloud models, use default endpoint
            self._endpoints = [None]  # None means use default litellm endpoint
            self._rr = itertools.cycle([0])
            logger.info(f"Using cloud API with model: {model_name}")

    def _get_retain_turns(self) -> int:
        """Return how many recent turns should be kept verbatim when summaries are used."""
        if not getattr(self.config.env.rollout, "use_last_assistant_response", False):
            return 0
        retain_turns = getattr(self.config.env.rollout, "retain_most_turns", 1)
        try:
            retain_turns = int(retain_turns)
        except (TypeError, ValueError):
            retain_turns = 1
        return max(1, retain_turns)

    @staticmethod
    def _chat_to_text(chat: List[Dict[str, str]]) -> str:
        """Convert a chat list into readable text lines."""
        lines: List[str] = []
        for turn in chat:
            if not turn.get("content"):
                continue
            role = turn.get("role", "user").capitalize()
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def _build_recent_turns_chat(
        self,
        turns: List[Dict[str, str]],
        retain_turns: int,
    ) -> List[Dict[str, str]]:
        """Build a short chat transcript from the most recent turns."""
        if retain_turns <= 0:
            return []

        # Get the last retain_turns turns
        recent_turns = turns[-retain_turns:] if len(turns) > retain_turns else turns

        return recent_turns

    def _context_to_chat(self, turns: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Convert turns list to chat format."""
        return turns

    def _prepare_history_message(self, trajectory_steps, system_prompt, user_prompt, retain_most_turns=1) -> str:
        """Prepare the agent history message from the trajectory.
        
        Different from base Agent, this method summarizes the conversation history to fit within context length.
        """
        if len(trajectory_steps) <= retain_most_turns - 1:
            return "", ""
        
        history_to_compress = ""
        # include system prompt
        history_to_compress = f"SYSTEM PROMPT:\n```\n{system_prompt}\n```\n"
        history_to_compress += f"INITIAL USER PROMPT:\n```\n{user_prompt}\n```\n"

        num_compress = len(trajectory_steps) - retain_most_turns
        compress_steps = trajectory_steps[:num_compress] if num_compress > 0 else trajectory_steps
        for idx, step in enumerate(compress_steps):
            thought = step['thought']
            action = step['action']
            observation = step['observation']
            history_to_compress += f'THOUGHT:\n```\n{thought}\n```\n'
            history_to_compress += f'ACTION:\n```\n{action}\n```\n'
            history_to_compress += f"\nOBSERVATION:\n```\n{observation}\n```\n"
        
        history_to_append = ""
        for step in trajectory_steps[-retain_most_turns:]:
            thought = step['thought']
            action = step['action']
            observation = step['observation']
            history_to_append += f'THOUGHT:\n```\n{thought}\n```\n'
            history_to_append += f'ACTION:\n```\n{action}\n```\n'
            history_to_append += f'OBSERVATION:\n```\n{observation}\n```\n'
        return history_to_compress, history_to_append

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images)
        into a format processable by the model.

        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
            memory_contexts (List[Dict[str, str]]): Conversation history for the current environment instance, may contain 'user', 'assistant' keys.

        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        data_source = gen_batch.non_tensor_batch['data_source'][item]
        trajectory_steps = gen_batch.non_tensor_batch['tools_kwargs'][item]['trajectory_steps']
        system_prompt = gen_batch.non_tensor_batch['tools_kwargs'][item]['system_prompt']
        user_prompt = gen_batch.non_tensor_batch['tools_kwargs'][item]['user_prompt']

        ##### turn the current conversation into a prompt #####
        retain_turns = self._get_retain_turns()

        history_to_compress, history_to_append = self._prepare_history_message(trajectory_steps[:-1], system_prompt, user_prompt, retain_turns)
        expected_action = trajectory_steps[-1]['action']

        if history_to_compress == "":
            raise ValueError("No history to compress for summarization.")

        compressor_prompt_version = self.config.env.rollout.get('compressor_prompt_version', 'v1')
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
        else:
            raise ValueError(f"Unsupported compressor_prompt_version: {compressor_prompt_version}")

        prompt_with_conv = format_prompt.format(conversation=history_to_compress)
        chat = [{"role": "user", "content": prompt_with_conv}]
        chat = np.array(chat)

        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )

        # Initialize return dict
        row_dict = {}

        raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)

        position_ids = compute_position_id_with_mask(attention_mask)

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            'index': item,
            'data_source': data_source,
            "turns": trajectory_steps,
            "history_to_append": history_to_append,
            "expected_actions": expected_action,
            "system_prompt": system_prompt,
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()

        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto,
    ) -> DataProto:
        """
        Process a batch of observations samples, converting environment observations into model-processable format.

        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
            memory_contexts (List[str]): Conversation history for the current environment instance, may contain 'user', 'assistant' keys.

        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []

        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            ##### Conversation Memory Implementation #####
            # Add memory context to the preprocessing inputs
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
            )
            processed_samples.append(processed)

        # Aggregate batch data
        batch = collate_fn(processed_samples)

        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def get_batch_response_from_agent(
        self,
        compressions: DataProto,
    ) -> Tuple[List[str], List[str]]:
        
        use_tool_call = self.config.env.swebench.get("use_tool_call", False)
        scaffold = self.config.env.swebench.get("scaffold", "")
        llm_name = self.config.env.response_agent.model_name

        async def process_all():
            batch_size = len(compressions.batch['input_ids'])
            # Global cap across ALL endpoints (keeps your original behavior)
            sem_global = asyncio.Semaphore(self.max_concurrency)

            # Optional per-endpoint caps (parallelize across gateways)
            ep_sems = None
            if self._per_endpoint_concurrency is not None:
                ep_sems = [asyncio.Semaphore(self._per_endpoint_concurrency) for _ in self._endpoints]

            async def _invoke_with_failover(order, invoker):
                """
                order: list of endpoint indices to try in sequence
                invoker: async function(endpoint_idx) -> response
                """
                last_exc = None
                for i in order:
                    try:
                        if ep_sems:
                            async with ep_sems[i]:
                                return await asyncio.wait_for(invoker(i), timeout=self._timeout_s)
                        else:
                            return await asyncio.wait_for(invoker(i), timeout=self._timeout_s)
                    except Exception as e:
                        last_exc = e
                        continue
                # If we get here, all endpoints failed
                raise last_exc or RuntimeError("All endpoints failed")

            async def process_item(idx: int):
                compression_ids = compressions.batch['responses'][idx]
                compression_text = self.tokenizer.decode(compression_ids, skip_special_tokens=True)

                # Parse recent turns and current query from the original turns
                turns = compressions.non_tensor_batch.get('turns', [[]])[idx]
                retain_turns = self._get_retain_turns()

                history_to_append = compressions.non_tensor_batch.get('history_to_append', [''])[idx]
                system_prompt = compressions.non_tensor_batch.get('system_prompt', [''])[idx]

                # Build the formatted prompt based on whether we have recent turns
                if retain_turns > 0:
                    # Format with summary + recent turns + current query
                    formatted_prompt = (
                        "You are a helpful assistant. Based on the summarized conversation history and recent detailed turns, "
                        "respond to the latest user message in the recent detailed turns.\n"
                        "## Conversation Summary:\n"
                        f"{compression_text}\n"
                        "## Recent Detailed Turns:\n"
                        f"{history_to_append}\n"
                    )
                else:
                    # Format with just summary + current query
                    obs_text = turns[-2]['observation']
                    formatted_prompt = (
                        "You are a helpful assistant. Based on the summarized conversation history, "
                        "generate a response to the user's query.\n"
                        "## Conversation History:\n"
                        f"{compression_text}\n"
                        "## Current User Query:\n"
                        f"{obs_text}"
                    )
                steps_remaining = self.config.env.swebench.get('max_steps', 40) - len(turns) + 1
                if steps_remaining > 0:
                    stepcount_message = f"Steps Remaining: {steps_remaining}"
                else:
                    stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
                formatted_prompt += f"\n{stepcount_message}\n"

                try:
                    async with sem_global:
                        # Choose where to start (round-robin), then try all for failover
                        n = len(self._endpoints)
                        start = next(self._rr) % n
                        order = [(start + k) % n for k in range(n)]

                        async def _call(endpoint_idx):
                            messages = [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": formatted_prompt}]
                            tools = None
                            if use_tool_call:
                                if scaffold == "r2egym":
                                    tools = [search_tool, file_editor, r2egym_bash_execute_tool, finish_tool]
                                elif scaffold == "openhands" or scaffold == "sweagent":
                                    tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]
                                if "vertex" not in llm_name.lower():
                                    # vertex is not supported yet: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude-prompt-caching
                                    # litellm might need dev install with vertex: https://github.com/BerriAI/litellm/issues/6898
                                    # add prompt caching for anthropic
                                    tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
                                    breakpoints_remaining = 3  # remaining 1 for system/tool (above)
                                    for message in reversed(messages):
                                        if message["role"] in ("user", "tool"):
                                            if breakpoints_remaining > 0:
                                                message["cache_control"] = {"type": "ephemeral"}
                                                breakpoints_remaining -= 1
                                            else:
                                                break
                            # Build API call parameters
                            call_params = {
                                "model": self.model_name,
                                "messages": messages,
                                "temperature": self.config.env.response_agent.temperature,
                                "max_tokens": 4000,
                                "n": 1,
                                "timeout": self._timeout_s,  # Set timeout for the API request itself
                                "tools": tools,
                            }

                            if not use_tool_call:
                                call_params["tool_choice"] = "none"
                                call_params["function_call"] = None

                            # Set api_base for vLLM endpoints
                            if self.using_local and self._endpoints[endpoint_idx] is not None:
                                call_params["api_base"] = self._endpoints[endpoint_idx]

                            return await litellm.acompletion(**call_params)

                        response = await _invoke_with_failover(order, _call)
                    
                    completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else -1
                    # response_text = response.choices[0].message.content.strip()
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    # response_text = "I apologize, but I'm unable to generate a response at this moment."
                    response = None
                    completion_tokens = -1
                    formatted_prompt = ""

                return idx, response, compression_text, completion_tokens, formatted_prompt

            results = await asyncio.gather(*[process_item(i) for i in range(batch_size)])
            results.sort(key=lambda x: x[0])
            responses = [r[1] for r in results]
            compression_texts = [r[2] for r in results]
            completion_tokens = [r[3] for r in results]
            formatted_prompts = [r[4] for r in results]
            return responses, compression_texts, completion_tokens, formatted_prompts

        # Create or fetch the loop, and (optionally) cap its thread pool
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self.max_executor_threads is not None:
            loop.set_default_executor(ThreadPoolExecutor(max_workers=self.max_executor_threads))

        return loop.run_until_complete(process_all())

    def gather_rollout_data(
        self,
        batch_list: List[Dict],
        rewards: np.ndarray,
        length_penalties: np.ndarray,
        traj_uid: np.ndarray,
    ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.

        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers

        Returns:
            DataProto: Collected and organized trajectory data
        """
        episode_rewards_mean = np.mean(rewards)
        episode_rewards_min = np.min(rewards)
        episode_rewards_max = np.max(rewards)
        length_penalty_mean = np.mean(length_penalties)

        effective_batch = []
        for bs, data in enumerate(batch_list):
            assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
            # episode_rewards
            data['episode_rewards'] = rewards[bs]
            data['episode_rewards_mean'] = episode_rewards_mean
            data['episode_rewards_min'] = episode_rewards_min
            data['episode_rewards_max'] = episode_rewards_max
            # episode_lengths
            data['episode_lengths'] = np.int64(1)
            data['episode_lengths_mean'] = np.int64(1)
            data['episode_lengths_min'] = np.int64(1)
            data['episode_lengths_max'] = np.int64(1)
            # length_penalties
            data['length_penalties'] = length_penalties[bs]
            data['length_penalties_mean'] = length_penalty_mean

            effective_batch.append(data)

        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            is_train: bool = True,
    ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
            is_train (bool): Whether the rollout is for training or evaluation

        Returns:
            batch_list (List[Dict]): List of trajectory data for each environment
            rewards (np.ndarray): Total rewards for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """
        if is_train:
            gen_batch = gen_batch.repeat(
                repeat_times=self.config.env.rollout.n, interleave=True
            )

        batch_size = len(gen_batch.batch['input_ids'])
        batch_output = None

        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid) # non-interleaved uids
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)

        batch = self.preprocess_batch(gen_batch=gen_batch)
        expected_actions = batch.non_tensor_batch['expected_actions']
        turns = batch.non_tensor_batch['turns']
        history_to_appends = batch.non_tensor_batch['history_to_append']

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tool_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tool_kwargs")
        batch_input = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        batch_input.meta_info = gen_batch.meta_info

        ### Generate summary from memory model ###
        batch_output = actor_rollout_wg.generate_sequences(batch_input)

        ### Merge batch and batch_output to have both turns and responses ###
        batch = batch.union(batch_output)

        ### Generate action from agent model ###
        text_actions, compression_texts, completion_tokens, formatted_prompts = self.get_batch_response_from_agent(
            compressions=batch,
        )

        batch.non_tensor_batch['uid'] = uid_batch
        batch.non_tensor_batch['traj_uid'] = traj_uid

        rewards = []
        length_penalties = []
        text_action_parsed_list = []
        reward_details_list = []
        for i in range(batch_size):
            _, text_action_parsed = parse_action(text_actions[i], use_tool_call=self.config.env.swebench.get('use_tool_call', False))
            expected_action_parsed = Action.from_string(expected_actions[i])
            # reward = action_consistency_reward(text_actions[i], expected_actions[i])
            reward, reward_details = action_consistency_reward_detailed(text_action_parsed, expected_action_parsed)
            length_penalty_coefficient = self.config.env.swebench.get('length_penalty_coefficient', 0.0)
            length_penalty = length_penalty_coefficient * completion_tokens[i]
            reward -= length_penalty
            reward = max(reward, 0.0)  # ensure non-negative reward
            rewards.append(reward)
            length_penalties.append(length_penalty)
            text_action_parsed_list.append(text_action_parsed)
            reward_details_list.append(json.dumps(reward_details))
        rewards = np.array(rewards, dtype=np.float32)

        batch.non_tensor_batch['rewards'] = rewards
        batch.non_tensor_batch['length_penalties'] = length_penalties
        # save the text actions so that we can analyze later
        # batch.non_tensor_batch['history_to_appends'] = history_to_appends
        batch.non_tensor_batch['reward_details'] = np.array(reward_details_list, dtype=object)
        # save the text actions so that we can analyze later
        batch.non_tensor_batch['text_actions'] = np.array([t.to_xml_string() for t in text_action_parsed_list], dtype=object)
        batch.non_tensor_batch['expected_actions'] = np.array(expected_actions, dtype=object)
        batch.non_tensor_batch['formatted_prompts'] = np.array(formatted_prompts, dtype=object)

        batch_list: list[dict] = to_list_of_dict(batch)

        return batch_list, rewards, traj_uid, length_penalties

    def multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        batch_list, rewards, traj_uid, length_penalties = \
            self.vanilla_multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            is_train=is_train,
        )
        assert len(batch_list) == len(rewards)
        assert len(batch_list) == len(traj_uid)
        assert len(batch_list) == len(length_penalties)

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            batch_list=batch_list,
            rewards=rewards,
            traj_uid=traj_uid,
            length_penalties=length_penalties,
        )

        return gen_batch_output
