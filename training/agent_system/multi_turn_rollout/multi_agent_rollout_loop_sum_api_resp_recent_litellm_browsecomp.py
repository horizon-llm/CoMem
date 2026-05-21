"""
BrowseComp rollout collector with summarization support.

Adapted from the GLM variant for BrowseComp tasks (google_search, scrape, submit_answer).
Uses litellm for agent LLM calls and action_consistency_reward for comparing
predicted vs expected actions (XML format stored in trajectory steps).
"""

import numpy as np
import logging
import yaml
import json
import uuid
import asyncio
import itertools
from concurrent.futures import ThreadPoolExecutor
import litellm
from transformers import PreTrainedTokenizer
from typing import List, Dict, Tuple

from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.models.transformers.qwen2_vl import get_rope_index

from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.multi_turn_rollout.browsecomp_function_match import action_consistency_reward, parse_action, Action, action_consistency_reward_detailed

logger = logging.getLogger(__name__)

# ---- BrowseComp tool definitions (OpenAI function-calling format) ----
browsecomp_google_search_tool = {
    "type": "function",
    "function": {
        "name": "google_search",
        "description": (
            "Search the web via Google (Serper API). Returns organic "
            "results with title, link, snippet, and position."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "Search query string.",
                },
                "gl": {
                    "type": "string",
                    "description": "Region code (ISO 3166-1 alpha-2, e.g. 'us').",
                    "default": "us",
                },
                "hl": {
                    "type": "string",
                    "description": "Language code (ISO 639-1, e.g. 'en').",
                    "default": "en",
                },
                "num": {
                    "type": "integer",
                    "description": "Number of results (default 10).",
                    "default": 10,
                },
                "tbs": {
                    "type": "string",
                    "description": (
                        "Time filter: 'qdr:h' (hour), 'qdr:d' (day), "
                        "'qdr:w' (week), 'qdr:m' (month), 'qdr:y' (year)."
                    ),
                },
            },
            "required": ["q"],
        },
    },
}

browsecomp_scrape_tool = {
    "type": "function",
    "function": {
        "name": "scrape",
        "description": (
            "Scrape / read a web page or other URL and return its "
            "content as readable text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to scrape.",
                },
            },
            "required": ["url"],
        },
    },
}

browsecomp_submit_answer_tool = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": (
            "Submit your final answer. Call this once you have found the "
            "answer to the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your final answer to the question.",
                },
            },
            "required": ["answer"],
        },
    },
}

BROWSECOMP_TOOLS = [browsecomp_google_search_tool, browsecomp_scrape_tool, browsecomp_submit_answer_tool]

# ---- Compressor prompts (reused from GLM variant) ----
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

compressor_prompt_v5 = """You are a helpful assistant that summarizes conversations for another model to use. You need to make sure that the summary captures all important details from the conversation so that the other model can understand the context just like reading the full conversation.

### Conversation:
{conversation}

Please provide a detailed summary of the above conversation. Make sure to include all necessary details for the other model to understand the context, especially entities, actions taken, and outcomes. But NEVER include any suggestions or recommendations for future actions—only summarize what has already occurred."""

COMPRESSOR_PROMPTS = {
    "v1": compressor_prompt_v1,
    "v2": compressor_prompt_v2,
    "v3": compressor_prompt_v3,
    "v5": compressor_prompt_v5,
}


def load_prompt_template(template_path: str) -> str:
    """Load a prompt template from a file."""
    with open(template_path, 'r') as file:
        config = yaml.safe_load(file)
    return config.get("query_prompt", "")


class BrowseCompSumLiteLLMTrajectoryCollector(TrajectoryCollector):
    """Trajectory collector for BrowseComp tasks with summarization + litellm agent."""

    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_concurrency = config.env.response_agent.max_concurrency
        self.max_executor_threads = config.env.response_agent.max_executor_threads
        self._setup_endpoints()

    # ------------------------------------------------------------------
    # Endpoint configuration (identical to GLM variant)
    # ------------------------------------------------------------------

    def _setup_endpoints(self):
        model_name = self.config.env.response_agent.model_name
        timeout_s = getattr(self.config.env.response_agent, "timeout_s", 60)
        self._per_endpoint_concurrency = getattr(self.config.env.response_agent, "per_endpoint_concurrency", None)

        self.model_name = model_name
        self._timeout_s = timeout_s

        litellm.drop_params = True
        litellm.set_verbose = False
        litellm.success_callback = []
        litellm.failure_callback = []

        self.using_local = "openai/" in model_name or "hosted_vllm" in model_name

        if self.using_local:
            addrs = getattr(self.config.env.response_agent, "address", None) or []
            if not addrs:
                raise ValueError(
                    f"Model '{model_name}' appears to be vLLM-hosted, but no addresses provided "
                    "in config.env.response_agent.address"
                )
            self._endpoints = addrs
            self._rr = itertools.cycle(range(len(self._endpoints)))
            litellm.api_key = None
            logger.info(f"Using vLLM endpoints with model: {model_name}, addresses: {addrs}")
        else:
            self._endpoints = [None]
            self._rr = itertools.cycle([0])
            logger.info(f"Using cloud API with model: {model_name}")

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    # Max characters per observation in history. BrowseComp scrape results can
    # be full web pages (100K+ chars); cap them to keep the compressor prompt
    # within max_prompt_length.
    MAX_OBS_CHARS = 4000

    @staticmethod
    def _truncate_obs(obs: str, limit: int) -> str:
        if len(obs) <= limit:
            return obs
        half = limit // 2
        return obs[:half] + "\n\n... [truncated] ...\n\n" + obs[-half:]

    def _prepare_history_message(self, trajectory_steps, system_prompt, user_prompt, retain_most_turns=1):
        """Split trajectory into (history_to_compress, history_to_append)."""
        if len(trajectory_steps) <= retain_most_turns - 1:
            return "", ""

        history_to_compress = f"SYSTEM PROMPT:\n```\n{system_prompt}\n```\n"
        history_to_compress += f"INITIAL USER PROMPT:\n```\n{user_prompt}\n```\n"

        num_compress = len(trajectory_steps) - retain_most_turns
        compress_steps = trajectory_steps[:num_compress] if num_compress > 0 else trajectory_steps
        for step in compress_steps:
            obs = self._truncate_obs(step["observation"], self.MAX_OBS_CHARS)
            history_to_compress += f'THOUGHT:\n```\n{step["thought"]}\n```\n'
            history_to_compress += f'ACTION:\n```\n{step["action"]}\n```\n'
            history_to_compress += f'\nOBSERVATION:\n```\n{obs}\n```\n'

        history_to_append = ""
        for step in trajectory_steps[-retain_most_turns:]:
            obs = self._truncate_obs(step["observation"], self.MAX_OBS_CHARS)
            history_to_append += f'THOUGHT:\n```\n{step["thought"]}\n```\n'
            history_to_append += f'ACTION:\n```\n{step["action"]}\n```\n'
            history_to_append += f'OBSERVATION:\n```\n{obs}\n```\n'

        return history_to_compress, history_to_append

    # ------------------------------------------------------------------
    # Batch preprocessing
    # ------------------------------------------------------------------

    def preprocess_single_sample(self, item: int, gen_batch: DataProto):
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        trajectory_steps = gen_batch.non_tensor_batch['tools_kwargs'][item]['trajectory_steps']
        system_prompt = gen_batch.non_tensor_batch['tools_kwargs'][item]['system_prompt']
        user_prompt = gen_batch.non_tensor_batch['tools_kwargs'][item]['user_prompt']
        problem_statement = gen_batch.non_tensor_batch['tools_kwargs'][item]['problem_statement']

        retain_turns = self.config.env.rollout.get('retain_most_turns', 1)
        history_to_compress, history_to_append = self._prepare_history_message(
            trajectory_steps[:-1], system_prompt, user_prompt, retain_turns
        )
        expected_action = trajectory_steps[-1]['action']

        if history_to_compress == "":
            raise ValueError("No history to compress for summarization.")

        compressor_version = self.config.env.rollout.get('compressor_prompt_version', 'v1')
        format_prompt = COMPRESSOR_PROMPTS.get(compressor_version)
        if format_prompt is None:
            raise ValueError(f"Unsupported compressor_prompt_version: {compressor_version}")

        prompt_with_conv = format_prompt.format(conversation=history_to_compress)
        chat = [{"role": "user", "content": prompt_with_conv}]
        chat = np.array(chat)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )

        raw_prompt = prompt_with_chat_template
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
            "problem_statement": problem_statement,
        }

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()

        return row_dict

    def preprocess_batch(self, gen_batch: DataProto) -> DataProto:
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        for item in range(batch_size):
            processed = self.preprocess_single_sample(item=item, gen_batch=gen_batch)
            processed_samples.append(processed)

        batch = collate_fn(processed_samples)
        return DataProto.from_single_dict(data=batch, meta_info=gen_batch.meta_info)

    # ------------------------------------------------------------------
    # Agent API calls (litellm) with BrowseComp tools
    # ------------------------------------------------------------------

    def get_batch_response_from_agent(
        self,
        compressions: DataProto,
    ) -> Tuple[List[str], List[str]]:

        use_tool_call = self.config.env.swebench.get("use_tool_call", True)
        glm_model = self.config.env.swebench.get("glm_model", False)
        llm_name = self.config.env.response_agent.model_name

        async def process_all():
            batch_size = len(compressions.batch['input_ids'])
            sem_global = asyncio.Semaphore(self.max_concurrency)

            ep_sems = None
            if self._per_endpoint_concurrency is not None:
                ep_sems = [asyncio.Semaphore(self._per_endpoint_concurrency) for _ in self._endpoints]

            async def _invoke_with_failover(order, invoker):
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
                raise last_exc or RuntimeError("All endpoints failed")

            async def process_item(idx: int):
                compression_ids = compressions.batch['responses'][idx]
                compression_text = self.tokenizer.decode(compression_ids, skip_special_tokens=True)

                turns = compressions.non_tensor_batch.get('turns', [[]])[idx]
                retain_turns = self.config.env.rollout.get('retain_most_turns', 1)
                history_to_append = compressions.non_tensor_batch.get('history_to_append', [''])[idx]
                system_prompt = compressions.non_tensor_batch.get('system_prompt', [''])[idx]
                problem_statement = compressions.non_tensor_batch.get('problem_statement', [''])[idx]

                if retain_turns > 0:
                    format_prompt = load_prompt_template(
                        self.config.env.response_agent.get("query_prompt_template_path", "")
                    )
                    formatted_prompt = format_prompt.format(
                        problem_statement=problem_statement,
                        summary=compression_text,
                        recent_messages=history_to_append,
                    )
                else:
                    obs_text = turns[-2]['observation']
                    formatted_prompt = (
                        "You are a helpful assistant. Based on the summarized conversation history, "
                        "generate a response to the user's query.\n"
                        "## Conversation History:\n"
                        f"{compression_text}\n"
                        "## Current User Query:\n"
                        f"{obs_text}"
                    )

                max_steps = self.config.env.swebench.get('max_steps', 30)
                steps_remaining = max_steps - len(turns) + 1
                if steps_remaining > 0:
                    stepcount_message = f"Steps Remaining: {steps_remaining}"
                else:
                    stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
                formatted_prompt += f"\n{stepcount_message}\n"

                try:
                    async with sem_global:
                        n = len(self._endpoints)
                        start = next(self._rr) % n
                        order = [(start + k) % n for k in range(n)]

                        async def _call(endpoint_idx):
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": formatted_prompt},
                            ]
                            tools = BROWSECOMP_TOOLS if use_tool_call else None

                            call_params = {
                                "model": self.model_name,
                                "messages": messages,
                                "temperature": self.config.env.response_agent.temperature,
                                "max_tokens": 10000,
                                "n": 1,
                                "timeout": self._timeout_s,
                                "tools": tools,
                            }

                            if not use_tool_call:
                                call_params["tool_choice"] = "none"
                                call_params["function_call"] = None

                            if self.using_local and self._endpoints[endpoint_idx] is not None:
                                call_params["api_base"] = self._endpoints[endpoint_idx]

                            return await litellm.acompletion(**call_params)

                        response = await _invoke_with_failover(order, _call)

                    completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else -1
                except Exception as e:
                    logger.exception(f"Error processing item {idx}: {e}")
                    response = None
                    completion_tokens = -1

                return idx, response, compression_text, completion_tokens, formatted_prompt

            results = await asyncio.gather(*[process_item(i) for i in range(batch_size)])
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
            loop.set_default_executor(ThreadPoolExecutor(max_workers=self.max_executor_threads))

        return loop.run_until_complete(process_all())

    # ------------------------------------------------------------------
    # Reward & data gathering
    # ------------------------------------------------------------------

    def gather_rollout_data(
        self,
        batch_list: List[Dict],
        rewards: np.ndarray,
        length_penalties: np.ndarray,
        traj_uid: np.ndarray,
    ) -> DataProto:
        episode_rewards_mean = np.mean(rewards)
        episode_rewards_min = np.min(rewards)
        episode_rewards_max = np.max(rewards)
        length_penalty_mean = np.mean(length_penalties)

        effective_batch = []
        for bs, data in enumerate(batch_list):
            assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
            data['episode_rewards'] = rewards[bs]
            data['episode_rewards_mean'] = episode_rewards_mean
            data['episode_rewards_min'] = episode_rewards_min
            data['episode_rewards_max'] = episode_rewards_max
            data['episode_lengths'] = np.int64(1)
            data['episode_lengths_mean'] = np.int64(1)
            data['episode_lengths_min'] = np.int64(1)
            data['episode_lengths_max'] = np.int64(1)
            data['length_penalties'] = length_penalties[bs]
            data['length_penalties_mean'] = length_penalty_mean
            effective_batch.append(data)

        return DataProto.from_single_dict(data=collate_fn(effective_batch))

    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs: EnvironmentManagerBase,
        is_train: bool = True,
    ) -> DataProto:
        if is_train:
            gen_batch = gen_batch.repeat(
                repeat_times=self.config.env.rollout.n, interleave=True
            )

        batch_size = len(gen_batch.batch['input_ids'])

        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(batch_size)], dtype=object)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)

        batch = self.preprocess_batch(gen_batch=gen_batch)
        expected_actions = batch.non_tensor_batch['expected_actions']
        turns = batch.non_tensor_batch['turns']

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

        # Generate summary from memory model
        batch_output = actor_rollout_wg.generate_sequences(batch_input)

        # Merge to have both turns and responses
        batch = batch.union(batch_output)

        # Generate action from agent model
        text_actions, compression_texts, completion_tokens, formatted_prompts = \
            self.get_batch_response_from_agent(compressions=batch)

        batch.non_tensor_batch['uid'] = uid_batch
        batch.non_tensor_batch['traj_uid'] = traj_uid

        use_tool_call = self.config.env.swebench.get("use_tool_call", True)
        glm_model = self.config.env.swebench.get("glm_model", False)

        rewards = []
        length_penalties = []
        text_action_parsed_list = []
        reward_details_list = []
        for i in range(batch_size):
            _, text_action_parsed = parse_action(
                text_actions[i], use_tool_call=use_tool_call, glm_model=glm_model
            )
            expected_action_parsed = Action.from_string(expected_actions[i])
            reward, reward_details = action_consistency_reward_detailed(
                text_action_parsed, expected_action_parsed
            )
            length_penalty_coefficient = self.config.env.swebench.get('length_penalty_coefficient', 0.0)
            length_penalty = length_penalty_coefficient * completion_tokens[i]
            reward -= length_penalty
            reward = max(reward, 0.0)
            rewards.append(reward)
            length_penalties.append(length_penalty)
            text_action_parsed_list.append(text_action_parsed)
            reward_details_list.append(json.dumps(reward_details))
        rewards = np.array(rewards, dtype=np.float32)

        batch.non_tensor_batch['rewards'] = rewards
        batch.non_tensor_batch['length_penalties'] = np.array(length_penalties, dtype=object)
        batch.non_tensor_batch['reward_details'] = np.array(reward_details_list, dtype=object)
        batch.non_tensor_batch['text_actions'] = np.array(
            [t.to_xml_string() for t in text_action_parsed_list], dtype=object
        )
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
        batch_list, rewards, traj_uid, length_penalties = self.vanilla_multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            envs=envs,
            is_train=is_train,
        )
        assert len(batch_list) == len(rewards)
        assert len(batch_list) == len(traj_uid)
        assert len(batch_list) == len(length_penalties)

        return self.gather_rollout_data(
            batch_list=batch_list,
            rewards=rewards,
            traj_uid=traj_uid,
            length_penalties=length_penalties,
        )
