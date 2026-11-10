import asyncio
import copy
import json
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import litellm
import requests
import yaml
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from r2egym.agenthub.action import Action
from r2egym.agenthub.environment.env import RepoEnv
from r2egym.agenthub.tools import (
    execute_bash_tool,
    file_editor,
    finish_tool,
    r2egym_bash_execute_tool,
    search_tool,
    str_replace_editor_tool,
    submit_tool,
)
from r2egym.agenthub.trajectory import Trajectory_Summ as Trajectory
from r2egym.agenthub.trajectory import TrajectoryStep_Summ as TrajectoryStep
from r2egym.agenthub.utils.log import get_logger

logger = get_logger(__name__)
MAX_CONTEXT_TOKENS = 65536

OPENAI_MODELS = [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o1-preview",
    "gpt-5",
    "gpt-5-mini",
]


##############################################################################
# Dataclasses
##############################################################################
@dataclass
class AgentArgs:
    system_prompt: str
    instance_prompt: str
    command_files: List[Path]
    llm_name: str
    summary_prompt: str = ""
    query_prompt: str = ""
    retain_most_turns: int = 1
    memory_model_name: str = "gpt-3.5-turbo"
    memory_model_temperature: float = 0.0
    memory_model_max_gen_tokens: int = 1536
    address: List[str] = None
    llm_base_url: Optional[str] = "http://localhost:8000/v1"
    demo_file: Optional[Path] = None
    use_demo: Optional[bool] = False
    use_optimized_schedule: Optional[bool] = False
    use_user_prompt: Optional[bool] = False
    nosummary: Optional[bool] = False
    other_args: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AgentArgs":
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)


@dataclass
class CompressionResult:
    """Structured container for background memory compression results."""

    summary: str = ""
    tokens: int = 0
    exec_time: float = 0.0
    finish_reason: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


##############################################################################
# Agent Class
##############################################################################
class COMEMAgent:
    """Agent that offloads conversation history compression to an asynchronous
    k-step-off (KSO) memory model while interacting with the environment.
    """

    def __init__(self, name: str, args: AgentArgs, logger=None):
        self.name = name
        self.args = args
        self.logger = logger or get_logger(name)

        if self.args.use_optimized_schedule:
            assert (
                self.args.retain_most_turns >= 3
            ), "retain_most_turns must be at least 3 for optimized schedule"

        self.llm_name = args.llm_name
        self.llm_base_url = (
            os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
            if ("openai/" in self.llm_name) or ("hosted_vllm" in self.llm_name)
            else None
        )
        self.memory_base_url = os.environ.get(
            "MEMORY_LLM_BASE_URL", "http://localhost:9001/v1"
        )
        self.system_prompt_template = args.system_prompt
        self.instance_prompt_template = args.instance_prompt
        self.summary_prompt_template = args.summary_prompt
        self.query_prompt_template = args.query_prompt
        self.command_files = args.command_files
        self.other_args = args.other_args or {}
        self.nosummary = args.nosummary or False
        self.logger.info(f"Initialized Agent: {self.name} with LLM: {self.llm_name}")
        self.max_retries = self.other_args.get("max_retries", 5)
        self.llm_timeout = self.other_args.get("timeout", 3000)

        self.mem_tokenizer = AutoTokenizer.from_pretrained(self.args.memory_model_name)

        self.agent_client = AsyncOpenAI(base_url=self.llm_base_url, api_key="dummy")
        self.mem_client = AsyncOpenAI(base_url=self.memory_base_url, api_key="dummy")

    def reset(self):
        """Reset the agent's trajectory."""
        self.trajectory_steps = []
        self.history = []

    # ------------------------------------------------------------------
    # Prompt & History Formatting
    # ------------------------------------------------------------------
    def prepare_system_message(
        self, problem_statement: str, structure: str, command_docs: str, demo: str
    ) -> str:
        """Prepare the system prompt by filling in the placeholders."""
        return self.system_prompt_template.format(
            command_docs=command_docs,
            demo=demo,
        )

    def prepare_history_to_append(self, append_steps: List[TrajectoryStep]) -> str:
        """Format trajectory steps as THOUGHT / ACTION / OBSERVATION blocks."""
        chunks = []
        for step in append_steps:
            chunks.append(
                f"THOUGHT:\n```\n{step.thought}\n```\n"
                f"ACTION:\n```\n{step.action}\n```\n"
                f"OBSERVATION:\n```\n{step.observation}\n```\n"
            )
        return "".join(chunks)

    def prepare_history_to_compress(self, compress_steps: List[TrajectoryStep]) -> str:
        """Format system/user headers + trajectory steps for memory compression."""
        header = (
            f"SYSTEM PROMPT:\n```\n{self.history[0]['content']}\n```\n"
            f"INITIAL USER PROMPT:\n```\n{self.history[1]['content']}\n```\n"
        )
        return header + self.prepare_history_to_append(compress_steps)

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens for a list of messages using litellm."""
        token_count = litellm.token_counter(model=self.llm_name, messages=messages)
        self.logger.info(f"Total tokens in conversation: {token_count}")
        return token_count

    def _count_summary_tokens(self, summary: str) -> int:
        """Count tokens for a summary string using the memory model's tokenizer."""
        token_count = len(self.mem_tokenizer.encode(summary))
        self.logger.info(f"Total tokens in summary: {token_count}")
        return token_count

    def _prepare_messages_for_summary(
        self, history_to_compress: str
    ) -> Tuple[List[Dict[str, str]], str, int]:
        if self.args.use_user_prompt:
            summary_prompt = self.summary_prompt_template.format(
                conversation=history_to_compress
            )
        else:
            summary_prompt = (
                "You are a helpful assistant that summarizes conversations for another model to use. "
                "You need to make sure that the summary captures all important details from the conversation "
                "so that the other model can understand the context just like reading the full conversation.\n\n"
                "### Conversation:\n"
                f"{history_to_compress}\n\n"
                "Please provide a detailed summary of the above conversation. Make sure to include all necessary "
                "details for the other model to understand the context, especially entities, actions taken, and outcomes. "
                "But NEVER include any suggestions or recommendations for future actions—only summarize what has already occurred."
            )
        messages = [{"role": "user", "content": summary_prompt}]
        summary_prompt_tokens = self._count_tokens(messages)
        logger.info(f"Total tokens in summary prompt: {summary_prompt_tokens}")
        return messages, summary_prompt, summary_prompt_tokens

    def _prepare_messages_for_query(
        self,
        problem_statement: str,
        summary: str,
        history_to_append: str,
        stepcount_message: str,
    ) -> List[Dict[str, str]]:
        if not summary.strip():
            return copy.deepcopy(self.history)

        if self.args.use_user_prompt:
            formatted_prompt = self.query_prompt_template.format(
                problem_statement=problem_statement,
                summary=summary,
                recent_messages=history_to_append,
            )
        else:
            formatted_prompt = (
                "You are a helpful assistant. Based on the summarized conversation history and recent detailed turns, "
                "respond to the latest user message in the recent detailed turns.\n"
                "## Conversation Summary:\n"
                f"{summary}\n"
                "## Recent Detailed Turns:\n"
                f"{history_to_append}\n"
            )
        formatted_prompt += f"\n{stepcount_message}\n"
        self.logger.info(f"QUERY PROMPT: {formatted_prompt}")
        return [
            {"role": "system", "content": self.history[0]["content"]},
            {"role": "user", "content": formatted_prompt},
        ]

    # ------------------------------------------------------------------
    # Async Model Queries (Memory Compressor & Policy Agent)
    # ------------------------------------------------------------------
    async def _start_memory_compression(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, int, float, Optional[str], float, float]:
        """Run asynchronous memory compression on the memory model endpoint.
        Returns a 6-tuple for backward compatibility with BenchmarkCOMEMAgent.
        """
        call_params = {
            "model": self.args.memory_model_name,
            "messages": messages,
            "n": 1,
            "temperature": self.args.memory_model_temperature,
            "max_tokens": self.args.memory_model_max_gen_tokens,
        }

        start_time = time.perf_counter()
        response = await self.mem_client.chat.completions.create(**call_params)
        end_time = time.perf_counter()

        summary_exec_time = end_time - start_time
        response_text = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason
        summary_tokens = self._count_summary_tokens(response_text)

        return (
            response_text,
            summary_tokens,
            summary_exec_time,
            finish_reason,
            start_time,
            end_time,
        )

    def _prepare_tools_and_caching(
        self, messages: List[Dict[str, str]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Select scaffold tools and apply ephemeral prompt-caching markers if enabled."""
        if not self.use_fn_calling:
            return None

        if self.scaffold == "r2egym":
            tools = [search_tool, file_editor, r2egym_bash_execute_tool, finish_tool]
        elif self.scaffold in ("openhands", "sweagent"):
            tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]
        else:
            tools = None

        if tools and "vertex" not in self.llm_name.lower():
            self.logger.warning(f"using prompt caching for {self.llm_name}")
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

    async def _execute_agent_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        include_glm_thinking: bool = True,
    ) -> Tuple[Any, float, float, float]:
        """Shared retry-wrapped completion call for both _model_query and _prefill_agent_model."""
        tools = self._prepare_tools_and_caching(messages)
        if "openai/" in self.llm_name or "hosted" in self.llm_name:
            litellm.api_key = None

        messages_ = copy.deepcopy(messages)
        vllm_llm_name = "/".join(self.llm_name.split("/")[1:])

        kwargs: Dict[str, Any] = (
            {} if tools else {"tool_choice": "none", "function_call": None}
        )
        if "o3" not in self.llm_name and "o4" not in self.llm_name:
            kwargs["temperature"] = temperature
        if include_glm_thinking and "GLM" in vllm_llm_name:
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled", "clear_thinking": False}
            }

        start_time = time.perf_counter()
        response = None
        for retries in range(self.max_retries):
            try:
                response = await self.agent_client.chat.completions.create(
                    model=vllm_llm_name,
                    tools=tools,
                    messages=messages_,
                    timeout=self.llm_timeout,
                    max_tokens=10000,
                    **kwargs,
                )
                break
            except Exception as e:
                self.logger.error(f"LLM query failed @ {retries}: {e}")
                if "RateLimitError" in str(e):
                    time.sleep(60)
                if retries + 1 >= self.max_retries:
                    raise e

        end_time = time.perf_counter()
        exec_time = end_time - start_time
        return response, exec_time, start_time, end_time

    async def _model_query(
        self, messages: List[Dict[str, str]], temperature: float = 0.0
    ) -> Tuple[Any, float, float, float, str]:
        """Query the policy LLM and scrape live vLLM Prometheus metrics."""
        response, exec_time, start_time, end_time = await self._execute_agent_completion(
            messages, temperature, include_glm_thinking=True
        )
        vllm_metrics = requests.get(
            self.llm_base_url.replace("/v1", "/metrics")
        ).text
        return response, exec_time, start_time, end_time, vllm_metrics

    async def _prefill_agent_model(
        self, messages: List[Dict[str, str]], temperature: float = 0.0
    ) -> Tuple[Any, float, float, float]:
        """Warm up the policy LLM KV cache without scraping metrics."""
        return await self._execute_agent_completion(
            messages, temperature, include_glm_thinking=False
        )

    # ------------------------------------------------------------------
    # Response Parsers
    # ------------------------------------------------------------------
    def parse_response(self, response: str) -> Tuple[str, Action]:
        """Extract <think>...</think> and <function=...>...</function> blocks."""
        pattern_thought = re.compile(r"(?s)(<think>.*?</think>)")
        pattern_action = re.compile(r"(?s)(<function=.*?</function>)")
        match_thought = pattern_thought.search(response)
        match_action = pattern_action.search(response)

        thought = match_thought.group(1).strip() if match_thought else ""
        action_str = match_action.group(1).strip() if match_action else ""
        return thought, Action.from_string(action_str)

    def parse_response_v2(self, response_text: str) -> Tuple[str, Action]:
        """Extract everything before <function=...> as thought, and the first <function=...> block as action."""
        pattern = re.compile(r"(?s)(<function=.*?</function>)")
        match = pattern.search(response_text)

        if match:
            action_str = match.group(1).strip()
            thought = response_text[: match.start()].strip()
        else:
            thought = response_text.strip()
            action_str = ""
        return thought, Action.from_string(action_str)

    def custom_parser(self, response) -> Tuple[str, Action]:
        thought = response.choices[0].message.content or ""
        try:
            tool_call = response.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            parameters = json.loads(tool_call.function.arguments)
            action = Action(function_name=function_name, parameters=parameters)
        except Exception:
            action = Action(function_name="", parameters={})
        return thought, action

    def custom_parser_glm(self, response) -> Tuple[str, Action]:
        """Custom parser for GLM-4.7 models which return reasoning in reasoning_content."""
        message = response.choices[0].message
        thought = getattr(message, "reasoning_content", None) or message.content or ""
        try:
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            parameters = json.loads(tool_call.function.arguments)
            action = Action(function_name=function_name, parameters=parameters)
        except Exception:
            action = Action(function_name="", parameters={})
        return thought, action

    # ------------------------------------------------------------------
    # Pipeline Helpers for run()
    # ------------------------------------------------------------------
    @staticmethod
    def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _append_turn_to_history(self, response, thought: str, action: Action, obs: Any):
        """Record the assistant turn and environment observation into self.history."""
        if self.use_fn_calling:
            assistant_response = response.choices[0].message.dict()
            if assistant_response.get("tool_calls"):
                assistant_response["tool_calls"] = assistant_response["tool_calls"][:1]
            self.history.append(assistant_response)
            try:
                tool_call = response.choices[0].message.tool_calls[0]
                self.history.append(
                    {
                        "role": "tool",
                        "content": str(obs),
                        "name": tool_call.function.name,
                        "tool_call_id": tool_call.id,
                    }
                )
                self.logger.warning("logging fn response as a tool call")
                self.logger.warning(
                    f"number of fn calls: {len(response.choices[0].message.tool_calls)}"
                )
            except Exception as e:
                self.logger.error(f"Error logging tool response: {e}")
                self.logger.warning("fallback: logging fn response as a tool call")
                self.history.append({"role": "user", "content": str(obs)})
        else:
            self.logger.warning("logging fn response as a user message")
            assistant_message = f"{thought}\n\n{action.to_xml_string()}"
            self.history.append({"role": "assistant", "content": assistant_message})
            self.history.append({"role": "user", "content": str(obs)})

    # ------------------------------------------------------------------
    # Main KSO Execution Loop
    # ------------------------------------------------------------------
    def run(
        self,
        env: "RepoEnv",
        use_fn_calling: bool = True,
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        max_token_limit: int = 65536,
        max_mem_token_limit: int = 131072,
        max_exec_time: int = 90,
        max_total_time: int = 50000,
        max_llm_time: int = 7200,
        temperature: float = 0,
        metadata: Optional[Dict[str, Any]] = None,
        scaffold: str = "r2egym",
    ) -> Trajectory:
        assert scaffold in [
            "r2egym",
            "openhands",
            "sweagent",
        ], "Scaffold must be either r2egym or openhands or sweagent"
        self.scaffold = scaffold
        metadata = metadata or {}
        start_time = time.perf_counter()
        self.llm_timeout = max_llm_time

        support_fn_calling = any(
            k in self.llm_name for k in ("gpt", "sonnet", "o3", "o4", "Coder", "GLM")
        )
        self.use_fn_calling = use_fn_calling and support_fn_calling
        self.logger.warning(f"Using fn calling: {self.use_fn_calling}")
        self.logger.info(f"Running agent {self.name} in environment {env}.")

        # Reset environment & agent state
        env.reset()
        env.add_commands(self.command_files)
        self.reset()

        problem_statement = env.runtime.get_task_instruction()
        gt_patch = env.runtime.commit.get_patch(test_file=True, non_test_file=False)
        system_prompt = self.system_prompt_template
        user_prompt = self.instance_prompt_template.format(
            problem_statement=problem_statement,
            gt_patch=gt_patch,
            working_dir="/testbed",
            test_patch_hint=metadata.get("test_patch_hint", ""),
            candidate_patch=metadata.get("candidate_patch", ""),
            candidate_patch_correctness=(
                "correct"
                if metadata.get("candidate_patch_correctness", False)
                else "incorrect"
            ),
        )
        if self.args.use_demo:
            with open(self.args.demo_file, "r") as file:
                demo = file.read()
            user_prompt = f"{demo}\n\n{user_prompt}"
        initial_user_prompt = user_prompt

        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Initialize KSO trajectory state
        done = False
        step_count = 0
        total_time_traj = 0.0
        exit_reason = "unknown"
        self.trajectory_steps: List[TrajectoryStep] = []

        current_summary = ""
        prefill_summary = ""
        compress_steps: List[TrajectoryStep] = []
        append_steps: List[TrajectoryStep] = []
        prefill_append_steps: List[TrajectoryStep] = []
        last_summary_step_idx = -1
        summary_prompt = ""
        summary_prompt_tokens = 0

        pending_memory_task: Optional[asyncio.Task] = None
        pending_prefill_task: Optional[asyncio.Task] = None
        loop = self._get_or_create_event_loop()
        traj_start_time = time.perf_counter()

        while not done:
            # --------------------------------------------------------------
            # Stage 1: Acquire background tasks from the previous step
            # --------------------------------------------------------------
            comp_res = CompressionResult()
            if pending_memory_task is not None:
                try:
                    comp_res = CompressionResult(
                        *loop.run_until_complete(pending_memory_task)
                    )
                    steps_since_summary = (
                        len(self.trajectory_steps) - last_summary_step_idx
                    )
                    if not self.args.use_optimized_schedule:
                        current_summary = comp_res.summary
                        append_steps = append_steps[-steps_since_summary:]
                    else:
                        prefill_summary = comp_res.summary
                        prefill_append_steps = append_steps[-steps_since_summary:]
                except Exception as e:
                    self.logger.error(f"Error retrieving memory task result: {e}")
                    exit_reason = "memory_compression_error"
                    done = True
                    break
            else:
                summary_prompt = ""
                summary_prompt_tokens = 0

            prefill_exec_time = 0.0
            if pending_prefill_task is not None:
                try:
                    _, prefill_exec_time, _, _ = loop.run_until_complete(
                        pending_prefill_task
                    )
                except Exception as e:
                    self.logger.error(f"Error retrieving prefill task result: {e}")
                    exit_reason = "prefill_error"
                    done = True
                    break

            # --------------------------------------------------------------
            # Stage 2: Schedule next background compression (or KV prefill)
            # --------------------------------------------------------------
            should_compress_this_step = (
                len(self.trajectory_steps) + 1
            ) % self.args.retain_most_turns == 0 and len(self.trajectory_steps) > 0

            if should_compress_this_step:
                summary_messages, summary_prompt, summary_prompt_tokens = (
                    self._prepare_messages_for_summary(
                        history_to_compress=self.prepare_history_to_compress(
                            compress_steps
                        )
                    )
                )
                if summary_prompt_tokens > max_mem_token_limit:
                    exit_reason = "token_limit_exceeded_before_memory_compression"
                    done = True
                    break

                pending_memory_task = loop.create_task(
                    self._start_memory_compression(messages=summary_messages)
                )
                last_summary_step_idx = len(self.trajectory_steps)
                if self.args.use_optimized_schedule:
                    current_summary = prefill_summary
                    append_steps = prefill_append_steps
            elif self.args.use_optimized_schedule:
                prefill_messages = self._prepare_messages_for_query(
                    problem_statement=problem_statement,
                    summary="" if self.nosummary else prefill_summary,
                    history_to_append=self.prepare_history_to_append(
                        prefill_append_steps
                    ),
                    stepcount_message="",
                )
                if self._count_tokens(prefill_messages) > max_token_limit:
                    exit_reason = "token_limit_exceeded_before_prefill"
                    done = True
                pending_prefill_task = loop.create_task(
                    self._prefill_agent_model(prefill_messages, temperature)
                )
            else:
                pending_memory_task = None

            # --------------------------------------------------------------
            # Stage 3: Build compact prompt & query policy LLM
            # --------------------------------------------------------------
            steps_remaining = max_steps - step_count
            stepcount_message = (
                f"Steps Remaining: {steps_remaining}"
                if steps_remaining > 0
                else "You have reached the maximum number of steps. Please submit your answer NOW."
            )
            messages = self._prepare_messages_for_query(
                problem_statement=problem_statement,
                summary="" if self.nosummary else current_summary,
                history_to_append=self.prepare_history_to_append(append_steps),
                stepcount_message=stepcount_message,
            )
            if self._count_tokens(messages) > max_token_limit:
                exit_reason = "token_limit_exceeded_before_llm_query"
                done = True
                break

            try:
                (
                    response,
                    llm_exec_time,
                    llm_start_time,
                    llm_end_time,
                    vllm_metrics,
                ) = loop.run_until_complete(self._model_query(messages, temperature))
            except ValueError as e:
                if "Total tokens:" in str(e) and ">" in str(e):
                    self.logger.error(f"Error querying LLM: {e}")
                    exit_reason = "total_tokens_exceeded_llm"
                    done = True
                    break
                raise
            except Exception as e:
                self.logger.error(f"Error querying LLM: {e}")
                self.logger.error(traceback.format_exc())
                exit_reason = "llm_query_error"
                done = True
                break

            # --------------------------------------------------------------
            # Stage 4: Parse response & execute environment step
            # --------------------------------------------------------------
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)
            else:
                prompt_tokens = -1
                completion_tokens = -1
                total_tokens = self._count_tokens(messages)

            self.response = response
            assistant_message = response.choices[0].message.content or ""
            if self.use_fn_calling:
                thought, action = (
                    self.custom_parser_glm(response)
                    if "GLM" in self.llm_name
                    else self.custom_parser(response)
                )
            else:
                thought, action = self.parse_response(assistant_message)

            try:
                obs, reward, done, info = env.step(action, timeout=max_exec_time)
            except Exception as e:
                obs = str(e)
                self.logger.error(f"Error during environment step: {obs}")

            env_exec_time = info["total_time"]
            total_step_time = (
                max(llm_exec_time, comp_res.exec_time + prefill_exec_time)
                + env_exec_time
            )
            total_time_traj += total_step_time
            step_count += 1

            # --------------------------------------------------------------
            # Stage 5: Update conversation history & check exit limits
            # --------------------------------------------------------------
            self._append_turn_to_history(response, thought, action, obs)

            if done:
                if steps_remaining > 0:
                    exit_reason = "agent"
                elif steps_remaining == 0:
                    exit_reason = "max_step_limit"
                else:
                    exit_reason = "agent_max_step_limit"
            elif step_count >= max_steps_absolute:
                exit_reason = "abs_step_limit"
                done = True
            elif total_time_traj >= max_total_time:
                exit_reason = "traj_time_limit"
                done = True

            # --------------------------------------------------------------
            # Stage 6: Record step telemetry & advance KSO windows
            # --------------------------------------------------------------
            trajectory_step = TrajectoryStep(
                step_idx=step_count - 1,
                thought=thought,
                action=action.to_xml_string(),
                observation=str(obs),
                done=done,
                info=info,
                summary=comp_res.summary,
                summary_prompt=summary_prompt,
                query_prompt=messages[1]["content"],
                history_to_append="",
                assistant_message="",
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_summary=comp_res.tokens,
                token_usage_summary_prompt=summary_prompt_tokens,
                token_usage_total=total_tokens,
                llm_exec_time=llm_exec_time,
                vllm_metrics=vllm_metrics,
                env_exec_time=env_exec_time,
                summ_exec_time=comp_res.exec_time,
                prefill_exec_time=prefill_exec_time,
                total_step_time=total_step_time,
                total_time_traj=total_time_traj,
                step_count=step_count,
                summary_finish_reason=comp_res.finish_reason,
                summary_start_time=comp_res.start_time,
                summary_finish_time=comp_res.end_time,
                agent_start_time=llm_start_time,
                agent_finish_time=llm_end_time,
            )
            self.trajectory_steps.append(trajectory_step)
            compress_steps.append(trajectory_step)
            append_steps.append(trajectory_step)

        output_patch = env.runtime.get_patch()
        conversation_history = json.loads(json.dumps(self.history, default=str))

        self.trajectory = Trajectory(
            trajectory_steps=[
                traj_step.model_dump() for traj_step in self.trajectory_steps
            ],
            problem_statement=problem_statement,
            docker_image=env.runtime.docker_image,
            agent_args=asdict(self.args),
            env_args=asdict(env.args),
            system_prompt=system_prompt,
            user_prompt=initial_user_prompt,
            conversation_history=conversation_history,
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_token_limit,
            max_llm_time=max_llm_time,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            exit_reason=exit_reason,
            output_patch=output_patch,
            retain_most_turns=self.args.retain_most_turns,
            use_optimized_schedule=self.args.use_optimized_schedule,
            traj_start_time=traj_start_time,
            traj_finish_time=time.perf_counter(),
        )

        self.logger.info(
            f"Agent completed in {time.perf_counter() - start_time} seconds."
        )
        return self.trajectory
