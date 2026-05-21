"""
Implement the k-step-off version (Comem-KSO)
"""
import os
import re
import copy
import yaml
import json
import time
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

import litellm

from r2egym.agenthub.action import Action
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.environment.env import RepoEnv
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.trajectory import TrajectoryStep_Summ as TrajectoryStep
from r2egym.agenthub.trajectory import Trajectory_Summ as Trajectory
from anthropic import Anthropic, AnthropicVertex  # Add Anthropic Vertex import
from r2egym.agenthub.tools import (
    r2egym_bash_execute_tool,
    search_tool,
    file_editor,
    finish_tool,
    str_replace_editor_tool,
    execute_bash_tool,
    submit_tool,
)
import traceback
import asyncio
import itertools
from openai import OpenAI, AsyncOpenAI
from transformers import AutoTokenizer
logger = get_logger(__name__)  # Logger for this module
MAX_CONTEXT_TOKENS = 65536  # Example max context length, adjust as needed

compressor_prompt_v5 = """You are a helpful assistant that summarizes conversations for another model to use. You need to make sure that the summary captures all important details from the conversation so that the other model can understand the context just like reading the full conversation.

### Conversation:
{conversation}

Please provide a detailed summary of the above conversation. Make sure to include all necessary details for the other model to understand the context, especially entities, actions taken, and outcomes. But NEVER include any suggestions or recommendations for future actions—only summarize what has already occurred."""

# Predefined list of OpenAI model names
OPENAI_MODELS = ['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', 'o1', 'o1-mini', 'o1-preview', 'gpt-5', 'gpt-5-mini']

##############################################################################
# AgentArgs Dataclass
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
    llm_base_url: Optional[str] = "http://localhost:8000/v1"  # None
    demo_file: Optional[Path] = None
    use_demo: Optional[bool] = False
    use_optimized_schedule: Optional[bool] = False
    use_user_prompt: Optional[bool] = False
    nosummary: Optional[bool] = False
    other_args: Optional[Dict[str, Any]] = None  # To handle extra configurations

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AgentArgs":
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)


##############################################################################
# Agent Class
##############################################################################
class COMEMAgent:
    """Agent handles the behavior of the model and how it interacts with the environment.
    
    This agent particularly utilizes the summarizers to compress the conversation history"""

    def __init__(self, name: str, args: AgentArgs, logger=None):
        self.name = name
        self.args = args
        if logger is None:
            self.logger = get_logger(name)
        else:
            self.logger = logger

        if self.args.use_optimized_schedule:
            assert self.args.retain_most_turns >= 3, "retain_most_turns must be at least 3 for optimized schedule" 
        
        self.llm_name = args.llm_name

        self.llm_base_url = (
            # "http://localhost:8000/v1"
            os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
            if ("openai/" in self.llm_name) or ("hosted_vllm" in self.llm_name)
            else None
        )
        self.memory_base_url = os.environ.get("MEMORY_LLM_BASE_URL", "http://localhost:9001/v1")
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

    def prepare_system_message(
        self, problem_statement: str, structure: str, command_docs: str, demo: str
    ) -> str:
        """Prepare the system prompt by filling in the placeholders."""
        system_prompt = self.system_prompt_template.format(
            # problem_statement=problem_statement,
            # structure=structure,
            command_docs=command_docs,
            demo=demo,
        )
        return system_prompt

    def prepare_history_to_compress(self, compress_steps) -> str:
        """Prepare the agent history message from the trajectory for compression."""
        # include system prompt
        history_to_compress = f"SYSTEM PROMPT:\n```\n{self.history[0]['content']}\n```\n"
        history_to_compress += f"INITIAL USER PROMPT:\n```\n{self.history[1]['content']}\n```\n"

        for idx, step in enumerate(compress_steps):
            thought = step.thought
            action = step.action
            observation = step.observation
            history_to_compress += f'THOUGHT:\n```\n{thought}\n```\n'
            history_to_compress += f'ACTION:\n```\n{action}\n```\n'
            history_to_compress += f"\nOBSERVATION:\n```\n{observation}\n```\n"
        return history_to_compress
    
    def prepare_history_to_append(self, append_steps) -> str:
        """Prepare the agent history message from the trajectory for appending."""
        history_to_append = ""
        for idx, step in enumerate(append_steps):
            thought = step.thought
            action = step.action
            observation = step.observation
            history_to_append += f'THOUGHT:\n```\n{thought}\n```\n'
            history_to_append += f'ACTION:\n```\n{action}\n```\n'
            history_to_append += f'OBSERVATION:\n```\n{observation}\n```\n'
        return history_to_append
    
    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Counts the tokens for a list of messages using the litellm library.
        Adjust as needed depending on the model and library.
        """
        token_count = litellm.token_counter(model=self.llm_name, messages=messages)
        self.logger.info(f"Total tokens in conversation: {token_count}")
        return token_count

    def _count_summary_tokens(self, summary: str) -> int:
        """
        Counts the tokens for a summary string using the memory model's tokenizer.
        """
        tokens = self.mem_tokenizer.encode(summary)
        token_count = len(tokens)
        self.logger.info(f"Total tokens in summary: {token_count}")
        return token_count

    def _prepare_messages_for_summary(self, history_to_compress: str) -> List[Dict[str, str]]:
        if self.args.use_user_prompt:
            summary_prompt = self.summary_prompt_template.format(conversation=history_to_compress)
        else:
            summary_prompt = compressor_prompt_v5.format(conversation=history_to_compress)
        messages = [
            {"role": "user", "content": summary_prompt},
        ]
        summary_prompt_tokens = self._count_tokens(messages)
        logger.info(f"Total tokens in summary prompt: {summary_prompt_tokens}")
        return messages, summary_prompt, summary_prompt_tokens
    
    async def _start_memory_compression(
        self,
        messages: List[Dict[str, str]],
    ):
        call_params = {
            "model": self.args.memory_model_name,
            "messages": messages,
            "n": 1,
        }

        # Only add temperature and max_tokens for vLLM endpoints
        call_params["temperature"] = self.args.memory_model_temperature
        call_params["max_tokens"] = self.args.memory_model_max_gen_tokens

        start_time = time.perf_counter()
        response = await self.mem_client.chat.completions.create(**call_params)
        end_time = time.perf_counter()
        summary_exec_time = end_time - start_time
        response_text = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason

        summary_tokens = self._count_summary_tokens(response_text)

        return response_text, summary_tokens, summary_exec_time, finish_reason, start_time, end_time

    def _prepare_messages_for_query(self, problem_statement: str, summary: str, history_to_append: str, stepcount_message: str) -> List[Dict[str, str]]:
        if summary.strip() != "":
            if self.args.use_user_prompt:
                formatted_prompt = self.query_prompt_template.format(
                    problem_statement=problem_statement,
                    summary=summary,
                    recent_messages=history_to_append,
                )
                formatted_prompt += f"\n{stepcount_message}\n"
                self.logger.info(f"QUERY PROMPT: {formatted_prompt}")
                messages = [
                    {"role": "system", "content": self.history[0]["content"]},
                    {"role": "user", "content": formatted_prompt},
                ]
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
            messages = [
                {"role": "system", "content": self.history[0]["content"]},
                {"role": "user", "content": formatted_prompt},
            ]
        else:
            messages = copy.deepcopy(self.history)
        return messages

    async def _model_query(
        self, messages: List[Dict[str, str]], temperature: float = 0.0,) -> Dict[str, Any]:
        """Query the LLM with the messages and measure execution time."""
        response = None
        retries = 0
        tools = None

        if self.use_fn_calling:
            if self.scaffold == "r2egym":
                tools = [search_tool, file_editor, r2egym_bash_execute_tool, finish_tool]
            elif self.scaffold == "openhands" or self.scaffold == "sweagent":
                tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]
            if "vertex" not in self.llm_name.lower():
                self.logger.warning(f"using prompt caching for {self.llm_name}")
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
        
        # check if using locally hosted models
        using_local = "openai/" in self.llm_name or "hosted" in self.llm_name
        if using_local:
            litellm.api_key = None
        
        messages_ = copy.deepcopy(messages)
        # total_tokens = self._count_tokens(messages_)
        # if total_tokens > MAX_CONTEXT_TOKENS:
        #     logger.warning(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")
        #     raise ValueError(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")

        # query the model with retries
        start_time = time.perf_counter()
        while retries < self.max_retries:
            try:
                kwargs = {
                    "tool_choice": "none",
                    "function_call": None,
                }
                if tools:
                    kwargs = {}
                if "o3" not in self.llm_name and "o4" not in self.llm_name:
                    kwargs["temperature"] = temperature
                # response = litellm.completion(
                #     model=self.llm_name,
                #     tools=tools,
                #     messages=messages_,
                #     timeout=self.llm_timeout,
                #     api_base=self.llm_base_url,
                #     # max_tokens=3000,
                #     **kwargs,
                # )
                vllm_llm_name = "/".join(self.llm_name.split("/")[1:])  # remove openai/
                if "GLM" in vllm_llm_name:
                    response = await self.agent_client.chat.completions.create(
                        model=vllm_llm_name,
                        tools=tools,
                        messages=messages_,
                        timeout=self.llm_timeout,
                        max_tokens=10000,
                        extra_body={
                            "thinking":{
                            "type":"enabled",
                            "clear_thinking": False  # False for Preserved Thinking
                        }},
                        **kwargs,
                    )
                else:
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
                retries += 1
                if "RateLimitError" in str(e):
                    time.sleep(60)
                if retries >= self.max_retries:
                    raise e
        
        # End timer, calculate total execution time, and include in response
        end_time = time.perf_counter()
        exec_time = end_time - start_time
        vllm_metrics = requests.get(
            self.llm_base_url.replace("/v1", "/metrics")
        ).text
        return response, exec_time, start_time, end_time, vllm_metrics
    
    async def _prefill_agent_model(
            self, messages: List[Dict[str, str]], temperature: float = 0.0,) -> Dict[str, Any]:
        """Query the LLM with the messages and measure execution time."""
        response = None
        retries = 0
        tools = None

        if self.use_fn_calling:
            if self.scaffold == "r2egym":
                tools = [search_tool, file_editor, r2egym_bash_execute_tool, finish_tool]
            elif self.scaffold == "openhands" or self.scaffold == "sweagent":
                tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]
            if "vertex" not in self.llm_name.lower():
                self.logger.warning(f"using prompt caching for {self.llm_name}")
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
        
        # check if using locally hosted models
        using_local = "openai/" in self.llm_name or "hosted" in self.llm_name
        if using_local:
            litellm.api_key = None
        
        messages_ = copy.deepcopy(messages)
        # total_tokens = self._count_tokens(messages_)
        # if total_tokens > MAX_CONTEXT_TOKENS:
        #     logger.warning(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")
        #     raise ValueError(f"Total tokens: {total_tokens} > {MAX_CONTEXT_TOKENS}")

        start_time = time.perf_counter()
        # query the model with retries
        while retries < self.max_retries:
            try:
                kwargs = {
                    "tool_choice": "none",
                    "function_call": None,
                }
                if tools:
                    kwargs = {}
                if "o3" not in self.llm_name and "o4" not in self.llm_name:
                    kwargs["temperature"] = temperature
                # response = litellm.completion(
                #     model=self.llm_name,
                #     tools=tools,
                #     messages=messages_,
                #     timeout=self.llm_timeout,
                #     api_base=self.llm_base_url,
                #     max_tokens=1,
                #     **kwargs,
                # )
                vllm_llm_name = "/".join(self.llm_name.split("/")[1:])  # remove openai/
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
                retries += 1
                if "RateLimitError" in str(e):
                    time.sleep(60)
                if retries >= self.max_retries:
                    raise e
        
        # End timer, calculate total execution time, and include in response
        end_time = time.perf_counter()
        exec_time = end_time - start_time
        return response, exec_time, start_time, end_time
    
    def parse_response(self, response: Dict[str, Any]) -> Tuple[str, Action]:
        """
        Parse the response from the LLM.
        """
        """
        Extracts:
        - thought: first thing in <think>...</think> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<think>` up to the first `</think>`
        pattern_thought = re.compile(r"(?s)(<think>.*?</think>)")
        pattern_action = re.compile(r"(?s)(<function=.*?</function>)")
        match_thought = pattern_thought.search(response)
        match_action = pattern_action.search(response)

        if match_thought:
            thought = match_thought.group(1)  # The entire <think>...</think> block
        else:
            thought = ""
        if match_action:
            action = match_action.group(1)  # The entire <function=...></function> block
        else:
            action = ""
        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action = Action.from_string(action)

        return thought, action

    def parse_response_v2(self, response_text: str) -> Tuple[str, Action]:
        """
        Extracts:
        - thought: everything before the first <function=...> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<function=` up to the first `</function>`
        pattern = re.compile(r"(?s)(<function=.*?</function>)")
        match = pattern.search(response_text)

        if match:
            action = match.group(1)  # The entire <function=...></function> block
            thought = response_text[: match.start()]  # Everything before the block
        else:
            # If no match, treat entire text as "thought"
            thought = response_text
            action = ""

        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action = Action.from_string(action)

        return thought, action

    def custom_parser(self, response):
        thought = response.choices[0].message.content
        if not thought:
            thought = ""

        try:
            function_name = response.choices[0].message.tool_calls[0].function.name
            parameters = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            action = Action(function_name=function_name, parameters=parameters)
        except:
            action = Action(function_name="", parameters={})

        return thought, action
    
    def custom_parser_glm(self, response):
        """
        Custom parser for GLM-4.7 models which return reasoning in
        'reasoning' or 'reasoning_content' fields on the tool_call object.
        """
        message = response.choices[0].message

        # GLM-4.7 puts reasoning on the tool_call object, not the message
        thought = ""
        try:
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                thought = message.reasoning_content
        except:
            pass

        # Fall back to message content if no reasoning found in tool_call
        if not thought and message.content:
            thought = message.content

        try:
            function_name = message.tool_calls[0].function.name
            parameters = json.loads(message.tool_calls[0].function.arguments)
            action = Action(function_name=function_name, parameters=parameters)
        except:
            action = Action(function_name="", parameters={})

        return thought, action
    
    def run(
        self,
        env: "RepoEnv",  # env: RepoEnv
        use_fn_calling: bool = True,
        # step limits TODO: maybe add these limits in the agent args
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        # token limits
        max_token_limit: int = 65536,  # 64k tokens
        max_mem_token_limit: int = 131072,  # 128k tokens
        # time limits
        max_exec_time: int = 90,  # 5 mins per env execution
        max_total_time: int = 50000,  # 20 minutes overall agent run limit
        max_llm_time: int = 7200,  # 2 mins per LLM timeout (note this is per query exlcuding retries | not enforcing hard limit since llm might hit rate limits etc)
        # temperature
        temperature=0,
        # additional metadata e.g. for hints / additional inputs etc
        metadata: Optional[Dict[str, Any]] = {},
        scaffold: str = "r2egym",
    ):
        assert scaffold in ["r2egym", "openhands", "sweagent"], "Scaffold must be either r2egym or openhands or sweagent"
        self.scaffold = scaffold
        # get the start time
        start_time = time.perf_counter()
        self.llm_timeout = max_llm_time

        # if self.llm_name is not gpt or sonnet, disable fn calling
        support_fn_calling = (
            "gpt" in self.llm_name
            or "sonnet" in self.llm_name
            or "o3" in self.llm_name
            or "o4" in self.llm_name
            or 'Coder' in self.llm_name
            or "GLM" in self.llm_name
        )
        self.use_fn_calling = use_fn_calling and support_fn_calling
        self.logger.warning(f"Using fn calling: {self.use_fn_calling}")

        # Log the environment and agent
        self.logger.info(f"Running agent {self.name} in environment {env}.")

        # Reset the environment and the agent
        env.reset()
        env.add_commands(self.command_files)
        self.reset()

        # Prepare problem_statement and structure from the environment
        problem_statement = env.runtime.get_task_instruction()
        self.logger.info(f"Problem Statement: {problem_statement}")
        gt_patch = env.runtime.commit.get_patch(test_file=True, non_test_file=False)

        # get system and instance prompts
        system_prompt = self.system_prompt_template
        user_prompt = self.instance_prompt_template.format(
            problem_statement=problem_statement,
            gt_patch=gt_patch,
            working_dir='/testbed',
            # base_commit=env.runtime.ds['base_commit'],
            test_patch_hint=metadata.get("test_patch_hint", ""),
            candidate_patch=metadata.get("candidate_patch", ""),
            candidate_patch_correctness=(
                "correct"
                if metadata.get("candidate_patch_correctness", False)
                else "incorrect"
            ),
        )
        self.logger.info(f"User Prompt: {user_prompt}")

        if self.args.use_demo:
            with open(self.args.demo_file, "r") as file:
                demo = file.read()
            user_prompt = f"{demo}\n\n{user_prompt}"
        self.logger.info(f"User Prompt with demo: {user_prompt}")
        initial_user_prompt = user_prompt

        # initialize the history
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # initialize the parameters
        obs = None
        done = False
        step_count = 0
        total_time_traj = 0
        self.trajectory_steps: List[TrajectoryStep] = []
        current_summary = ""  # Reusable summary from last compression
        prefill_summary = ""  # Summary used for prefill (if any)
        compress_steps = []  # Steps to be compressed in next summary
        append_steps = []  # Steps to be appended
        prefill_append_steps = []  # Steps to be appended for prefill (if any)
        last_summary_step_idx = -1  # Index of the last step that was summarized

        # for interleaved memory compression: store the async task for next step's memory compression
        pending_memory_task = None
        pending_prefill_task = None

        # Create or fetch the event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        traj_start_time = time.perf_counter()
        # agent loop
        while not done:
            
            ###### INTERLEAVED EXECUTION ######
            # Step 1: Retrieve summaries from previous step's task (if available)
            if pending_memory_task is not None:
                try:
                    ready_summary, summary_tokens, summary_exec_time, finish_reason, summary_start_time, summary_end_time = loop.run_until_complete(pending_memory_task)
                    if not self.args.use_optimized_schedule:
                        current_summary = ready_summary  # update current summary immediately
                        append_steps = append_steps[-(len(self.trajectory_steps) - last_summary_step_idx):]  # keep only the steps after last summary
                    else:
                        # for optimized schedule, we only update the prefill summary here
                        prefill_summary = ready_summary
                        prefill_append_steps = append_steps[-(len(self.trajectory_steps) - last_summary_step_idx):]  # keep only the steps after last summary
                except Exception as e:
                    self.logger.error(f"Error retrieving memory task result: {e}")
                    ready_summary = ""
                    summary_tokens = 0
                    summary_exec_time = 0
                    finish_reason = None
                    summary_start_time = None
                    summary_end_time = None
                    exit_reason = "memory_compression_error"
                    done = True
                    break
            else:
                ready_summary = ""  # Summary retrieved from task
                summary_tokens = 0
                summary_exec_time = 0
                summary_prompt = ""
                summary_prompt_tokens = 0
                finish_reason = None
                summary_start_time = None
                summary_end_time = None

            if pending_prefill_task is not None:
                try:
                    _, prefill_exec_time, prefill_start_time, prefill_end_time = loop.run_until_complete(pending_prefill_task)
                except Exception as e:
                    self.logger.error(f"Error retrieving prefill task result: {e}")
                    prefill_exec_time = 0
                    done = True
                    exit_reason = "prefill_error"
                    break
            else:
                prefill_exec_time = 0

            should_compress_this_step = (len(self.trajectory_steps) + 1) % self.args.retain_most_turns == 0 and len(self.trajectory_steps) > 0

            # Step 2: Immediately kick off memory compression for the CURRENT contexts so it can
            # overlap with response generation below.
            if should_compress_this_step:
                # Use run_in_executor to run synchronous function in parallel
                summary_messages, summary_prompt, summary_prompt_tokens = self._prepare_messages_for_summary(
                    history_to_compress=self.prepare_history_to_compress(compress_steps)
                )
                if summary_prompt_tokens > max_mem_token_limit:
                    exit_reason = "token_limit_exceeded_before_memory_compression"
                    done = True
                    break

                # Start the async memory compression task for the NEXT step
                pending_memory_task = loop.create_task(
                    self._start_memory_compression(
                        messages=summary_messages,
                    )
                )
                last_summary_step_idx = len(self.trajectory_steps)
                if self.args.use_optimized_schedule:
                    # use the summary from last step
                    current_summary = prefill_summary
                    append_steps = prefill_append_steps
            else:
                if self.args.use_optimized_schedule:
                    if self.nosummary:
                        messages = self._prepare_messages_for_query(
                            problem_statement=problem_statement,
                            summary="",
                            history_to_append=self.prepare_history_to_append(prefill_append_steps),
                            stepcount_message="",
                        )
                    else:
                        messages = self._prepare_messages_for_query(
                            problem_statement=problem_statement,
                            summary=prefill_summary,
                            history_to_append=self.prepare_history_to_append(prefill_append_steps),
                            stepcount_message="",
                        )
                    message_token_count = self._count_tokens(messages)
                    if message_token_count > max_token_limit:
                        exit_reason = "token_limit_exceeded_before_prefill"
                        done = True
                    pending_prefill_task = loop.create_task(
                        self._prefill_agent_model(
                            messages,
                            temperature,
                        )
                    )
                else:
                    pending_memory_task = None

            # Prepare the agent's message history
            # self.logger.info(isinstance(steps_remaining, int))
            # Add steps remaining message
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                stepcount_message = f"Steps Remaining: {steps_remaining}"
            else:
                stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
            history_to_append = self.prepare_history_to_append(append_steps)
            if self.nosummary:
                messages = self._prepare_messages_for_query(
                    problem_statement=problem_statement,
                    summary="",
                    history_to_append=history_to_append,
                    stepcount_message=stepcount_message,
                )
            else:
                messages = self._prepare_messages_for_query(
                    problem_statement=problem_statement,
                    summary=current_summary,
                    history_to_append=history_to_append,
                    stepcount_message=stepcount_message,
                )
            message_token_count = self._count_tokens(messages)
            if message_token_count > max_token_limit:
                exit_reason = "token_limit_exceeded_before_llm_query"
                done = True
                break

            # Now wait for the model query to complete
            try:
                response, llm_exec_time, llm_start_time, llm_end_time, vllm_metrics = loop.run_until_complete(
                    self._model_query(
                        messages,
                        temperature,
                    )
                )
            except ValueError as e:
                # Check if this is a token limit exceeded error
                if "Total tokens:" in str(e) and ">" in str(e):
                    self.logger.error(f"Error querying LLM: {e}")
                    self.logger.error(f"Error querying LLM: {traceback.format_exc()}")
                    done = True
                    exit_reason = "total_tokens_exceeded_llm"
                    break
                else:
                    # Re-raise other ValueErrors
                    raise
            except Exception as e:
                self.logger.error(f"Error querying LLM: {e}")
                self.logger.error(f"Error querying LLM: {traceback.format_exc()}")
                done = True
                exit_reason = "llm_query_error"
                break
                
            # Log total tokens in the response
            if hasattr(response, "usage"):
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)

                prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
                self.logger.warning(f"Prompt Token Details: {prompt_tokens_details}")
                self.logger.info(
                    f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}"
                )
            else:
                completion_tokens = -1
                prompt_tokens = -1
                total_tokens = -1
                total_tokens =  self._count_tokens(messages)
                self.logger.warning(
                    "No token usage information available in the response."
                )
            
            # Parse the LLM response to get 'thought' and 'action'
            self.response = response  # for debugging
            assistant_message = response.choices[0].message.content or ""
            self.logger.info(f"Assistant's message:\n{assistant_message}\n")

            if self.use_fn_calling:
                if "GLM" in self.llm_name:
                    thought, action = self.custom_parser_glm(response)
                else:
                    thought, action = self.custom_parser(response)
            else:
                thought, action = self.parse_response(assistant_message)

            action_str = action.to_xml_string()
            self.logger.info(f"THOUGHT:\n{thought}\n")
            self.logger.info(f"ACTION:\n{action.to_bashcmd()}\n")

            # Send the action to the environment
            try:
                obs, reward, done, info = env.step(action, timeout=max_exec_time)
                # env.runtime.commit_after_step(step_count)
            except Exception as e:
                obs = str(e)
                self.logger.error(f"Error during environment step: {obs}")
            
            env_exec_time = info["total_time"]
            total_step_time = max(llm_exec_time, summary_exec_time + prefill_exec_time) + env_exec_time
            total_time_traj += total_step_time
            step_count += 1  # Increment the step count

            if self.use_fn_calling:
                assistant_response = response.choices[0].message.dict()
                if assistant_response.get("tool_calls", None):
                    assistant_response["tool_calls"] = assistant_response["tool_calls"][
                        :1
                    ]  # only keep the first tool call
                self.history.append(assistant_response)
                # add tool response / user response to history
                try:
                    function_name = (
                        response.choices[0].message.tool_calls[0].function.name
                    )
                    function_id = response.choices[0].message.tool_calls[0].id
                    self.history.append(
                        {
                            "role": "tool",
                            "content": str(obs),
                            "name": function_name,
                            "tool_call_id": function_id,
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
                # assistant_message = f"{thought}\n\n{original_xml_str}"
                self.history.append({"role": "assistant", "content": assistant_message})
                self.history.append({"role": "user", "content": str(obs)})
            
            # Log the thought, action, and observation
            self.logger.info(f"OBSERVATION:\n{obs}\n")
            self.logger.info("-" * 50)

            # Check if the agent has reached limits or done
            # check if agent has finished naturally i.e. the agent uses the finish tool
            if done:
                if steps_remaining > 0:
                    self.logger.info(
                        f"Agent has finished naturally before step limit. current step count: {step_count}. max steps: {max_steps}."
                    )
                    exit_reason = "agent"
                elif steps_remaining == 0:
                    self.logger.info(
                        f"Agent finised on reaching the maximum number of steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "max_step_limit"
                else:
                    self.logger.info(
                        f"Agent has finished after continuing past the max steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "agent_max_step_limit"
            elif step_count >= max_steps_absolute:
                self.logger.info(
                    f"Agent reached max steps: {max_steps_absolute}. Exiting."
                )
                exit_reason = "abs_step_limit"
                done = True

            elif total_time_traj >= max_total_time:
                self.logger.info(f"Agent reached max time: {max_total_time}. Exiting.")
                exit_reason = "traj_time_limit"
                done = True

            # Create a TrajectoryStep object and append to the list
            trajectory_step = TrajectoryStep(
                # key parts
                step_idx=step_count - 1,
                thought=thought,
                action=action.to_xml_string(),
                observation=str(obs),
                done=done,
                info=info,  # also store the info to be safe
                summary=ready_summary,
                summary_prompt=summary_prompt,
                query_prompt=messages[1]["content"],
                history_to_append="",
                assistant_message="",
                # tokens
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_summary=summary_tokens,
                token_usage_summary_prompt=summary_prompt_tokens,
                token_usage_total=total_tokens,
                # metadata (current step stats)
                llm_exec_time=llm_exec_time,
                vllm_metrics=vllm_metrics,
                env_exec_time=env_exec_time,
                summ_exec_time=summary_exec_time,
                prefill_exec_time=prefill_exec_time,
                total_step_time=total_step_time,
                total_time_traj=total_time_traj,
                step_count=step_count,
                summary_finish_reason=finish_reason,
                summary_start_time=summary_start_time,
                summary_finish_time=summary_end_time,
                agent_start_time=llm_start_time,
                agent_finish_time=llm_end_time,
            )
            self.trajectory_steps.append(trajectory_step)

            compress_steps.append(trajectory_step)
            append_steps.append(trajectory_step)
        
        # get the output patch
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD")
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD -- . ':(exclude)pyproject.toml'")
        # env.runtime.soft_git_reset()

        # compute output patch cummulatively from the start using git diff from the initial commit
        output_patch = env.runtime.get_patch()

        # Create a Trajectory object
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
            exit_reason=exit_reason,  # reason for exiting. must be one of the [agent, max_step_limit, agent_max_step_limit, abs_step_limit, token_limit, traj_time_limit, llm_query_error]
            output_patch=output_patch,
            retain_most_turns=self.args.retain_most_turns,
            use_optimized_schedule=self.args.use_optimized_schedule,
            traj_start_time=traj_start_time,
            traj_finish_time=time.perf_counter(),
        )

        self.logger.info(f"Agent completed in {time.perf_counter() - start_time} seconds.")
        return self.trajectory