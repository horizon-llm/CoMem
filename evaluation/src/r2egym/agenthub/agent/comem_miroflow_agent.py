"""
CoMeM (Compressed Memory) agent for MiroFlow / BrowseComp tasks.

Identical interaction loop to ``MiroFlowAgent`` but replaces the full
conversation history with a compressed summary (produced by a memory model)
+ a small window of recent verbatim turns, exactly matching the rollout
format used during GRPO training.
"""

import copy
import json
import os
import time
import traceback
import asyncio
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import litellm
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from r2egym.agenthub.action import Action
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.trajectory import TrajectoryStep_Summ as TrajectoryStep
from r2egym.agenthub.trajectory import Trajectory_Summ as Trajectory

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompts — identical to MiroFlowAgent (miroflow_agent.py:29-52)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are a research agent that finds precise answers to questions by searching the web.

You have access to the following tools:
- **google_search**: Search Google for information.
- **scrape**: Read the content of a web page / PDF / file given its URL.
- **submit_answer**: Submit your final answer once you are confident.

## Guidelines
1. Start by searching Google with a well-crafted query.
2. Read promising links with `scrape` to gather evidence.
3. If the first search is insufficient, refine your query and search again.
4. When you have found the answer, call `submit_answer` immediately.
5. Be precise and concise – do not include units unless the question asks for them.
6. For lists use comma-separated values.  No trailing punctuation.
"""

DEFAULT_USER_PROMPT = """\
Find the answer to the following question by searching the web:

{task_question}

Search thoroughly, verify from multiple sources if possible, and submit your \
final answer using the `submit_answer` tool."""

# Compressor prompt — identical to training (browsecomp_fn_calling.yaml / rollout collector)
COMPRESSOR_PROMPT = """\
You are a helpful assistant that summarizes conversations for another model to use. You need to make sure that the summary captures all important details from the conversation so that the other model can understand the context just like reading the full conversation.

### Conversation:
{conversation}

Please provide a detailed summary of the above conversation. Make sure to include all necessary details for the other model to understand the context, especially entities, actions taken, and outcomes. But NEVER include any suggestions or recommendations for future actions—only summarize what has already occurred."""

# Query prompt — identical to training (browsecomp_fn_calling.yaml)
QUERY_PROMPT = """\
Find the answer to the following question by searching the web:

{problem_statement}

Search thoroughly, verify from multiple sources if possible, and submit your final answer using the `submit_answer` tool. Please refer to the summary and recent messages for context, which contains our previous search attempts and what you have found so far.

Here are the summary of our previous conversations:
<summary>
{summary}
</summary>
Here are the most recent messages exchanged that are not included in the summary:
<recent_messages>
{recent_messages}
</recent_messages>

VERY IMPORTANT: each response must include both reasoning and a function call."""

OPENAI_MODELS = [
    'gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo',
    'o1', 'o1-mini', 'o1-preview', 'gpt-5', 'gpt-5-mini', 'gpt-4.1',
]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

@dataclass
class CoMeMBrowseCompAgentArgs:
    llm_name: str
    llm_base_url: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    max_retries: int = 5
    timeout: int = 3000
    max_context_tokens: int = 128_000
    # Memory model
    memory_model_name: str = "gpt-3.5-turbo"
    memory_model_address: Optional[str] = None
    memory_model_temperature: float = 0.0
    memory_model_max_gen_tokens: int = 1536
    retain_most_turns: int = 1


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CoMeMBrowseCompAgent:
    """Function-calling BrowseComp agent with compressed-memory context management."""

    def __init__(self, name: str, args: CoMeMBrowseCompAgentArgs, logger=None):
        self.name = name
        self.args = args
        self.llm_name = args.llm_name
        self.llm_base_url = args.llm_base_url or os.environ.get("LLM_BASE_URL")
        self.max_retries = args.max_retries
        self.llm_timeout = args.timeout
        self.max_context_tokens = args.max_context_tokens
        self.system_prompt_template = args.system_prompt or DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = args.user_prompt or DEFAULT_USER_PROMPT

        if logger is None:
            self.logger = get_logger(name)
        else:
            self.logger = logger

        self.trajectory_steps: List[TrajectoryStep] = []
        self.history: List[Dict[str, Any]] = []
        self.final_answer: str = ""

        # Agent LLM async client (matches comem_agent_kso.py)
        self.agent_client = AsyncOpenAI(
            base_url=self.llm_base_url, api_key="dummy"
        )

        # Memory model setup
        self.mem_tokenizer = AutoTokenizer.from_pretrained(args.memory_model_name)
        self._setup_memory_client()

    def _setup_memory_client(self):
        model_name = self.args.memory_model_name
        if model_name in OPENAI_MODELS:
            self.mem_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self._mem_use_openai = True
        else:
            addr = self.args.memory_model_address
            if not addr:
                raise ValueError(
                    f"Memory model '{model_name}' is not an OpenAI model but no "
                    "memory_model_address provided."
                )
            self.mem_client = AsyncOpenAI(api_key="dummy", base_url=addr)
            self._mem_use_openai = False

    def reset(self):
        self.trajectory_steps = []
        self.history = []
        self.final_answer = ""

    # ---- token counting ---------------------------------------------------

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            return litellm.token_counter(model=self.llm_name, messages=messages)
        except Exception:
            return sum(len(str(m.get("content", ""))) // 4 for m in messages)

    def _count_summary_tokens(self, summary: str) -> int:
        return len(self.mem_tokenizer.encode(summary))

    # ---- memory compression -----------------------------------------------

    # Max characters per observation in history — matches the training rollout
    # collector. BrowseComp scrape results can be full web pages; cap them to
    # keep the compressor prompt within the context budget.
    MAX_OBS_CHARS = 4000

    @staticmethod
    def _truncate_obs(obs: str, limit: int) -> str:
        if len(obs) <= limit:
            return obs
        half = limit // 2
        return obs[:half] + "\n\n... [truncated] ...\n\n" + obs[-half:]

    def prepare_history_to_compress(self, compress_steps: list) -> str:
        """Format steps for the compressor prompt.
        Matches COMEMAgent.prepare_history_to_compress in comem_agent_kso.py."""
        history = f"SYSTEM PROMPT:\n```\n{self.history[0]['content']}\n```\n"
        history += f"INITIAL USER PROMPT:\n```\n{self.history[1]['content']}\n```\n"
        for step in compress_steps:
            obs = self._truncate_obs(step.observation, self.MAX_OBS_CHARS)
            history += f'THOUGHT:\n```\n{step.thought}\n```\n'
            history += f'ACTION:\n```\n{step.action}\n```\n'
            history += f"\nOBSERVATION:\n```\n{obs}\n```\n"
        return history

    def prepare_history_to_append(self, append_steps: list) -> str:
        """Format recent steps as text.
        Matches COMEMAgent.prepare_history_to_append in comem_agent_kso.py."""
        history = ""
        for step in append_steps:
            obs = self._truncate_obs(step.observation, self.MAX_OBS_CHARS)
            history += f'THOUGHT:\n```\n{step.thought}\n```\n'
            history += f'ACTION:\n```\n{step.action}\n```\n'
            history += f'OBSERVATION:\n```\n{obs}\n```\n'
        return history

    # ---- memory compression ------------------------------------------------

    def _prepare_messages_for_summary(self, history_to_compress: str):
        """Build the compressor prompt messages + count tokens.
        Matches COMEMAgent._prepare_messages_for_summary in comem_agent_kso.py."""
        summary_prompt = COMPRESSOR_PROMPT.format(conversation=history_to_compress)
        messages = [{"role": "user", "content": summary_prompt}]
        summary_prompt_tokens = self._count_tokens(messages)
        return messages, summary_prompt, summary_prompt_tokens

    async def _start_memory_compression(self, messages: list):
        """Run memory model to compress history.
        Matches COMEMAgent._start_memory_compression in comem_agent_kso.py."""
        call_params = {
            "model": self.args.memory_model_name,
            "messages": messages,
            "n": 1,
        }
        if not self._mem_use_openai:
            call_params["temperature"] = self.args.memory_model_temperature
            call_params["max_tokens"] = self.args.memory_model_max_gen_tokens

        start = time.perf_counter()
        response = await self.mem_client.chat.completions.create(**call_params)
        end = time.perf_counter()

        text = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason
        summary_tokens = self._count_summary_tokens(text)

        return text, summary_tokens, end - start, finish_reason, start, end

    # ---- message construction (matches comem_agent_kso.py) -------------------

    def _prepare_messages_for_query(
        self, summary: str, history_to_append: str,
        stepcount_message: str, task_question: str,
    ) -> List[Dict[str, str]]:
        """Build messages for the agent LLM.

        When summary exists: 2-message format [system, user_with_query_prompt].
        When no summary:     full multi-turn conversation history.
        Matches COMEMAgent._prepare_messages_for_query in comem_agent_kso.py.
        """
        if summary.strip():
            formatted_prompt = QUERY_PROMPT.format(
                problem_statement=task_question,
                summary=summary,
                recent_messages=history_to_append,
            )
            formatted_prompt += f"\n{stepcount_message}\n"
            messages = [
                {"role": "system", "content": self.history[0]["content"]},
                {"role": "user", "content": formatted_prompt},
            ]
        else:
            # No summary yet — use full conversation history
            messages = copy.deepcopy(self.history)
        return messages

    # ---- LLM query (async, matches comem_agent_kso.py) ----------------------

    async def _model_query(
        self, messages, tools, temperature=0,
    ):
        retries = 0
        messages_ = copy.deepcopy(messages)

        # Strip "hosted_vllm/" or "openai/" prefix for vLLM
        vllm_llm_name = "/".join(self.llm_name.split("/")[1:])

        start_time = time.perf_counter()
        while retries < self.max_retries:
            try:
                kwargs = {}
                if "o3" not in self.llm_name and "o4" not in self.llm_name:
                    kwargs["temperature"] = temperature

                if "GLM" in vllm_llm_name:
                    response = await self.agent_client.chat.completions.create(
                        model=vllm_llm_name,
                        tools=tools,
                        messages=messages_,
                        timeout=self.llm_timeout,
                        max_tokens=10000,
                        extra_body={
                            "thinking": {
                                "type": "enabled",
                                "clear_thinking": False,
                            }
                        },
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
                self.logger.error(f"LLM query failed @ retry {retries}: {e}")
                retries += 1
                if "RateLimitError" in str(e):
                    time.sleep(60)
                if retries >= self.max_retries:
                    raise

        end_time = time.perf_counter()
        exec_time = end_time - start_time
        return response, exec_time

    # ---- response parsing ------------------------------------------------

    @staticmethod
    def parse_response(response) -> Tuple[str, Action]:
        thought = response.choices[0].message.content or ""
        try:
            tc = response.choices[0].message.tool_calls[0]
            fn_name = tc.function.name
            params = json.loads(tc.function.arguments)
            return thought, Action(function_name=fn_name, parameters=params)
        except (AttributeError, IndexError, TypeError, json.JSONDecodeError):
            return thought, Action(function_name="", parameters={})

    # ---- main loop -------------------------------------------------------

    def run(
        self,
        env,
        max_steps: int = 30,
        max_steps_absolute: int = 50,
        max_exec_time: int = 300,
        max_total_time: int = 7200,
        max_token_limit: int = 128_000,
        temperature: float = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        start_time = time.time()
        env.reset()
        self.reset()

        tools = env.get_tool_definitions()
        system_prompt = self.system_prompt_template
        task_question = env.get_task_instruction()
        user_prompt = self.user_prompt_template.format(task_question=task_question)

        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        done = False
        step_count = 0
        total_time_traj = 0.0
        exit_reason = "unknown"
        retain_most_turns = self.args.retain_most_turns

        # KSO state — matches comem_agent_kso.py
        current_summary = ""
        compress_steps: List[TrajectoryStep] = []
        append_steps: List[TrajectoryStep] = []
        last_summary_step_idx = -1
        summary_prompt = ""
        summary_prompt_tokens = 0

        # Event loop for async memory compression
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        pending_memory_task = None

        while not done:
            ###### INTERLEAVED EXECUTION (matches comem_agent_kso.py) ######

            # Step 1: Retrieve summary from previous step's async task
            if pending_memory_task is not None:
                try:
                    (ready_summary, summary_tokens, summary_exec_time,
                     finish_reason, summary_start_time, summary_end_time) = \
                        loop.run_until_complete(pending_memory_task)
                    current_summary = ready_summary
                    # Trim append_steps to only steps after last summary
                    append_steps = append_steps[-(len(self.trajectory_steps) - last_summary_step_idx):]
                except Exception as e:
                    self.logger.error(f"Error retrieving memory task: {e}")
                    ready_summary = ""
                    summary_tokens = 0
                    summary_exec_time = 0.0
                    finish_reason = None
                    exit_reason = "memory_compression_error"
                    done = True
                    break
            else:
                ready_summary = ""
                summary_tokens = 0
                summary_exec_time = 0.0
                finish_reason = None

            # Step 2: K-step-off — only compress every retain_most_turns steps
            should_compress = (
                (len(self.trajectory_steps) + 1) % retain_most_turns == 0
                and len(self.trajectory_steps) > 0
            )

            if should_compress:
                summary_messages, summary_prompt, summary_prompt_tokens = \
                    self._prepare_messages_for_summary(
                        self.prepare_history_to_compress(compress_steps)
                    )
                if summary_prompt_tokens > max_token_limit:
                    exit_reason = "token_limit_exceeded_before_memory_compression"
                    done = True
                    break

                pending_memory_task = loop.create_task(
                    self._start_memory_compression(summary_messages)
                )
                last_summary_step_idx = len(self.trajectory_steps)
            else:
                pending_memory_task = None

            # Step 3: Build messages and query agent LLM
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                stepcount_message = f"Steps Remaining: {steps_remaining}"
            else:
                stepcount_message = (
                    "You have reached the maximum number of steps. "
                    "Please submit your answer NOW."
                )

            history_to_append = self.prepare_history_to_append(append_steps)
            messages = self._prepare_messages_for_query(
                summary=current_summary,
                history_to_append=history_to_append,
                stepcount_message=stepcount_message,
                task_question=task_question,
            )

            # Token guard
            message_token_count = self._count_tokens(messages)
            if message_token_count > max_token_limit:
                self.logger.error(
                    f"Token limit: {message_token_count} > {max_token_limit}"
                )
                exit_reason = "token_limit_exceeded_before_llm_query"
                done = True
                break

            # LLM call (async, matches comem_agent_kso.py)
            try:
                response, llm_exec_time = loop.run_until_complete(
                    self._model_query(messages, tools, temperature)
                )
            except ValueError as e:
                if "Total tokens:" in str(e) and ">" in str(e):
                    self.logger.error(f"LLM error: {e}\n{traceback.format_exc()}")
                    exit_reason = "total_tokens_exceeded_llm"
                    done = True
                    break
                else:
                    raise
            except Exception as e:
                self.logger.error(f"LLM error: {e}\n{traceback.format_exc()}")
                exit_reason = "llm_query_error"
                done = True
                break

            # Token usage
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", -1) if usage else -1
            completion_tokens = getattr(usage, "completion_tokens", -1) if usage else -1
            total_tokens_resp = getattr(usage, "total_tokens", -1) if usage else -1

            # Parse
            thought, action = self.parse_response(response)
            self.logger.info(
                f"Step {step_count}: {action.function_name}"
                f"({json.dumps(action.parameters, ensure_ascii=False)[:200]})"
            )

            # Execute
            try:
                obs, reward, done, info = env.step(action, timeout=max_exec_time)
            except Exception as e:
                obs = str(e)
                info = {"total_time": 0.0}
                self.logger.error(f"Env step error: {e}")

            env_exec_time = info.get("total_time", 0.0)
            total_step_time = max(llm_exec_time, summary_exec_time) + env_exec_time
            total_time_traj += total_step_time
            step_count += 1

            # Update full conversation history
            assistant_msg = response.choices[0].message
            try:
                assistant_dict = assistant_msg.model_dump()
            except AttributeError:
                assistant_dict = assistant_msg.dict()
            if assistant_dict.get("tool_calls"):
                assistant_dict["tool_calls"] = assistant_dict["tool_calls"][:1]
            self.history.append(assistant_dict)

            try:
                fn_name = assistant_msg.tool_calls[0].function.name
                fn_id = assistant_msg.tool_calls[0].id
                self.history.append({
                    "role": "tool",
                    "content": str(obs),
                    "name": fn_name,
                    "tool_call_id": fn_id,
                })
            except (AttributeError, IndexError, TypeError):
                self.history.append({"role": "user", "content": str(obs)})

            # Capture final answer
            if action.function_name == "submit_answer":
                self.final_answer = action.parameters.get("answer", "")

            # Exit checks
            if done:
                exit_reason = "agent" if steps_remaining > 0 else "max_step_limit"
            elif step_count >= max_steps_absolute:
                exit_reason = "abs_step_limit"
                done = True
            elif total_time_traj >= max_total_time:
                exit_reason = "traj_time_limit"
                done = True
            elif total_tokens_resp >= max_token_limit:
                exit_reason = "token_limit"
                done = True

            # Record trajectory step
            trajectory_step = TrajectoryStep(
                step_idx=step_count - 1,
                thought=thought,
                action=action.to_xml_string(),
                observation=str(obs),
                done=done,
                info=info,
                summary=ready_summary,
                summary_prompt=summary_prompt,
                query_prompt=messages[1]["content"] if current_summary else "",
                history_to_append="",
                assistant_message="",
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_summary=summary_tokens,
                token_usage_summary_prompt=summary_prompt_tokens,
                token_usage_total=total_tokens_resp,
                llm_exec_time=llm_exec_time,
                env_exec_time=env_exec_time,
                summ_exec_time=summary_exec_time,
                total_step_time=total_step_time,
                total_time_traj=total_time_traj,
                step_count=step_count,
                summary_finish_reason=finish_reason,
            )
            self.trajectory_steps.append(trajectory_step)

            # KSO: accumulate into both lists (matches comem_agent_kso.py:967-968)
            compress_steps.append(trajectory_step)
            append_steps.append(trajectory_step)

        # Build Trajectory
        conversation = json.loads(json.dumps(self.history, default=str))

        self.trajectory = Trajectory(
            trajectory_steps=[s.model_dump() for s in self.trajectory_steps],
            problem_statement=task_question,
            docker_image="miroflow/browsecomp",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            conversation_history=conversation,
            env_args={
                "task_id": env.args.task_id,
                "ground_truth": env.args.ground_truth,
            },
            agent_args=asdict(self.args),
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_token_limit,
            max_llm_time=self.llm_timeout,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            exit_reason=exit_reason,
            output_patch="",
            retain_most_turns=retain_most_turns,
        )

        elapsed = time.time() - start_time
        self.logger.info(
            f"Completed in {elapsed:.1f}s | {step_count} steps | exit={exit_reason}"
        )
        return self.trajectory
