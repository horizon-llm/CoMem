"""
Agent for MiroFlow environments (BrowseComp-EN and similar web-research tasks).

Uses OpenAI-compatible function calling with tool schemas provided by
``MiroFlowEnv.get_tool_definitions()``.  Works with any ``litellm``-supported
model (GPT-4o, Claude, open-source via vLLM, etc.).
"""

import copy
import json
import os
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import litellm

from r2egym.agenthub.action import Action
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default prompts
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


# ---------------------------------------------------------------------------
# AgentArgs
# ---------------------------------------------------------------------------


@dataclass
class MiroFlowAgentArgs:
    llm_name: str
    llm_base_url: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    max_retries: int = 5
    timeout: int = 3000
    max_context_tokens: int = 128_000


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class MiroFlowAgent:
    """Function-calling agent for ``MiroFlowEnv``."""

    def __init__(self, name: str, args: MiroFlowAgentArgs, logger=None):
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

    # ---- helpers ---------------------------------------------------------

    def reset(self):
        self.trajectory_steps = []
        self.history = []
        self.final_answer = ""

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        try:
            return litellm.token_counter(model=self.llm_name, messages=messages)
        except Exception:
            return sum(len(str(m.get("content", ""))) // 4 for m in messages)

    # ---- LLM query -------------------------------------------------------

    def model_query(
        self,
        messages: List[Dict[str, str]],
        tools: List[dict],
        temperature: float = 0,
    ) -> Tuple[Any, float]:
        retries = 0
        start = time.time()
        using_local = "openai/" in self.llm_name or "hosted" in self.llm_name
        if using_local:
            litellm.api_key = None

        while retries < self.max_retries:
            try:
                kwargs: dict = {}
                if "o3" not in self.llm_name and "o4" not in self.llm_name:
                    kwargs["temperature"] = temperature
                response = litellm.completion(
                    model=self.llm_name,
                    messages=messages,
                    tools=tools,
                    timeout=self.llm_timeout,
                    api_base=self.llm_base_url,
                    **kwargs,
                )
                return response, time.time() - start
            except Exception as e:
                self.logger.error(f"LLM query failed @ retry {retries}: {e}")
                retries += 1
                if "RateLimitError" in str(e):
                    time.sleep(60)
                if retries >= self.max_retries:
                    raise
        raise RuntimeError("Unreachable")

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
        env,  # MiroFlowEnv
        max_steps: int = 30,
        max_steps_absolute: int = 50,
        max_exec_time: int = 300,
        max_total_time: int = 7200,
        max_token_limit: int = 128_000,
        temperature: float = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Run the agent loop.  Returns a ``Trajectory`` object."""
        start_time = time.time()

        # Reset env + agent
        env.reset()
        self.reset()

        # Tool definitions from the environment
        tools = env.get_tool_definitions()

        # Build initial messages
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

        while not done:
            steps_remaining = max_steps - step_count

            # Append step-count hint to the last user/tool message
            if steps_remaining > 0:
                step_msg = f"\n[Steps remaining: {steps_remaining}]"
            else:
                step_msg = (
                    "\n[Maximum steps reached. Submit your answer NOW "
                    "using submit_answer.]"
                )

            messages = copy.deepcopy(self.history)
            if messages[-1]["role"] in ("user", "tool"):
                messages[-1]["content"] = (
                    str(messages[-1].get("content", "")) + step_msg
                )

            # Token guard
            total_tokens = self._count_tokens(messages)
            if total_tokens > max_token_limit:
                self.logger.error(
                    f"Token limit: {total_tokens} > {max_token_limit}"
                )
                exit_reason = "token_limit"
                break

            # LLM call
            try:
                response, llm_exec_time = self.model_query(
                    messages, tools, temperature
                )
            except Exception as e:
                self.logger.error(f"LLM error: {e}\n{traceback.format_exc()}")
                exit_reason = "llm_query_error"
                break

            # Token usage
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", -1) if usage else -1
            completion_tokens = (
                getattr(usage, "completion_tokens", -1) if usage else -1
            )
            total_tokens = getattr(usage, "total_tokens", -1) if usage else -1

            # Parse
            thought, action = self.parse_response(response)
            self.logger.info(
                f"Step {step_count}: {action.function_name}"
                f"({json.dumps(action.parameters, ensure_ascii=False)[:200]})"
            )
            if thought:
                self.logger.info(f"  Thought: {thought[:300]}")

            # Execute
            try:
                obs, reward, done, info = env.step(action, timeout=max_exec_time)
            except Exception as e:
                obs = str(e)
                info = {"total_time": 0.0}
                self.logger.error(f"Env step error: {e}")

            # Log observation summary
            obs_str = str(obs)
            if action.function_name == "google_search":
                try:
                    obs_data = json.loads(obs_str)
                    organic = obs_data.get("organic", [])
                    if organic:
                        titles = [r.get("title", "")[:60] for r in organic[:3]]
                        self.logger.info(f"  Search: {len(organic)} results. Top: {titles}")
                    elif "error" in obs_data:
                        self.logger.info(f"  Search error: {obs_data['error']}")
                    else:
                        self.logger.info("  Search: 0 results")
                except (json.JSONDecodeError, AttributeError):
                    self.logger.info(f"  Obs: {obs_str[:200]}")
            elif action.function_name == "scrape":
                try:
                    obs_data = json.loads(obs_str)
                    cc = obs_data.get("char_count", 0)
                    err = obs_data.get("error")
                    if err:
                        self.logger.info(f"  Scrape error: {err}")
                    else:
                        preview = obs_data.get("content", "")[:120].replace("\n", " ")
                        self.logger.info(f"  Scraped {cc} chars: {preview}...")
                except (json.JSONDecodeError, AttributeError):
                    self.logger.info(f"  Obs: {obs_str[:200]}")
            elif action.function_name == "submit_answer":
                self.logger.info(f"  Answer: {action.parameters.get('answer', '')[:200]}")
            else:
                self.logger.info(f"  Obs: {obs_str[:200]}")

            env_exec_time = info.get("total_time", 0.0)
            total_step_time = llm_exec_time + env_exec_time
            total_time_traj += total_step_time
            step_count += 1

            # Update history (function-calling format)
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
                self.history.append(
                    {
                        "role": "tool",
                        "content": str(obs),
                        "name": fn_name,
                        "tool_call_id": fn_id,
                    }
                )
            except (AttributeError, IndexError, TypeError):
                self.history.append({"role": "user", "content": str(obs)})

            # Capture final answer
            if action.function_name == "submit_answer":
                self.final_answer = action.parameters.get("answer", "")

            # Exit checks
            if done:
                exit_reason = (
                    "agent" if steps_remaining > 0 else "max_step_limit"
                )
            elif step_count >= max_steps_absolute:
                exit_reason = "abs_step_limit"
                done = True
            elif total_time_traj >= max_total_time:
                exit_reason = "traj_time_limit"
                done = True
            elif total_tokens >= max_token_limit:
                exit_reason = "token_limit"
                done = True

            # Record trajectory step
            self.trajectory_steps.append(
                TrajectoryStep(
                    step_idx=step_count - 1,
                    thought=thought,
                    action=action.to_xml_string(),
                    observation=str(obs),
                    done=done,
                    info=info,
                    token_usage_prompt=prompt_tokens,
                    token_usage_completion=completion_tokens,
                    token_usage_total=total_tokens,
                    llm_exec_time=llm_exec_time,
                    env_exec_time=env_exec_time,
                    total_step_time=total_step_time,
                    total_time_traj=total_time_traj,
                    step_count=step_count,
                )
            )

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
        )

        elapsed = time.time() - start_time
        self.logger.info(
            f"Completed in {elapsed:.1f}s | {step_count} steps | exit={exit_reason}"
        )
        return self.trajectory
