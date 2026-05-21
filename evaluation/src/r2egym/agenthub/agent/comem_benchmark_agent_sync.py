"""
Synchronous benchmark agent for COMEM-KSO latency measurement.

Subclass of BenchmarkCOMEMAgent that replaces the asynchronous interleaved
execution with strictly sequential execution:
  1. Memory compression runs to completion FIRST
  2. Agent model query runs AFTER memory compression finishes

This isolates the contribution of the async systems design from the learned
memory compression itself, answering the question:
  "How much of the reported efficiency improvement comes from the asynchronous
   systems design versus the learned memory compression itself?"

The ONLY difference from the async version is the execution ordering in run().
All other logic (padding, token control, parsing, trajectory recording) is
identical.
"""

import copy
import json
import time
import requests
import asyncio
import traceback
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from r2egym.agenthub.agent.comem_benchmark_agent import BenchmarkCOMEMAgent
from r2egym.agenthub.agent.comem_agent_kso import AgentArgs
from r2egym.agenthub.trajectory import TrajectoryStep_Summ as TrajectoryStep
from r2egym.agenthub.trajectory import Trajectory_Summ as Trajectory


class SyncBenchmarkCOMEMAgent(BenchmarkCOMEMAgent):
    """COMEM-KSO benchmark agent with strictly sequential (synchronous) execution.

    Memory compression and agent model queries never overlap. This lets us
    measure the pure cost of memory compression without any async hiding.

    Overrides only run() — all other methods (padding, token control, parsing,
    memory compression, model query) are inherited unchanged.
    """

    def run(
        self,
        env,
        use_fn_calling: bool = True,
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        max_token_limit: int = 65536,
        max_mem_token_limit: int = 131072,
        max_exec_time: int = 90,
        max_total_time: int = 50000,
        max_llm_time: int = 7200,
        temperature=0,
        metadata: Optional[Dict[str, Any]] = {},
        scaffold: str = "r2egym",
    ):
        """Run the agent loop with SYNCHRONOUS memory compression.

        This is a modified copy of COMEMAgent.run() where memory compression
        runs to completion before the agent model query, instead of overlapping.

        total_step_time = summary_exec_time + llm_exec_time + env_exec_time
        (sum instead of max, because nothing overlaps)
        """
        assert scaffold in ["r2egym", "openhands", "sweagent"]
        self.scaffold = scaffold
        start_time = time.perf_counter()
        self.llm_timeout = max_llm_time

        support_fn_calling = (
            "gpt" in self.llm_name
            or "sonnet" in self.llm_name
            or "o3" in self.llm_name
            or "o4" in self.llm_name
            or "Coder" in self.llm_name
            or "GLM" in self.llm_name
        )
        self.use_fn_calling = use_fn_calling and support_fn_calling
        self.logger.warning(f"Using fn calling: {self.use_fn_calling}")
        self.logger.warning("Running in SYNCHRONOUS mode (no async overlap)")

        self.logger.info(f"Running agent {self.name} in environment {env}.")

        env.reset()
        env.add_commands(self.command_files)
        self.reset()

        problem_statement = env.runtime.get_task_instruction()
        self.logger.info(f"Problem Statement: {problem_statement}")
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
        self.logger.info(f"User Prompt: {user_prompt}")

        if self.args.use_demo:
            with open(self.args.demo_file, "r") as file:
                demo = file.read()
            user_prompt = f"{demo}\n\n{user_prompt}"
        self.logger.info(f"User Prompt with demo: {user_prompt}")
        initial_user_prompt = user_prompt

        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        obs = None
        done = False
        step_count = 0
        total_time_traj = 0
        self.trajectory_steps: List[TrajectoryStep] = []
        current_summary = ""
        compress_steps = []
        append_steps = []
        last_summary_step_idx = -1
        exit_reason = "unknown"

        # Create or fetch the event loop (still needed for awaiting async methods)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        traj_start_time = time.perf_counter()

        while not done:
            ###### SYNCHRONOUS EXECUTION ######
            # Step 1: Run memory compression SYNCHRONOUSLY (if needed)
            should_compress_this_step = (
                (len(self.trajectory_steps) + 1) % self.args.retain_most_turns == 0
                and len(self.trajectory_steps) > 0
            )

            summary_prompt = ""
            summary_prompt_tokens = 0
            ready_summary = ""
            summary_tokens = 0
            summary_exec_time = 0
            finish_reason = None
            summary_start_time = None
            summary_end_time = None

            if should_compress_this_step:
                summary_messages, summary_prompt, summary_prompt_tokens = (
                    self._prepare_messages_for_summary(
                        history_to_compress=self.prepare_history_to_compress(compress_steps)
                    )
                )
                if summary_prompt_tokens > max_mem_token_limit:
                    exit_reason = "token_limit_exceeded_before_memory_compression"
                    done = True
                    break

                # SYNCHRONOUS: run memory compression to completion BEFORE agent query
                try:
                    (
                        ready_summary,
                        summary_tokens,
                        summary_exec_time,
                        finish_reason,
                        summary_start_time,
                        summary_end_time,
                    ) = loop.run_until_complete(
                        self._start_memory_compression(messages=summary_messages)
                    )
                    current_summary = ready_summary
                    last_summary_step_idx = len(self.trajectory_steps)
                    append_steps = append_steps[
                        -(len(self.trajectory_steps) - last_summary_step_idx) :
                    ]
                except Exception as e:
                    self.logger.error(f"Error in memory compression: {e}")
                    exit_reason = "memory_compression_error"
                    done = True
                    break

            # Step 2: Prepare and run the agent model query (AFTER memory is done)
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

            try:
                response, llm_exec_time, llm_start_time, llm_end_time, vllm_metrics = (
                    loop.run_until_complete(self._model_query(messages, temperature))
                )
            except ValueError as e:
                if "Total tokens:" in str(e) and ">" in str(e):
                    self.logger.error(f"Error querying LLM: {e}")
                    done = True
                    exit_reason = "total_tokens_exceeded_llm"
                    break
                else:
                    raise
            except Exception as e:
                self.logger.error(f"Error querying LLM: {e}")
                self.logger.error(f"Error querying LLM: {traceback.format_exc()}")
                done = True
                exit_reason = "llm_query_error"
                break

            # Log token usage
            if hasattr(response, "usage"):
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)
                self.logger.info(
                    f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}"
                )
            else:
                completion_tokens = -1
                prompt_tokens = -1
                total_tokens = self._count_tokens(messages)
                self.logger.warning("No token usage information available in the response.")

            # Parse the response
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
            except Exception as e:
                obs = str(e)
                self.logger.error(f"Error during environment step: {obs}")

            env_exec_time = info["total_time"]

            # SYNCHRONOUS: total_step_time is the SUM (no overlap)
            total_step_time = llm_exec_time + summary_exec_time + env_exec_time
            total_time_traj += total_step_time
            step_count += 1

            # Update history
            if self.use_fn_calling:
                assistant_response = response.choices[0].message.dict()
                if assistant_response.get("tool_calls", None):
                    assistant_response["tool_calls"] = assistant_response["tool_calls"][:1]
                self.history.append(assistant_response)
                try:
                    function_name = response.choices[0].message.tool_calls[0].function.name
                    function_id = response.choices[0].message.tool_calls[0].id
                    self.history.append(
                        {
                            "role": "tool",
                            "content": str(obs),
                            "name": function_name,
                            "tool_call_id": function_id,
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Error logging tool response: {e}")
                    self.history.append({"role": "user", "content": str(obs)})
            else:
                assistant_message = f"{thought}\n\n{action.to_xml_string()}"
                self.history.append({"role": "assistant", "content": assistant_message})
                self.history.append({"role": "user", "content": str(obs)})

            self.logger.info(f"OBSERVATION:\n{obs}\n")
            self.logger.info("-" * 50)

            # Check termination conditions
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
                query_prompt=messages[1]["content"],
                history_to_append="",
                assistant_message="",
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_summary=summary_tokens,
                token_usage_summary_prompt=summary_prompt_tokens,
                token_usage_total=total_tokens,
                llm_exec_time=llm_exec_time,
                vllm_metrics=vllm_metrics,
                env_exec_time=env_exec_time,
                summ_exec_time=summary_exec_time,
                prefill_exec_time=0,  # no prefill in sync mode
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

        # Build output
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
            use_optimized_schedule=False,  # sync mode, no optimized schedule
            traj_start_time=traj_start_time,
            traj_finish_time=time.perf_counter(),
        )

        self.logger.info(f"Agent completed in {time.perf_counter() - start_time} seconds.")
        return self.trajectory
