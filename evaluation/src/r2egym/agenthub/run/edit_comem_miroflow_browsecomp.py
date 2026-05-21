"""
Run BrowseComp-EN evaluation using MiroFlowEnv + CoMeMBrowseCompAgent.

This is the CoMeM (compressed-memory) variant that uses a memory model to
summarize conversation history, matching the training rollout format exactly.

Usage:
    python -m r2egym.agenthub.run.edit_comem_miroflow_browsecomp \
        --data_path  data/browsecomp-test/standardized_data.jsonl \
        --llm_name   hosted_vllm/zai-org/GLM-4.7 \
        --memory_model_name  hosted_vllm/Qwen/Qwen3-4B \
        --memory_model_address http://localhost:8001/v1
"""

import json
import os
import time
import threading
import concurrent.futures
import traceback
from typing import Any, Dict, List, Optional

from fire import Fire

from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.environment.miroflow_env import MiroFlowEnv, MiroFlowEnvArgs
from r2egym.agenthub.agent.comem_miroflow_agent import (
    CoMeMBrowseCompAgent,
    CoMeMBrowseCompAgentArgs,
)

logger = get_logger(__name__)
file_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_browsecomp_tasks(
    data_path: str, max_tasks: int = None,
) -> List[Dict[str, Any]]:
    tasks = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    if max_tasks:
        tasks = tasks[:max_tasks]
    logger.info(f"Loaded {len(tasks)} task(s) from {data_path}")
    return tasks


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an expert evaluator. Compare the model's answer to the correct answer.

Question: {question}

Correct Answer: {ground_truth}

Model's Answer: {model_answer}

Decide whether the model's answer is semantically correct.
- Consider alternative phrasings, abbreviations, and equivalent expressions.
- Numerical answers must be within reasonable precision.
- The answer does NOT need to be an exact string match.

Respond with exactly one letter:
  A  – if the model's answer is CORRECT
  B  – if the model's answer is INCORRECT

Your response (A or B):"""


def evaluate_answer(
    model_answer: str,
    ground_truth: str,
    question: str,
    judge_model: str = "gpt-4.1",
) -> Dict[str, Any]:
    import litellm

    if not model_answer or not model_answer.strip():
        return {"judge_result": "B", "is_correct": False}

    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        model_answer=model_answer,
    )
    try:
        resp = litellm.completion(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2,
            temperature=0,
        )
        result = resp.choices[0].message.content.strip().upper()
        return {"judge_result": result, "is_correct": result.startswith("A")}
    except Exception as e:
        logger.error(f"Judge failed: {e}")
        return {"judge_result": "ERROR", "is_correct": False}


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_single_task(
    task: Dict[str, Any],
    agent_args: CoMeMBrowseCompAgentArgs,
    max_steps: int,
    max_steps_absolute: int,
    max_exec_time: int,
    max_total_time: int,
    temperature: float,
    output_dir: str,
    judge_model: str,
    latency_test: bool = False,
) -> Dict[str, Any]:
    task_id = task.get("task_id", "unknown")
    task_question = task.get("task_question", "")
    ground_truth = task.get("ground_truth", "")

    logger.info(f"Task {task_id}")

    env_args = MiroFlowEnvArgs(
        task_id=task_id,
        task_question=task_question,
        ground_truth=ground_truth,
    )
    env = MiroFlowEnv(env_args)
    agent = CoMeMBrowseCompAgent(name=f"comem_{task_id}", args=agent_args)

    try:
        trajectory = agent.run(
            env=env,
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            temperature=temperature,
        )
        final_answer = agent.final_answer or ""

        eval_result = evaluate_answer(
            model_answer=final_answer,
            ground_truth=ground_truth,
            question=task_question,
            judge_model=judge_model,
        )

        result: Dict[str, Any] = {
            "task_id": task_id,
            "task_question": task_question,
            "ground_truth": ground_truth,
            "model_answer": final_answer,
            "is_correct": eval_result["is_correct"],
            "judge_result": eval_result["judge_result"],
            "num_steps": len(agent.trajectory_steps),
            "exit_reason": trajectory.exit_reason,
        }

        if not latency_test:
            os.makedirs(output_dir, exist_ok=True)
            traj_path = os.path.join(output_dir, f"task_{task_id}.json")
            with open(traj_path, "w") as f:
                f.write(trajectory.model_dump_json(indent=2))

            with file_lock:
                with open(os.path.join(output_dir, "results.jsonl"), "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        status = "CORRECT" if result["is_correct"] else "WRONG"
        logger.info(
            f"  Task {task_id}: [{status}] answer='{final_answer[:80]}' "
            f"({len(agent.trajectory_steps)} steps, {trajectory.exit_reason})"
        )
        return result

    except Exception as e:
        logger.error(f"  Task {task_id} FAILED: {e}\n{traceback.format_exc()}")
        return {"task_id": task_id, "error": str(e), "is_correct": False}
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_path: str,
    llm_name: str = "hosted_vllm/zai-org/GLM-4.7",
    llm_base_url: str = None,
    output_dir: str = "logs/browsecomp_comem",
    max_tasks: int = None,
    max_steps: int = 30,
    max_steps_absolute: int = 50,
    max_exec_time: int = 300,
    max_total_time: int = 7200,
    max_concurrent: int = 1,
    temperature: float = 0,
    judge_model: str = "gpt-4.1",
    max_context_tokens: int = 128_000,
    # CoMeM-specific
    memory_model_name: str = "gpt-3.5-turbo",
    memory_model_address: str = None,
    memory_model_temperature: float = 0.0,
    memory_model_max_gen_tokens: int = 1536,
    retain_most_turns: int = 1,
    latency_test: bool = False,
):
    """
    Run BrowseComp-EN evaluation with CoMeM (compressed memory).

    Args:
        data_path:                   Path to standardized_data.jsonl.
        llm_name:                    Agent LLM model id.
        llm_base_url:                Agent LLM API base URL.
        output_dir:                  Output directory.
        max_tasks:                   Cap on tasks (None = all).
        max_steps:                   Soft step limit.
        max_steps_absolute:          Hard step limit.
        max_exec_time:               Timeout per tool call (seconds).
        max_total_time:              Wall-clock limit per task (seconds).
        max_concurrent:              Parallel tasks.
        temperature:                 Agent LLM temperature.
        judge_model:                 Model for answer evaluation.
        max_context_tokens:          Context-window budget.
        memory_model_name:           Memory/summarizer model id.
        memory_model_address:        Memory model API base URL (for vLLM).
        memory_model_temperature:    Memory model temperature.
        memory_model_max_gen_tokens: Max generation tokens for summary.
        retain_most_turns:           Number of recent turns to keep verbatim.
        latency_test:                Defer all file I/O to end; write latency summary.
    """
    if not os.environ.get("SERPER_API_KEY"):
        logger.warning("SERPER_API_KEY not set - google_search will fail")

    tasks = load_browsecomp_tasks(data_path, max_tasks)
    if not tasks:
        logger.error("No tasks loaded.")
        return

    agent_args = CoMeMBrowseCompAgentArgs(
        llm_name=llm_name,
        llm_base_url=llm_base_url,
        max_context_tokens=max_context_tokens,
        memory_model_name=memory_model_name,
        memory_model_address=memory_model_address,
        memory_model_temperature=memory_model_temperature,
        memory_model_max_gen_tokens=memory_model_max_gen_tokens,
        retain_most_turns=retain_most_turns,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Resume support
    results_path = os.path.join(output_dir, "results.jsonl")
    completed_ids: set = set()
    completed_results: List[Dict[str, Any]] = []
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    completed_ids.add(str(r.get("task_id", "")))
                    completed_results.append(r)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} completed, skipping.")

    tasks = [t for t in tasks if str(t.get("task_id", "")) not in completed_ids]
    if not tasks:
        logger.info("All tasks already completed.")

    results: List[Dict[str, Any]] = list(completed_results)
    common_kwargs = dict(
        agent_args=agent_args,
        max_steps=max_steps,
        max_steps_absolute=max_steps_absolute,
        max_exec_time=max_exec_time,
        max_total_time=max_total_time,
        temperature=temperature,
        output_dir=output_dir,
        judge_model=judge_model,
        latency_test=latency_test,
    )

    if latency_test:
        latency_results = []

    start_time = time.perf_counter()
    if max_concurrent > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {
                pool.submit(run_single_task, task, **common_kwargs): task
                for task in tasks
            }
            if latency_test:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            latency_results.append(result)
                    except Exception as e:
                        logger.error(f"Task failed: {e}")
            else:
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
    else:
        for task in tasks:
            result = run_single_task(task, **common_kwargs)
            if latency_test:
                if result is not None:
                    latency_results.append(result)
            else:
                results.append(result)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    if latency_test:
        # Write all results at the end
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "results.jsonl")
        with open(results_path, "a") as f:
            for result in latency_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        # Write latency summary
        n_tasks = len(latency_results)
        latency_file = os.path.join(output_dir, "latency.txt")
        with open(latency_file, "w") as f:
            f.write(f"Total time for {n_tasks} tasks: {elapsed_time:.2f} seconds\n")
            if n_tasks > 0:
                f.write(f"Average time per task: {elapsed_time / n_tasks:.2f} seconds\n")
        logger.info(f"Latency test: {n_tasks} tasks in {elapsed_time:.2f}s "
                     f"({elapsed_time / max(n_tasks, 1):.2f}s/task)")
        results.extend(latency_results)

    # Summary
    correct = sum(1 for r in results if r.get("is_correct", False))
    total = len(results)
    accuracy = correct / total * 100 if total else 0

    summary_lines = [
        "",
        "=" * 60,
        "BrowseComp-EN CoMeM Results",
        "=" * 60,
        f"  Agent LLM   : {llm_name}",
        f"  Memory Model: {memory_model_name}",
        f"  Retain Turns: {retain_most_turns}",
        f"  Tasks       : {total}",
        f"  Correct     : {correct}",
        f"  Accuracy    : {accuracy:.1f}%",
        "=" * 60,
    ]
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "model": llm_name,
                "memory_model": memory_model_name,
                "retain_most_turns": retain_most_turns,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
            },
            f,
            indent=2,
        )

    logger.info(summary_text)


if __name__ == "__main__":
    Fire(main)
