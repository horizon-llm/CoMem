"""
Entry point for COMEM-KSO latency benchmarking.

Supports both single benchmarks and concurrent batch execution
(via ProcessPoolExecutor) to measure latency under load.

Usage:
    # Single benchmark
    python src/r2egym/agenthub/run/edit_benchmark.py run_benchmark \
        --input_token_target 4096 --output_token_target 2048 --max_steps 20

    # Concurrent batch benchmark (like the original runagent_multiple)
    python src/r2egym/agenthub/run/edit_benchmark.py run_benchmark_batch \
        --input_token_target 4096 --output_token_target 2048 \
        --num_instances 32 --max_workers 32

    # Grid over multiple (input, output) configs, each run as a batch
    python src/r2egym/agenthub/run/edit_benchmark.py run_benchmark_grid \
        --input_token_targets "2048,4096,8192" --output_token_targets "2048,4096" \
        --num_instances 32 --max_workers 32
"""

import json
import os
import time
import csv
import threading
import yaml
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fire import Fire

from r2egym.agenthub.agent.comem_agent_kso import AgentArgs
from r2egym.agenthub.agent.comem_benchmark_agent import BenchmarkCOMEMAgent
from r2egym.agenthub.environment.dummy_env import DummyEnv
from r2egym.agenthub.utils.log import get_logger
from r2egym.logging import setup_logging, INFO
from r2egym.agenthub.trajectory import TrajectoryStep_Summ as TrajectoryStep
from r2egym.agenthub.trajectory import Trajectory_Summ as Trajectory
from transformers import AutoTokenizer

logger = get_logger(__name__)
file_lock = threading.Lock()


def run_benchmark(
    instance_id: int = 0,
    input_token_target: int = 4096,
    output_token_target: int = 2048,
    memory_output_token_target: int = -1,
    observation_tokens: int = 200,
    problem_statement_tokens: int = 500,
    max_steps: int = 20,
    max_steps_absolute: int = 100,
    llm_name: str = "hosted_vllm/agentica-org/DeepSWE-Preview",
    memory_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    memory_model_temperature: float = 0.0,
    memory_model_max_gen_tokens: int = 1536,
    retain_most_turns: int = 3,
    use_optimized_schedule: bool = True,
    temperature: float = 0.0,
    max_token_limit: int = 65536,
    max_mem_token_limit: int = 131072,
    max_total_time: int = 50000,
    exp_name: Optional[str] = None,
    nosummary: bool = False,
    use_user_prompt: bool = True,
    prompt_version: int = 5,
) -> Optional[str]:
    """Run a single benchmark instance.

    Args:
        instance_id: Unique ID for this instance (for logging in batch mode).
        input_token_target: Target total input tokens per LLM call.
        output_token_target: Target output tokens per LLM call (via min_tokens/max_tokens).
        observation_tokens: Tokens in each dummy observation.
        problem_statement_tokens: Tokens in the dummy problem statement.
        max_steps: Soft step limit (used for "steps remaining" message).
        max_steps_absolute: Hard step limit (agent always runs this many steps).
        llm_name: Model name for the agent LLM.
        memory_model_name: Model name for the memory/summarization LLM.
        retain_most_turns: Number of recent turns to keep in full context.
        use_optimized_schedule: Enable KSO scheduling.
        temperature: Sampling temperature.
        max_token_limit: Max tokens for agent model context.
        max_mem_token_limit: Max tokens for memory model context.
        exp_name: Experiment name for logging.
        nosummary: If True, skip memory compression.
        use_user_prompt: Use user prompt templates from config.
        prompt_version: Config version to load.

    Returns:
        Trajectory JSON string, or None on failure.
    """
    if exp_name is None:
        exp_name = f"bench_in{input_token_target}_out{output_token_target}"

    bench_logger = setup_logging(
        name=f"{exp_name}_inst{instance_id}",
        log_file=f"run_logs/{exp_name}/instance_{instance_id}.log",
        console=True,
        level=INFO,
    )
    bench_logger.info(
        f"[Instance {instance_id}] Starting benchmark: input={input_token_target}, "
        f"output={output_token_target}, steps={max_steps_absolute}, model={llm_name}"
    )

    # Load tokenizer for dummy text generation
    model_name = "/".join(llm_name.split("/")[1:])
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    # Create dummy environment
    env = DummyEnv(
        observation_tokens=observation_tokens,
        problem_statement_tokens=problem_statement_tokens,
        tokenizer=tokenizer,
    )

    # Load agent config
    if use_user_prompt:
        config_path = Path(
            f"./src/r2egym/agenthub/config/comem/r2egym/edit_fn_calling_v{prompt_version}.yaml"
        )
    else:
        config_path = Path(
            "./src/r2egym/agenthub/config/r2egym/edit_non_fn_calling.yaml"
        )

    # Also load the full-context config to align system/instance prompts,
    # so that both benchmarks start with the same base prompt size.
    fc_config_path = Path(
        "./src/r2egym/agenthub/config/r2egym/edit_non_fn_calling.yaml"
    )

    agent_args = AgentArgs.from_yaml(config_path)
    # Override system/instance prompts with the full-context config so both
    # benchmarks have identical base prompt sizes for fair comparison.
    with open(fc_config_path, "r") as _f:
        _fc_cfg = yaml.safe_load(_f)
    agent_args.system_prompt = _fc_cfg["system_prompt"]
    agent_args.instance_prompt = _fc_cfg["instance_prompt"]

    agent_args.llm_name = llm_name
    agent_args.memory_model_name = memory_model_name
    agent_args.memory_model_temperature = memory_model_temperature
    agent_args.memory_model_max_gen_tokens = memory_model_max_gen_tokens
    agent_args.retain_most_turns = retain_most_turns
    agent_args.use_optimized_schedule = use_optimized_schedule
    agent_args.use_user_prompt = use_user_prompt
    agent_args.nosummary = nosummary

    # Create benchmark agent
    # memory_output_token_target=-1 means no forced memory output length (use default)
    mem_out_target = memory_output_token_target if memory_output_token_target > 0 else None
    agent = BenchmarkCOMEMAgent(
        name=f"BenchmarkAgent_{instance_id}",
        args=agent_args,
        input_token_target=input_token_target,
        output_token_target=output_token_target,
        memory_output_token_target=mem_out_target,
        logger=bench_logger,
    )

    # Run the agent (uses the real COMEMAgent.run() loop)
    try:
        trajectory = agent.run(
            env,
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            temperature=temperature,
            use_fn_calling=False,
            scaffold="r2egym",
            max_token_limit=max_token_limit,
            max_mem_token_limit=max_mem_token_limit,
            max_total_time=max_total_time,
        )
    except Exception as e:
        bench_logger.error(f"[Instance {instance_id}] Benchmark failed: {e}")
        import traceback
        bench_logger.error(traceback.format_exc())
        return None

    bench_logger.info(
        f"[Instance {instance_id}] Completed: {trajectory.exit_reason}, "
        f"steps={len(trajectory.trajectory_steps)}, "
        f"total_time={trajectory.traj_finish_time - trajectory.traj_start_time:.2f}s"
    )

    return trajectory.model_dump_json()


def run_benchmark_batch(
    input_token_target: int = 4096,
    output_token_target: int = 2048,
    memory_output_token_target: int = -1,
    observation_tokens: int = 200,
    problem_statement_tokens: int = 500,
    num_instances: int = 32,
    max_workers: Optional[int] = None,
    max_steps: int = 20,
    max_steps_absolute: int = 100,
    llm_name: str = "hosted_vllm/agentica-org/DeepSWE-Preview",
    memory_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    memory_model_temperature: float = 0.0,
    memory_model_max_gen_tokens: int = 1536,
    retain_most_turns: int = 3,
    use_optimized_schedule: bool = True,
    temperature: float = 0.0,
    max_token_limit: int = 65536,
    max_mem_token_limit: int = 131072,
    max_total_time: int = 50000,
    traj_dir: str = "./traj_benchmark",
    exp_name: Optional[str] = None,
    nosummary: bool = False,
    use_user_prompt: bool = True,
    prompt_version: int = 5,
    latency_test: bool = True,
):
    """Run multiple benchmark instances concurrently using ProcessPoolExecutor.

    This mirrors the original runagent_multiple pattern: submit num_instances
    concurrent benchmark runs to measure batch latency under load.

    Args:
        num_instances: Number of concurrent benchmark instances to run.
        max_workers: Maximum number of processes. Defaults to num_instances.
        latency_test: If True, collect all results before writing and produce
                      a latency summary file.
        Other args: Same as run_benchmark.
    """
    if max_workers is None:
        max_workers = num_instances

    if exp_name is None:
        exp_name = (
            f"bench_in{input_token_target}_out{output_token_target}"
            f"_n{num_instances}_w{max_workers}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"

    logger.warning(
        f"Starting batch benchmark: {num_instances} instances, {max_workers} workers, "
        f"input={input_token_target}, output={output_token_target}, "
        f"steps={max_steps_absolute}, model={llm_name}"
    )

    start_time = time.perf_counter()
    if latency_test:
        results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(
                run_benchmark,
                instance_id=i,
                input_token_target=input_token_target,
                output_token_target=output_token_target,
                memory_output_token_target=memory_output_token_target,
                observation_tokens=observation_tokens,
                problem_statement_tokens=problem_statement_tokens,
                max_steps=max_steps,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                memory_model_name=memory_model_name,
                memory_model_temperature=memory_model_temperature,
                memory_model_max_gen_tokens=memory_model_max_gen_tokens,
                retain_most_turns=retain_most_turns,
                use_optimized_schedule=use_optimized_schedule,
                temperature=temperature,
                max_token_limit=max_token_limit,
                max_mem_token_limit=max_mem_token_limit,
                max_total_time=max_total_time,
                exp_name=exp_name,
                nosummary=nosummary,
                use_user_prompt=use_user_prompt,
                prompt_version=prompt_version,
            ): i
            for i in range(num_instances)
        }

        if latency_test:
            for future in concurrent.futures.as_completed(future_to_id):
                inst_id = future_to_id[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Exception for instance {inst_id}: {e}")
        else:
            with open(jsonl_file, "a") as f:
                for future in concurrent.futures.as_completed(future_to_id):
                    inst_id = future_to_id[future]
                    try:
                        result = future.result()
                        if result is not None:
                            with file_lock:
                                f.write(result + "\n")
                    except Exception as e:
                        logger.error(f"Exception for instance {inst_id}: {e}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    if latency_test:
        # Save all results to jsonl
        with open(jsonl_file, "w") as f:
            for result in results:
                f.write(result + "\n")

        # Write latency summary
        latency_file = str(jsonl_file).replace(".jsonl", "_latency.txt")
        with open(latency_file, "w") as f:
            f.write(f"Benchmark Configuration:\n")
            f.write(f"  input_token_target: {input_token_target}\n")
            f.write(f"  output_token_target: {output_token_target}\n")
            f.write(f"  num_instances: {num_instances}\n")
            f.write(f"  max_workers: {max_workers}\n")
            f.write(f"  max_steps_absolute: {max_steps_absolute}\n")
            f.write(f"  model: {llm_name}\n")
            f.write(f"  memory_model: {memory_model_name}\n")
            f.write(f"\n")
            f.write(f"Total wall time for {len(results)} instances: {elapsed_time:.2f} seconds\n")
            if len(results) > 0:
                f.write(f"Average wall time per instance: {elapsed_time / len(results):.2f} seconds\n")

                # Aggregate per-step metrics across all instances
                all_steps = []
                for r in results:
                    traj = json.loads(r)
                    all_steps.extend(traj["trajectory_steps"])

                if all_steps:
                    n = len(all_steps)
                    avg_llm = sum(s["llm_exec_time"] for s in all_steps) / n
                    avg_summ = sum(s["summ_exec_time"] for s in all_steps) / n
                    avg_step = sum(s["total_step_time"] for s in all_steps) / n
                    avg_prompt = sum(s["token_usage_prompt"] for s in all_steps) / n
                    avg_completion = sum(s["token_usage_completion"] for s in all_steps) / n

                    f.write(f"\nPer-step averages (across {n} total steps from {len(results)} instances):\n")
                    f.write(f"  avg_llm_exec_time: {avg_llm:.3f}s\n")
                    f.write(f"  avg_summ_exec_time: {avg_summ:.3f}s\n")
                    f.write(f"  avg_total_step_time: {avg_step:.3f}s\n")
                    f.write(f"  avg_prompt_tokens: {avg_prompt:.1f}\n")
                    f.write(f"  avg_completion_tokens: {avg_completion:.1f}\n")

        logger.warning(f"Latency summary saved to {latency_file}")

    logger.warning(
        f"Batch benchmark completed: {len(results) if latency_test else 'N/A'} instances "
        f"in {elapsed_time:.2f}s"
    )

    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"Config: input={input_token_target}, output={output_token_target}")
    print(f"Instances: {num_instances}, Workers: {max_workers}")
    print(f"Total wall time: {elapsed_time:.2f}s")
    if latency_test and len(results) > 0:
        print(f"Successful: {len(results)}/{num_instances}")
        print(f"Avg wall time per instance: {elapsed_time / len(results):.2f}s")
    print(f"Trajectories saved to: {jsonl_file}")
    print(f"{'='*80}")


def run_benchmark_grid(
    input_token_targets: str = "2048,4096,8192,16384",
    output_token_targets: str = "2048,4096,8192",
    memory_output_token_targets: str = "-1",
    max_steps_list: str = "20",
    max_workers_list: str = "32",
    observation_tokens: int = 200,
    problem_statement_tokens: int = 500,
    num_instances: int = 32,
    llm_name: str = "hosted_vllm/agentica-org/DeepSWE-Preview",
    memory_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    memory_model_temperature: float = 0.0,
    memory_model_max_gen_tokens: int = 1536,
    retain_most_turns: int = 3,
    use_optimized_schedule: bool = True,
    temperature: float = 0.0,
    max_token_limit: int = 65536,
    max_mem_token_limit: int = 131072,
    max_total_time: int = 50000,
    traj_dir: str = "./traj_benchmark",
    exp_name: Optional[str] = None,
    nosummary: bool = False,
    use_user_prompt: bool = True,
    prompt_version: int = 5,
    latency_test: bool = True,
):
    """Run batch benchmarks across a grid of configurations.

    Sweeps over all combinations of:
    - input_token_targets (input context size)
    - output_token_targets (output generation size)
    - memory_output_token_targets (memory model generation size; -1 = use default)
    - max_steps_list (number of agent steps per instance)
    - max_workers_list (concurrency level)

    For each combination, runs num_instances concurrent benchmark instances
    (just like runagent_multiple) and collects latency metrics.

    Args:
        input_token_targets: Comma-separated list of input token targets.
        output_token_targets: Comma-separated list of output token targets.
        memory_output_token_targets: Comma-separated list of memory output token targets (-1 = default).
        max_steps_list: Comma-separated list of step counts to test.
        max_workers_list: Comma-separated list of worker counts (concurrency levels) to test.
        num_instances: Number of instances per config.
        Other args: Same as run_benchmark.
    """
    def _parse_int_list(val):
        """Parse a value that may be a string, tuple, or list into a list of ints."""
        if isinstance(val, (list, tuple)):
            return [int(x) for x in val]
        return [int(x.strip()) for x in str(val).split(",")]

    input_targets = _parse_int_list(input_token_targets)
    output_targets = _parse_int_list(output_token_targets)
    mem_out_targets = _parse_int_list(memory_output_token_targets)
    steps_targets = _parse_int_list(max_steps_list)
    workers_targets = _parse_int_list(max_workers_list)

    if exp_name is None:
        exp_name = f"bench_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    grid_traj_dir = os.path.join(traj_dir, exp_name)
    os.makedirs(grid_traj_dir, exist_ok=True)

    summary_results = []
    total_configs = (
        len(input_targets) * len(output_targets) * len(mem_out_targets)
        * len(steps_targets) * len(workers_targets)
    )
    config_idx = 0

    for inp in input_targets:
        for out in output_targets:
            for mem_out in mem_out_targets:
                for steps in steps_targets:
                    for workers in workers_targets:
                        config_idx += 1
                        mem_label = f"m{mem_out}" if mem_out > 0 else "mdef"
                        config_exp = f"in{inp}_out{out}_{mem_label}_s{steps}_w{workers}_n{num_instances}"
                        logger.warning(
                            f"[{config_idx}/{total_configs}] Running batch benchmark: "
                            f"input={inp}, output={out}, mem_out={mem_out}, steps={steps}, "
                            f"workers={workers}, instances={num_instances}"
                        )

                        batch_start = time.perf_counter()
                        run_benchmark_batch(
                            input_token_target=inp,
                            output_token_target=out,
                            memory_output_token_target=mem_out,
                            observation_tokens=observation_tokens,
                            problem_statement_tokens=problem_statement_tokens,
                            num_instances=num_instances,
                            max_workers=workers,
                            max_steps=steps,
                            max_steps_absolute=steps,
                            llm_name=llm_name,
                            memory_model_name=memory_model_name,
                            memory_model_temperature=memory_model_temperature,
                            memory_model_max_gen_tokens=memory_model_max_gen_tokens,
                            retain_most_turns=retain_most_turns,
                            use_optimized_schedule=use_optimized_schedule,
                            temperature=temperature,
                            max_token_limit=max_token_limit,
                            max_mem_token_limit=max_mem_token_limit,
                            max_total_time=max_total_time,
                            traj_dir=grid_traj_dir,
                            exp_name=config_exp,
                            nosummary=nosummary,
                            use_user_prompt=use_user_prompt,
                            prompt_version=prompt_version,
                            latency_test=latency_test,
                        )
                        batch_wall_time = time.perf_counter() - batch_start

                        # Read back the results from the jsonl file
                        jsonl_path = os.path.join(grid_traj_dir, f"{config_exp}.jsonl")
                        if os.path.exists(jsonl_path):
                            all_steps = []
                            n_instances = 0
                            with open(jsonl_path) as f:
                                for line in f:
                                    traj = json.loads(line)
                                    all_steps.extend(traj["trajectory_steps"])
                                    n_instances += 1

                            if all_steps:
                                n = len(all_steps)
                                result = {
                                    "input_tokens": inp,
                                    "output_tokens": out,
                                    "mem_output_tokens": mem_out,
                                    "max_steps": steps,
                                    "max_workers": workers,
                                    "instances": n_instances,
                                    "total_steps": n,
                                "avg_llm_time": round(sum(s["llm_exec_time"] for s in all_steps) / n, 3),
                                    "avg_summ_time": round(sum(s["summ_exec_time"] for s in all_steps) / n, 3),
                                    "avg_step_time": round(sum(s["total_step_time"] for s in all_steps) / n, 3),
                                    "avg_prompt_tokens": round(sum(s["token_usage_prompt"] for s in all_steps) / n, 1),
                                    "avg_completion_tokens": round(sum(s["token_usage_completion"] for s in all_steps) / n, 1),
                                    "batch_wall_time": round(batch_wall_time, 2),
                                }
                                summary_results.append(result)
                                logger.warning(
                                    f"  -> instances={n_instances}, steps={n}, "
                                    f"avg_llm={result['avg_llm_time']:.3f}s, "
                                    f"avg_summ={result['avg_summ_time']:.3f}s, "
                                    f"batch_wall={batch_wall_time:.2f}s"
                                )
                        else:
                            logger.error(f"  -> No results file found at {jsonl_path}")

    # Write grid summary CSV
    if summary_results:
        csv_file = os.path.join(grid_traj_dir, "grid_latency_summary.csv")
        fieldnames = list(summary_results[0].keys())
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_results)
        logger.warning(f"Grid summary saved to {csv_file}")

    # Print grid summary table
    print(f"\n{'='*155}")
    print("GRID BENCHMARK SUMMARY")
    print(f"{'='*155}")
    print(
        f"{'Input':>8} {'Output':>8} {'MemOut':>8} {'Steps':>7} {'Workers':>8} {'Inst':>6} {'TotSteps':>9} "
        f"{'AvgLLM(s)':>10} {'AvgSumm(s)':>11} {'AvgStep(s)':>11} "
        f"{'AvgPrompt':>10} {'AvgCompl':>9} {'BatchWall(s)':>13}"
    )
    print("-" * 155)
    for r in summary_results:
        print(
            f"{r['input_tokens']:>8} {r['output_tokens']:>8} "
            f"{r['mem_output_tokens']:>8} "
            f"{r['max_steps']:>7} {r['max_workers']:>8} "
            f"{r['instances']:>6} {r['total_steps']:>9} "
            f"{r['avg_llm_time']:>10.3f} {r['avg_summ_time']:>11.3f} "
            f"{r['avg_step_time']:>11.3f} {r['avg_prompt_tokens']:>10.1f} "
            f"{r['avg_completion_tokens']:>9.1f} {r['batch_wall_time']:>13.2f}"
        )
    print(f"{'='*155}")


if __name__ == "__main__":
    Fire({
        "run_benchmark": run_benchmark,
        "run_benchmark_batch": run_benchmark_batch,
        "run_benchmark_grid": run_benchmark_grid,
    })
