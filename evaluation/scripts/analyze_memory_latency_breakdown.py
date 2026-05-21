#!/usr/bin/env python3
"""
Detailed latency breakdown of the memory model in COMEM trajectories.

Analyzes real trajectory data to answer:
1. What is the incremental cost of the memory model per step?
2. How much memory time is hidden behind agent inference (overlap)?
3. What is the effective added latency (wall-clock overhead)?
4. How does memory latency scale with context length (proxy for queuing)?

Usage:
    python scripts/analyze_memory_latency_breakdown.py \
        --traj_jsonl ./traj/experiment.jsonl \
        --retain_most_turns 4 \
        --output_json ./analysis_results/latency_breakdown.json

    # Also include vLLM bench results for TTFT-based queuing analysis:
    python scripts/analyze_memory_latency_breakdown.py \
        --traj_jsonl ./traj/experiment.jsonl \
        --retain_most_turns 4 \
        --vllm_bench_dir ./benchmarks-qwen3-4b-h200/
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectories(jsonl_path: str) -> List[Dict[str, Any]]:
    trajs = []
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                trajs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed JSON at line {line_num}: {e}")
    return trajs


def _stats(arr: np.ndarray) -> Dict[str, float]:
    if len(arr) == 0:
        return {"mean": 0, "std": 0, "median": 0, "p25": 0, "p75": 0, "min": 0, "max": 0, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


# ---------------------------------------------------------------------------
# 1. Per-step latency breakdown (from trajectory timestamps)
# ---------------------------------------------------------------------------

def compute_latency_breakdown(
    all_trajs: List[Dict[str, Any]],
    retain_most_turns: int = 3,
) -> Dict[str, Any]:
    """Compute detailed latency breakdown from trajectory timestamps.

    For each step, we have:
    - summary_start_time, summary_finish_time (memory model, perf_counter)
    - agent_start_time, agent_finish_time (agent model, perf_counter)
    - summ_exec_time, llm_exec_time, env_exec_time, prefill_exec_time
    - total_step_time = max(llm_exec_time, summ_exec_time + prefill_exec_time) + env_exec_time

    Since memory compression runs async overlapped with agent inference,
    the overlap and effective overhead can be computed from timestamps.
    """
    # Collect per-step records
    all_steps = []        # all steps
    compression_steps = []  # steps where memory model was called
    non_compression_steps = []  # steps without memory call

    for traj in all_trajs:
        for step in traj.get("trajectory_steps", []):
            record = {
                "step_idx": step.get("step_idx", 0),
                "llm_exec_time": step.get("llm_exec_time", 0.0),
                "summ_exec_time": step.get("summ_exec_time", 0.0),
                "prefill_exec_time": step.get("prefill_exec_time") or 0.0,
                "env_exec_time": step.get("env_exec_time", 0.0),
                "total_step_time": step.get("total_step_time", 0.0),
                "summary_start_time": step.get("summary_start_time"),
                "summary_finish_time": step.get("summary_finish_time"),
                "agent_start_time": step.get("agent_start_time"),
                "agent_finish_time": step.get("agent_finish_time"),
                "token_usage_summary_prompt": step.get("token_usage_summary_prompt", 0),
                "token_usage_summary": step.get("token_usage_summary", 0),
                "token_usage_prompt": step.get("token_usage_prompt", 0),
                "token_usage_completion": step.get("token_usage_completion", 0),
            }
            all_steps.append(record)

            if record["summ_exec_time"] > 0:
                compression_steps.append(record)
            else:
                non_compression_steps.append(record)

    if not all_steps:
        return {"error": "No steps found in trajectories"}

    # --- A. Basic time breakdown (all steps) ---
    llm_times = np.array([s["llm_exec_time"] for s in all_steps])
    summ_times = np.array([s["summ_exec_time"] for s in all_steps])
    prefill_times = np.array([s["prefill_exec_time"] for s in all_steps])
    env_times = np.array([s["env_exec_time"] for s in all_steps])
    total_times = np.array([s["total_step_time"] for s in all_steps])

    # --- B. Overlap analysis (compression steps only) ---
    # The memory task runs concurrently with agent inference.
    # Overlap = time where both are running simultaneously.
    # From timestamps: overlap = max(0, min(agent_end, summ_end) - max(agent_start, summ_start))
    overlaps = []
    effective_overheads = []
    overlap_fractions = []  # fraction of memory time hidden behind agent
    hidden_fractions = []   # fraction of memory time that is "free" (hidden)

    for s in compression_steps:
        summ_start = s["summary_start_time"]
        summ_finish = s["summary_finish_time"]
        agent_start = s["agent_start_time"]
        agent_finish = s["agent_finish_time"]

        if all(t is not None for t in [summ_start, summ_finish, agent_start, agent_finish]):
            # Overlap: time where both memory and agent are running
            overlap = max(0, min(agent_finish, summ_finish) - max(agent_start, summ_start))
            overlaps.append(overlap)

            summ_duration = summ_finish - summ_start
            agent_duration = agent_finish - agent_start

            # Effective overhead: memory time NOT hidden behind agent
            # = max(0, summ finishes after agent)
            effective_overhead = max(0, summ_finish - agent_finish)
            effective_overheads.append(effective_overhead)

            # Fraction of memory time that overlaps with agent (= "free")
            if summ_duration > 0:
                hidden_frac = overlap / summ_duration
                hidden_fractions.append(hidden_frac)

            # Fraction of agent time spent overlapping with memory
            if agent_duration > 0:
                overlap_fractions.append(overlap / agent_duration)

    overlaps = np.array(overlaps) if overlaps else np.array([])
    effective_overheads = np.array(effective_overheads) if effective_overheads else np.array([])
    hidden_fractions = np.array(hidden_fractions) if hidden_fractions else np.array([])

    # --- C. Effective overhead as fraction of step time ---
    # total_step_time = max(llm, summ + prefill) + env
    # baseline (no memory) = llm + env
    # overhead = total_step_time - (llm + env)
    step_overheads = []
    step_overhead_fracs = []
    for s in all_steps:
        baseline = s["llm_exec_time"] + s["env_exec_time"]
        overhead = s["total_step_time"] - baseline
        step_overheads.append(max(0, overhead))
        if baseline > 0:
            step_overhead_fracs.append(max(0, overhead) / baseline)
    step_overheads = np.array(step_overheads)
    step_overhead_fracs = np.array(step_overhead_fracs)

    # --- D. Memory latency vs context length (proxy for queuing under load) ---
    # As context grows, memory input tokens increase. If there's queuing,
    # we'd see summ_exec_time increase faster than token count.
    mem_input_tokens = np.array([s["token_usage_summary_prompt"] for s in compression_steps])
    mem_output_tokens = np.array([s["token_usage_summary"] for s in compression_steps])
    summ_times_comp = np.array([s["summ_exec_time"] for s in compression_steps])

    # Tokens per second for memory model (higher = no queuing, lower = possible queuing)
    mem_tok_per_sec = (mem_input_tokens + mem_output_tokens) / np.maximum(summ_times_comp, 1e-6)

    # --- E. Per-step-index analysis (how metrics evolve over trajectory) ---
    step_idx_data = {}
    for s in all_steps:
        idx = s["step_idx"]
        if idx not in step_idx_data:
            step_idx_data[idx] = {
                "llm_exec_time": [], "summ_exec_time": [], "env_exec_time": [],
                "total_step_time": [], "effective_overhead": [],
            }
        step_idx_data[idx]["llm_exec_time"].append(s["llm_exec_time"])
        step_idx_data[idx]["summ_exec_time"].append(s["summ_exec_time"])
        step_idx_data[idx]["env_exec_time"].append(s["env_exec_time"])
        step_idx_data[idx]["total_step_time"].append(s["total_step_time"])
        baseline = s["llm_exec_time"] + s["env_exec_time"]
        step_idx_data[idx]["effective_overhead"].append(max(0, s["total_step_time"] - baseline))

    per_step_idx = {}
    for idx in sorted(step_idx_data.keys()):
        per_step_idx[idx] = {}
        for metric, values in step_idx_data[idx].items():
            arr = np.array(values)
            per_step_idx[idx][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n": len(arr),
            }

    return {
        "num_trajectories": len(all_trajs),
        "num_total_steps": len(all_steps),
        "num_compression_steps": len(compression_steps),
        "num_non_compression_steps": len(non_compression_steps),
        "retain_most_turns": retain_most_turns,

        # A. Time breakdown (all steps)
        "time_breakdown_all_steps": {
            "llm_exec_time_sec": _stats(llm_times),
            "summ_exec_time_sec": _stats(summ_times),
            "prefill_exec_time_sec": _stats(prefill_times),
            "env_exec_time_sec": _stats(env_times),
            "total_step_time_sec": _stats(total_times),
        },

        # B. Overlap analysis (compression steps with timestamps)
        "overlap_analysis": {
            "overlap_sec": _stats(overlaps),
            "effective_overhead_sec": _stats(effective_overheads),
            "hidden_fraction": _stats(hidden_fractions),
            "num_steps_with_timestamps": len(overlaps),
        },

        # C. Effective overhead
        "effective_overhead": {
            "per_step_overhead_sec": _stats(step_overheads),
            "per_step_overhead_fraction": _stats(step_overhead_fracs),
        },

        # D. Memory throughput consistency (proxy for queuing)
        "memory_throughput_consistency": {
            "mem_tok_per_sec": _stats(mem_tok_per_sec),
            "mem_input_tokens": _stats(mem_input_tokens),
            "summ_exec_time_sec": _stats(summ_times_comp),
        },

        # E. Per-step-index evolution
        "per_step_idx": per_step_idx,
    }


# ---------------------------------------------------------------------------
# 2. vLLM TTFT-based queuing analysis (Option A)
# ---------------------------------------------------------------------------

def parse_vllm_bench_output(filepath: str) -> Optional[Dict[str, Any]]:
    text = Path(filepath).read_text()
    result = {"file": filepath}

    fname = Path(filepath).stem
    m = re.match(r"vllm_bench_batch(\d+)_inp(\d+)_outp(\d+)_repeat(\d+)", fname)
    if m:
        result["batch_size"] = int(m.group(1))
        result["input_length"] = int(m.group(2))
        result["output_length"] = int(m.group(3))
        result["repeat"] = int(m.group(4))

    for key, pattern in {
        "request_throughput_req_per_sec": r"Request throughput \(req/s\):\s*([\d.]+)",
        "output_throughput_tok_per_sec": r"Output token throughput \(tok/s\):\s*([\d.]+)",
        "total_throughput_tok_per_sec": r"Total token throughput \(tok/s\):\s*([\d.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s*([\d.]+)",
        "median_ttft_ms": r"Median TTFT \(ms\):\s*([\d.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s*([\d.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s*([\d.]+)",
        "median_tpot_ms": r"Median TPOT \(ms\):\s*([\d.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s*([\d.]+)",
        "mean_itl_ms": r"Mean ITL \(ms\):\s*([\d.]+)",
        "median_itl_ms": r"Median ITL \(ms\):\s*([\d.]+)",
        "p99_itl_ms": r"P99 ITL \(ms\):\s*([\d.]+)",
    }.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[key] = float(match.group(1))

    return result if len(result) > 1 else None


def compute_ttft_queuing_analysis(bench_dir: str) -> Dict[str, Any]:
    """Analyze TTFT across concurrency levels to isolate queuing overhead.

    At concurrency=1, TTFT ≈ pure prefill compute (no queuing).
    At higher concurrency, TTFT increase = queuing overhead.
    """
    benchmarks = []
    for f in sorted(Path(bench_dir).glob("*.out")):
        parsed = parse_vllm_bench_output(str(f))
        if parsed and "median_ttft_ms" in parsed:
            benchmarks.append(parsed)

    if not benchmarks:
        return {"error": "No benchmark files with TTFT data found"}

    # Group by batch_size, average across repeats
    by_batch = {}
    for b in benchmarks:
        bs = b.get("batch_size", 0)
        if bs not in by_batch:
            by_batch[bs] = []
        by_batch[bs].append(b)

    # Compute per-batch averages
    rows = []
    for bs in sorted(by_batch.keys()):
        runs = by_batch[bs]
        row = {
            "batch_size": bs,
            "num_repeats": len(runs),
            "mean_ttft_ms": np.mean([r["mean_ttft_ms"] for r in runs if "mean_ttft_ms" in r]),
            "median_ttft_ms": np.mean([r["median_ttft_ms"] for r in runs if "median_ttft_ms" in r]),
            "p99_ttft_ms": np.mean([r["p99_ttft_ms"] for r in runs if "p99_ttft_ms" in r]),
            "mean_tpot_ms": np.mean([r["mean_tpot_ms"] for r in runs if "mean_tpot_ms" in r]),
            "median_tpot_ms": np.mean([r["median_tpot_ms"] for r in runs if "median_tpot_ms" in r]),
            "p99_tpot_ms": np.mean([r["p99_tpot_ms"] for r in runs if "p99_tpot_ms" in r]),
            "total_throughput_tok_per_sec": np.mean(
                [r["total_throughput_tok_per_sec"] for r in runs if "total_throughput_tok_per_sec" in r]
            ),
        }
        rows.append(row)

    # Baseline TTFT at concurrency=1 (pure compute, no queuing)
    baseline_ttft = None
    for r in rows:
        if r["batch_size"] == 1:
            baseline_ttft = r["median_ttft_ms"]
            break

    # Add queuing overhead estimate
    for r in rows:
        if baseline_ttft is not None:
            r["estimated_queue_time_ms"] = max(0, r["median_ttft_ms"] - baseline_ttft)
            r["queue_fraction"] = r["estimated_queue_time_ms"] / r["median_ttft_ms"] if r["median_ttft_ms"] > 0 else 0
        else:
            r["estimated_queue_time_ms"] = None
            r["queue_fraction"] = None

    return {
        "baseline_ttft_ms": baseline_ttft,
        "concurrency_sweep": rows,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_stats(label: str, stats: Dict[str, float], unit: str = ""):
    u = f" {unit}" if unit else ""
    print(f"  {label}:")
    print(f"    mean={stats['mean']:.2f}{u}  median={stats['median']:.2f}{u}  "
          f"std={stats['std']:.2f}{u}")
    print(f"    [p25={stats['p25']:.2f}, p75={stats['p75']:.2f}, "
          f"min={stats['min']:.2f}, max={stats['max']:.2f}]  (n={stats['n']})")


def print_results(
    breakdown: Dict[str, Any],
    ttft_analysis: Optional[Dict[str, Any]],
):
    print_section("MEMORY MODEL LATENCY BREAKDOWN")

    if "error" in breakdown:
        print(f"\n  ERROR: {breakdown['error']}")
        return

    print(f"\n  Trajectories: {breakdown['num_trajectories']}")
    print(f"  Total steps: {breakdown['num_total_steps']}")
    print(f"  Compression steps: {breakdown['num_compression_steps']}  "
          f"({breakdown['num_compression_steps']/breakdown['num_total_steps']:.1%} of steps)")
    print(f"  retain_most_turns: {breakdown['retain_most_turns']}")

    # A. Time breakdown
    tb = breakdown["time_breakdown_all_steps"]
    print_section("1. Per-Step Time Breakdown (all steps)")
    print_stats("Agent LLM time", tb["llm_exec_time_sec"], "s")
    print_stats("Memory summarization time", tb["summ_exec_time_sec"], "s")
    print_stats("KSO prefill time", tb["prefill_exec_time_sec"], "s")
    print_stats("Environment exec time", tb["env_exec_time_sec"], "s")
    print_stats("Total step time", tb["total_step_time_sec"], "s")

    mean_llm = tb["llm_exec_time_sec"]["mean"]
    mean_summ = tb["summ_exec_time_sec"]["mean"]
    mean_env = tb["env_exec_time_sec"]["mean"]
    mean_total = tb["total_step_time_sec"]["mean"]
    print(f"\n  Time composition (means):")
    print(f"    Agent LLM:  {mean_llm:.2f}s  ({mean_llm/mean_total:.1%})")
    print(f"    Memory:     {mean_summ:.2f}s  ({mean_summ/mean_total:.1%})  [overlapped with agent]")
    print(f"    Env exec:   {mean_env:.2f}s  ({mean_env/mean_total:.1%})")
    print(f"    Total:      {mean_total:.2f}s")

    # B. Overlap analysis
    oa = breakdown["overlap_analysis"]
    if oa["num_steps_with_timestamps"] > 0:
        print_section("2. Overlap Analysis (compression steps with timestamps)")
        print(f"  Steps with timestamp data: {oa['num_steps_with_timestamps']}")
        print_stats("Overlap duration (agent & memory concurrent)", oa["overlap_sec"], "s")
        print_stats("Effective overhead (memory exceeds agent by)", oa["effective_overhead_sec"], "s")
        print_stats("Hidden fraction (memory time hidden behind agent)", oa["hidden_fraction"])

        mean_hidden = oa["hidden_fraction"]["mean"]
        mean_overhead = oa["effective_overhead_sec"]["mean"]
        print(f"\n  Summary:")
        print(f"    {mean_hidden:.1%} of memory computation is hidden behind agent inference")
        print(f"    Effective added wall-clock time per compression step: {mean_overhead:.3f}s")
    else:
        print_section("2. Overlap Analysis")
        print("  No timestamp data available (summary_start_time/agent_start_time are null)")

    # C. Effective overhead
    eo = breakdown["effective_overhead"]
    print_section("3. Effective Overhead (all steps)")
    print_stats("Per-step overhead", eo["per_step_overhead_sec"], "s")
    print_stats("Per-step overhead fraction (overhead/baseline)", eo["per_step_overhead_fraction"])
    mean_frac = eo["per_step_overhead_fraction"]["mean"]
    print(f"\n  Average overhead: {mean_frac:.2%} of baseline (agent + env) step time")

    # D. Memory throughput consistency
    mc = breakdown["memory_throughput_consistency"]
    print_section("4. Memory Model Throughput Consistency")
    print_stats("Memory throughput (tok/s per request)", mc["mem_tok_per_sec"], "tok/s")
    print_stats("Memory input tokens", mc["mem_input_tokens"], "tok")
    print_stats("Memory exec time", mc["summ_exec_time_sec"], "s")
    cv = mc["mem_tok_per_sec"]["std"] / mc["mem_tok_per_sec"]["mean"] if mc["mem_tok_per_sec"]["mean"] > 0 else 0
    print(f"\n  Coefficient of variation: {cv:.2f}")
    print(f"  (Low CV = consistent throughput, no significant queuing delays)")

    # E. Per-step evolution
    psi = breakdown["per_step_idx"]
    if psi:
        print_section("5. Latency Evolution Over Trajectory Steps")
        print(f"  {'Step':>5}  {'Agent(s)':>10}  {'Memory(s)':>10}  {'Env(s)':>10}  "
              f"{'Total(s)':>10}  {'Overhead(s)':>12}  {'n':>5}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*5}")
        for idx in sorted(psi.keys()):
            s = psi[idx]
            print(f"  {idx:>5}  "
                  f"{s['llm_exec_time']['mean']:>10.2f}  "
                  f"{s['summ_exec_time']['mean']:>10.2f}  "
                  f"{s['env_exec_time']['mean']:>10.2f}  "
                  f"{s['total_step_time']['mean']:>10.2f}  "
                  f"{s['effective_overhead']['mean']:>12.3f}  "
                  f"{s['llm_exec_time']['n']:>5}")

    # F. TTFT queuing analysis (Option A)
    if ttft_analysis and "error" not in ttft_analysis:
        print_section("6. TTFT-Based Queuing Analysis (vLLM benchmarks)")
        print(f"  Baseline TTFT (concurrency=1): {ttft_analysis['baseline_ttft_ms']:.1f} ms")
        print(f"\n  {'Concurrency':>12}  {'TTFT(ms)':>10}  {'p99 TTFT':>10}  "
              f"{'Queue(ms)':>10}  {'Queue%':>8}  {'TPOT(ms)':>10}  {'Throughput':>12}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*12}")
        for r in ttft_analysis["concurrency_sweep"]:
            queue_ms = r.get("estimated_queue_time_ms")
            queue_pct = r.get("queue_fraction")
            print(f"  {r['batch_size']:>12}  "
                  f"{r['median_ttft_ms']:>10.1f}  "
                  f"{r['p99_ttft_ms']:>10.1f}  "
                  f"{queue_ms:>10.1f}  " if queue_ms is not None else f"{'N/A':>10}  "
                  f"{queue_pct:>7.1%}  " if queue_pct is not None else f"{'N/A':>8}  "
                  f"{r['median_tpot_ms']:>10.2f}  "
                  f"{r['total_throughput_tok_per_sec']:>10.1f} tok/s")
        print(f"\n  Interpretation:")
        print(f"    TTFT increase from concurrency=1 to higher = queuing overhead")
        print(f"    TPOT stays stable = decode throughput unaffected by queuing")

    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detailed latency breakdown of COMEM memory model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--traj_jsonl", type=str, nargs="*", required=True,
                        help="Path(s) to COMEM trajectory .jsonl file(s)")
    parser.add_argument("--retain_most_turns", type=int, default=3)
    parser.add_argument("--vllm_bench_dir", type=str, default=None,
                        help="Directory with vllm bench .out files for TTFT analysis")
    parser.add_argument("--output_json", type=str, default=None)

    args = parser.parse_args()

    # Load trajectories
    all_trajs = []
    for f in args.traj_jsonl:
        print(f"Loading {f}...")
        all_trajs.extend(load_trajectories(f))
    print(f"Loaded {len(all_trajs)} trajectories.")

    # Option B: Trajectory-based breakdown
    breakdown = compute_latency_breakdown(all_trajs, args.retain_most_turns)

    # Option A: TTFT-based queuing analysis
    ttft_analysis = None
    if args.vllm_bench_dir:
        ttft_analysis = compute_ttft_queuing_analysis(args.vllm_bench_dir)

    # Print
    print_results(breakdown, ttft_analysis)

    # Save
    if args.output_json:
        output = {"latency_breakdown": breakdown, "ttft_queuing_analysis": ttft_analysis}
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
