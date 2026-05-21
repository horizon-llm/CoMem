#!/usr/bin/env python3
"""
Analyze memory model throughput requirements for COMEM system.

Computes:
1. **Required throughput** — how much memory model capacity each concurrent agent
   needs, derived from real COMEM trajectory data.
2. **Maximum throughput** — the peak throughput of the memory model server,
   parsed from `vllm bench serve` output files.
3. **Agent-to-server ratio** — how many concurrent agent instances a single
   memory model server can support.

Usage:
    # From a single trajectory JSONL file (one JSON line per trajectory):
    python scripts/analyze_memory_throughput.py \
        --traj_jsonl ./traj/experiment.jsonl \
        --retain_most_turns 3

    # With vLLM benchmark results for max throughput:
    python scripts/analyze_memory_throughput.py \
        --traj_jsonl ./traj/experiment.jsonl \
        --vllm_bench_dir ./benchmarks-qwen3-4b-h100/ \
        --retain_most_turns 3

    # Estimation mode (no trajectory data needed):
    python scripts/analyze_memory_throughput.py \
        --estimate \
        --mem_input_tokens 4000 \
        --mem_output_tokens 1500 \
        --agent_step_time 15.0 \
        --retain_most_turns 3 \
        --max_throughput_tok_per_sec 50000
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. Load trajectory data
# ---------------------------------------------------------------------------

def load_trajectories(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load trajectories from a single .jsonl file (one JSON line per trajectory).

    Each line is a Trajectory_Summ.model_dump_json() — a full trajectory object
    containing a 'trajectory_steps' list with per-step data.
    """
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


# ---------------------------------------------------------------------------
# 2. Extract memory model metrics from trajectories
# ---------------------------------------------------------------------------

def extract_memory_metrics(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract per-step memory model metrics from a single trajectory.

    Returns one record per step where memory compression was actually invoked
    (i.e., summ_exec_time > 0).
    """
    records = []
    for step in traj.get("trajectory_steps", []):
        summ_exec_time = step.get("summ_exec_time", 0.0)
        if summ_exec_time > 0:
            records.append({
                "step_idx": step.get("step_idx", step.get("step_count", 0)),
                "mem_input_tokens": step.get("token_usage_summary_prompt", 0),
                "mem_output_tokens": step.get("token_usage_summary", 0),
                "summ_exec_time": summ_exec_time,
                "llm_exec_time": step.get("llm_exec_time", 0.0),
                "prefill_exec_time": step.get("prefill_exec_time") or 0.0,
                "total_step_time": step.get("total_step_time", 0.0),
                "env_exec_time": step.get("env_exec_time", 0.0),
                "agent_input_tokens": step.get("token_usage_prompt", 0),
                "agent_output_tokens": step.get("token_usage_completion", 0),
            })
    return records


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


def compute_required_throughput(
    all_trajs: List[Dict[str, Any]],
    retain_most_turns: int = 3,
) -> Dict[str, Any]:
    """Compute required memory model throughput from trajectory data.

    Two demand metrics:
    - Sustained demand: average load accounting for 1-in-K call frequency
      = mean(mem_total_tokens) / (K * mean_step_time)
    - Burst demand: peak load during a compression step
      = mem_total_tokens / llm_exec_time  (must finish within agent inference time)
    """
    all_records = []
    for traj in all_trajs:
        all_records.extend(extract_memory_metrics(traj))

    if not all_records:
        return {"error": "No memory compression steps found in trajectories"}

    mem_input = np.array([r["mem_input_tokens"] for r in all_records])
    mem_output = np.array([r["mem_output_tokens"] for r in all_records])
    mem_total = mem_input + mem_output
    summ_time = np.array([r["summ_exec_time"] for r in all_records])
    llm_time = np.array([r["llm_exec_time"] for r in all_records])
    agent_input = np.array([r["agent_input_tokens"] for r in all_records])
    agent_output = np.array([r["agent_output_tokens"] for r in all_records])

    # Observed memory model throughput per request
    mem_throughput_per_req = mem_total / np.maximum(summ_time, 1e-6)
    mem_output_throughput = mem_output / np.maximum(summ_time, 1e-6)

    # Bottleneck analysis: how often does summary take longer than agent inference?
    is_bottleneck = summ_time > llm_time
    bottleneck_frac = np.mean(is_bottleneck)

    # Burst demand: during a compression step, must process all tokens within llm_exec_time
    burst_demand = mem_total / np.maximum(llm_time, 1e-6)

    # Mean total step time across ALL steps (not just compression steps)
    all_step_times = []
    for traj in all_trajs:
        for step in traj.get("trajectory_steps", []):
            all_step_times.append(step.get("total_step_time", 0.0))
    mean_step_time = np.mean(all_step_times) if all_step_times else 0.0

    # Sustained demand: request_rate * tokens_per_request
    req_rate_per_agent = 1.0 / (retain_most_turns * mean_step_time) if mean_step_time > 0 else 0.0
    sustained_demand_per_agent = req_rate_per_agent * float(np.mean(mem_total))

    return {
        "num_trajectories": len(all_trajs),
        "num_compression_steps": len(all_records),
        "num_total_steps": len(all_step_times),
        "retain_most_turns": retain_most_turns,
        "mem_input_tokens": _stats(mem_input),
        "mem_output_tokens": _stats(mem_output),
        "mem_total_tokens": _stats(mem_total),
        "summ_exec_time_sec": _stats(summ_time),
        "llm_exec_time_sec": _stats(llm_time),
        "mean_total_step_time_sec": float(mean_step_time),
        "observed_mem_throughput_tok_per_sec": _stats(mem_throughput_per_req),
        "observed_mem_output_throughput_tok_per_sec": _stats(mem_output_throughput),
        "bottleneck_fraction": float(bottleneck_frac),
        "summ_exceeds_llm_by_sec": _stats(np.maximum(summ_time - llm_time, 0)),
        "burst_demand_per_agent_tok_per_sec": _stats(burst_demand),
        "req_rate_per_agent_req_per_sec": float(req_rate_per_agent),
        "sustained_demand_per_agent_tok_per_sec": float(sustained_demand_per_agent),
        "agent_input_tokens": _stats(agent_input),
        "agent_output_tokens": _stats(agent_output),
    }


# ---------------------------------------------------------------------------
# 3. Parse vLLM benchmark output for max throughput
# ---------------------------------------------------------------------------

def parse_vllm_bench_output(filepath: str) -> Optional[Dict[str, Any]]:
    """Parse a vllm bench serve output file."""
    text = Path(filepath).read_text()
    result = {"file": filepath}

    fname = Path(filepath).stem
    m = re.match(r"vllm_bench_batch(\d+)_inp(\d+)_outp(\d+)_repeat(\d+)", fname)
    if m:
        result["batch_size"] = int(m.group(1))
        result["input_length"] = int(m.group(2))
        result["output_length"] = int(m.group(3))
        result["repeat"] = int(m.group(4))

    # Parse key-value lines like "Request throughput (req/s):              0.70"
    # The format is: <label>:<whitespace><number>
    for key, pattern in {
        "request_throughput_req_per_sec": r"Request throughput \(req/s\):\s*([\d.]+)",
        "output_throughput_tok_per_sec": r"Output token throughput \(tok/s\):\s*([\d.]+)",
        "peak_output_throughput_tok_per_sec": r"Peak output token throughput \(tok/s\):\s*([\d.]+)",
        "total_throughput_tok_per_sec": r"Total token throughput \(tok/s\):\s*([\d.]+)",
    }.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[key] = float(match.group(1))

    for key, pattern in {
        "median_ttft_ms": r"Median TTFT \(ms\):\s*([\d.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s*([\d.]+)",
        "median_tpot_ms": r"Median TPOT \(ms\):\s*([\d.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s*([\d.]+)",
    }.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[key] = float(match.group(1))

    return result if len(result) > 1 else None


def load_vllm_benchmarks(bench_dir: str) -> List[Dict[str, Any]]:
    results = []
    for f in sorted(Path(bench_dir).glob("*.out")):
        parsed = parse_vllm_bench_output(str(f))
        if parsed:
            results.append(parsed)
    return results


def find_best_matching_benchmarks(
    benchmarks: List[Dict[str, Any]],
    target_input_len: int,
    target_output_len: int,
) -> List[Dict[str, Any]]:
    """Find benchmark runs closest to target input/output lengths.

    Returns all runs at the best-matching (input_length, output_length),
    sorted by batch_size.
    """
    if not benchmarks:
        return []

    best_dist = float("inf")
    best_pair = None
    for b in benchmarks:
        inp = b.get("input_length", 0)
        out = b.get("output_length", 0)
        dist = abs(inp - target_input_len) + abs(out - target_output_len)
        if dist < best_dist:
            best_dist = dist
            best_pair = (inp, out)

    matched = [
        b for b in benchmarks
        if b.get("input_length") == best_pair[0] and b.get("output_length") == best_pair[1]
    ]
    return sorted(matched, key=lambda b: b.get("batch_size", 0))


def find_peak_throughput(benchmarks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [b for b in benchmarks if "total_throughput_tok_per_sec" in b]
    if not valid:
        return None
    return max(valid, key=lambda b: b["total_throughput_tok_per_sec"])


# ---------------------------------------------------------------------------
# 4. Compute agent-to-server ratio
# ---------------------------------------------------------------------------

def compute_ratio(
    required: Dict[str, Any],
    max_throughput_tok_per_sec: float,
) -> Dict[str, Any]:
    sustained = required.get("sustained_demand_per_agent_tok_per_sec", 0)
    burst_raw = required.get("burst_demand_per_agent_tok_per_sec", 0)
    if isinstance(burst_raw, dict):
        burst_mean = burst_raw.get("mean", 0)
        burst_p75 = burst_raw.get("p75", 0)
    else:
        burst_mean = float(burst_raw)
        burst_p75 = float(burst_raw)

    return {
        "max_server_throughput_tok_per_sec": max_throughput_tok_per_sec,
        "sustained_demand_per_agent_tok_per_sec": sustained,
        "burst_demand_per_agent_mean_tok_per_sec": burst_mean,
        "burst_demand_per_agent_p75_tok_per_sec": burst_p75,
        "ratio_sustained": max_throughput_tok_per_sec / sustained if sustained > 0 else float("inf"),
        "ratio_burst_mean": max_throughput_tok_per_sec / burst_mean if burst_mean > 0 else float("inf"),
        "ratio_burst_p75": max_throughput_tok_per_sec / burst_p75 if burst_p75 > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# 5. Estimation mode
# ---------------------------------------------------------------------------

def estimate_required_throughput(
    mem_input_tokens: int,
    mem_output_tokens: int,
    agent_step_time: float,
    retain_most_turns: int,
) -> Dict[str, Any]:
    total_tokens = mem_input_tokens + mem_output_tokens
    req_rate = 1.0 / (retain_most_turns * agent_step_time)
    sustained_demand = req_rate * total_tokens
    burst_demand = total_tokens / agent_step_time

    return {
        "mode": "estimation",
        "mem_input_tokens": mem_input_tokens,
        "mem_output_tokens": mem_output_tokens,
        "mem_total_tokens": total_tokens,
        "agent_step_time_sec": agent_step_time,
        "retain_most_turns": retain_most_turns,
        "req_rate_per_agent_req_per_sec": req_rate,
        "sustained_demand_per_agent_tok_per_sec": sustained_demand,
        "burst_demand_per_agent_tok_per_sec": burst_demand,
    }


# ---------------------------------------------------------------------------
# 6. Pretty printing
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_stats(label: str, stats: Dict[str, float], unit: str = ""):
    u = f" {unit}" if unit else ""
    print(f"  {label}:")
    print(f"    mean={stats['mean']:.1f}{u}  median={stats['median']:.1f}{u}  "
          f"std={stats['std']:.1f}{u}")
    print(f"    [p25={stats['p25']:.1f}, p75={stats['p75']:.1f}, "
          f"min={stats['min']:.1f}, max={stats['max']:.1f}]  (n={stats['n']})")


def print_results(
    required: Dict[str, Any],
    matched_benchmarks: List[Dict[str, Any]],
    peak_benchmark: Optional[Dict[str, Any]],
    ratio: Optional[Dict[str, Any]],
):
    print_section("MEMORY MODEL THROUGHPUT ANALYSIS")

    if "error" in required:
        print(f"\n  ERROR: {required['error']}")
        return

    is_estimate = required.get("mode") == "estimation"

    if is_estimate:
        print_section("1. Required Throughput (Estimated)")
        print(f"  Parameters:")
        print(f"    Memory input tokens:  {required['mem_input_tokens']}")
        print(f"    Memory output tokens: {required['mem_output_tokens']}")
        print(f"    Memory total tokens:  {required['mem_total_tokens']}")
        print(f"    Agent step time:      {required['agent_step_time_sec']:.1f}s")
        print(f"    retain_most_turns:    {required['retain_most_turns']}")
        print(f"\n  Demand per agent:")
        print(f"    Request rate:          {required['req_rate_per_agent_req_per_sec']:.4f} req/s")
        print(f"    Sustained throughput:  {required['sustained_demand_per_agent_tok_per_sec']:.1f} tok/s")
        print(f"    Burst throughput:      {required['burst_demand_per_agent_tok_per_sec']:.1f} tok/s")
    else:
        print(f"\n  Trajectories: {required['num_trajectories']}")
        print(f"  Compression steps: {required['num_compression_steps']}  "
              f"(out of {required['num_total_steps']} total steps)")
        print(f"  retain_most_turns: {required['retain_most_turns']}")

        print_section("1. Memory Model Token Usage (per compression call)")
        print_stats("Input tokens", required["mem_input_tokens"], "tok")
        print_stats("Output tokens", required["mem_output_tokens"], "tok")
        print_stats("Total tokens", required["mem_total_tokens"], "tok")

        print_section("2. Timing")
        print_stats("Summary exec time", required["summ_exec_time_sec"], "s")
        print_stats("Agent LLM exec time (=time budget)", required["llm_exec_time_sec"], "s")
        print(f"\n  Mean total step time (all steps): {required['mean_total_step_time_sec']:.2f}s")
        print(f"  Bottleneck fraction (summ > LLM): {required['bottleneck_fraction']:.1%}")
        if required['bottleneck_fraction'] > 0:
            print_stats("Summ exceeds LLM by", required["summ_exceeds_llm_by_sec"], "s")

        print_section("3. Observed Memory Model Throughput (single-request)")
        print_stats("Total throughput (in+out)/time", required["observed_mem_throughput_tok_per_sec"], "tok/s")
        print_stats("Output throughput (out/time)", required["observed_mem_output_throughput_tok_per_sec"], "tok/s")

        print_section("4. Required Throughput per Agent")
        print_stats("Burst demand (tokens_per_call / time_budget)", required["burst_demand_per_agent_tok_per_sec"], "tok/s")
        print(f"\n  Request rate per agent: {required['req_rate_per_agent_req_per_sec']:.4f} req/s")
        print(f"  Sustained demand per agent: {required['sustained_demand_per_agent_tok_per_sec']:.1f} tok/s")

        mean_inp = int(required["mem_input_tokens"]["mean"])
        mean_out = int(required["mem_output_tokens"]["mean"])
        print(f"\n  Recommended vLLM benchmark parameters (matching real workload):")
        print(f"    --random-input-len {mean_inp}")
        print(f"    --random-output-len {mean_out}")

    if matched_benchmarks:
        print_section("5. Memory Model Maximum Throughput (vLLM benchmarks)")
        if not is_estimate:
            mean_inp = int(required["mem_input_tokens"]["mean"])
            mean_out = int(required["mem_output_tokens"]["mean"])
            best_inp = matched_benchmarks[0].get("input_length", "?")
            best_out = matched_benchmarks[0].get("output_length", "?")
            print(f"\n  Target workload: in={mean_inp}, out={mean_out}")
            print(f"  Best matching benchmark config: in={best_inp}, out={best_out}")

        print(f"\n  Concurrency sweep (sorted by batch_size):")
        for b in matched_benchmarks:
            bs = b.get("batch_size", "?")
            req_tp = b.get("request_throughput_req_per_sec", 0)
            out_tp = b.get("output_throughput_tok_per_sec", 0)
            tot_tp = b.get("total_throughput_tok_per_sec", 0)
            print(f"    batch={bs:>4}  |  req={req_tp:>7.2f} req/s  "
                  f"out={out_tp:>9.1f} tok/s  total={tot_tp:>9.1f} tok/s")

        if peak_benchmark:
            peak_tp = peak_benchmark["total_throughput_tok_per_sec"]
            peak_bs = peak_benchmark.get("batch_size", "?")
            print(f"\n  >> Peak throughput: {peak_tp:.1f} tok/s  (at batch_size={peak_bs})")

    if ratio:
        print_section("6. Agent-to-Server Ratio")
        print(f"  Max server throughput: {ratio['max_server_throughput_tok_per_sec']:.1f} tok/s")
        print(f"\n  Agents supportable per memory server:")
        print(f"    By sustained demand:    {ratio['ratio_sustained']:>8.0f} agents")
        print(f"    By burst demand (mean): {ratio['ratio_burst_mean']:>8.0f} agents")
        print(f"    By burst demand (p75):  {ratio['ratio_burst_p75']:>8.0f} agents")
        k = required.get("retain_most_turns", 3)
        print(f"\n  Interpretation:")
        print(f"    One memory model server can serve ~{ratio['ratio_sustained']:.0f}x agent model servers")
        print(f"    (sustained demand accounts for the 1/{k} call frequency)")

    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze memory model throughput requirements for COMEM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--traj_jsonl", type=str, nargs="*", default=None,
                        help="Path(s) to COMEM trajectory .jsonl file(s)")
    parser.add_argument("--retain_most_turns", type=int, default=3,
                        help="Compression frequency (default: 3)")

    parser.add_argument("--estimate", action="store_true",
                        help="Use estimation mode (no trajectory data needed)")
    parser.add_argument("--mem_input_tokens", type=int, default=4000,
                        help="Estimated avg memory input tokens (default: 4000)")
    parser.add_argument("--mem_output_tokens", type=int, default=1500,
                        help="Estimated avg memory output tokens (default: 1500)")
    parser.add_argument("--agent_step_time", type=float, default=15.0,
                        help="Estimated avg agent step time in seconds (default: 15.0)")

    parser.add_argument("--vllm_bench_dir", type=str, default=None,
                        help="Directory containing vllm bench .out files")
    parser.add_argument("--vllm_bench_files", type=str, nargs="*", default=None,
                        help="Specific vllm bench .out files")
    parser.add_argument("--max_throughput_tok_per_sec", type=float, default=None,
                        help="Override: manually specify max throughput (tok/s)")

    parser.add_argument("--output_json", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    # ---- Step 1: Required throughput ----
    if args.estimate:
        required = estimate_required_throughput(
            mem_input_tokens=args.mem_input_tokens,
            mem_output_tokens=args.mem_output_tokens,
            agent_step_time=args.agent_step_time,
            retain_most_turns=args.retain_most_turns,
        )
    else:
        if not args.traj_jsonl:
            print("ERROR: Provide --traj_jsonl <file.jsonl> or use --estimate mode.")
            sys.exit(1)

        all_trajs = []
        for f in args.traj_jsonl:
            print(f"Loading {f}...")
            all_trajs.extend(load_trajectories(f))
        print(f"Loaded {len(all_trajs)} trajectories from {len(args.traj_jsonl)} file(s).")

        required = compute_required_throughput(all_trajs, args.retain_most_turns)

    # ---- Step 2: Max throughput ----
    benchmarks = []
    if args.vllm_bench_dir:
        benchmarks = load_vllm_benchmarks(args.vllm_bench_dir)
    if args.vllm_bench_files:
        for f in args.vllm_bench_files:
            parsed = parse_vllm_bench_output(f)
            if parsed:
                benchmarks.append(parsed)

    matched_benchmarks = []
    peak_benchmark = None
    if benchmarks:
        if args.estimate:
            target_inp = args.mem_input_tokens
            target_out = args.mem_output_tokens
        else:
            target_inp = int(required.get("mem_input_tokens", {}).get("mean", 4000))
            target_out = int(required.get("mem_output_tokens", {}).get("mean", 1500))

        matched_benchmarks = find_best_matching_benchmarks(benchmarks, target_inp, target_out)
        peak_benchmark = find_peak_throughput(matched_benchmarks)

    # ---- Step 3: Ratio ----
    ratio = None
    max_tp = args.max_throughput_tok_per_sec
    if max_tp is None and peak_benchmark:
        max_tp = peak_benchmark["total_throughput_tok_per_sec"]
    if max_tp is not None and max_tp > 0:
        ratio = compute_ratio(required, max_tp)

    # ---- Print ----
    print_results(required, matched_benchmarks, peak_benchmark, ratio)

    # ---- Save ----
    if args.output_json:
        output = {
            "required_throughput": required,
            "matched_benchmarks": matched_benchmarks,
            "peak_benchmark": peak_benchmark,
            "ratio": ratio,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
