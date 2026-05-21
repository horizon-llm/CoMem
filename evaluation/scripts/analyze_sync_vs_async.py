#!/usr/bin/env python3
"""
Compare ASYNC (interleaved) vs SYNC (sequential) COMEM benchmark results.

Reads .jsonl trajectory files from both async and sync benchmark runs and
produces a side-by-side comparison that isolates the contribution of the
asynchronous systems design vs. the learned memory compression itself.

Key metrics:
  - async total_step_time ≈ max(llm_time, summ_time) + env_time
  - sync  total_step_time ≈ llm_time + summ_time + env_time
  - async_benefit = sync_step_time - async_step_time  (time saved by overlapping)

Usage:
    # Compare two grid benchmark directories:
    python scripts/analyze_sync_vs_async.py \
        --async_dir ./traj_benchmark/async-bench-h100-20260329_120000 \
        --sync_dir  ./traj_benchmark/sync-bench-h100-20260329_130000 \
        --output_dir ./analysis_results/sync_vs_async

    # Compare individual JSONL files:
    python scripts/analyze_sync_vs_async.py \
        --async_jsonl ./traj_benchmark/async/in512_out1024_m2048_s64_w16_n16.jsonl \
        --sync_jsonl  ./traj_benchmark/sync/sync_in512_out1024_m2048_s64_w16_n16.jsonl \
        --output_dir ./analysis_results/sync_vs_async
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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


def find_jsonl_files(directory: str) -> List[str]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.glob("*.jsonl"))


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
# Extract per-step metrics
# ---------------------------------------------------------------------------

def extract_step_metrics(trajs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract per-step timing and token metrics from trajectories."""
    records = []
    for traj in trajs:
        for step in traj.get("trajectory_steps", []):
            records.append({
                "step_idx": step.get("step_idx", 0),
                "llm_exec_time": step.get("llm_exec_time", 0.0),
                "summ_exec_time": step.get("summ_exec_time", 0.0),
                "prefill_exec_time": step.get("prefill_exec_time") or 0.0,
                "env_exec_time": step.get("env_exec_time", 0.0),
                "total_step_time": step.get("total_step_time", 0.0),
                "token_usage_prompt": step.get("token_usage_prompt", 0),
                "token_usage_completion": step.get("token_usage_completion", 0),
                "token_usage_summary": step.get("token_usage_summary", 0),
                "token_usage_summary_prompt": step.get("token_usage_summary_prompt", 0),
                "summary_start_time": step.get("summary_start_time"),
                "summary_finish_time": step.get("summary_finish_time"),
                "agent_start_time": step.get("agent_start_time"),
                "agent_finish_time": step.get("agent_finish_time"),
            })
    return records


def compute_summary(records: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Compute summary statistics for a set of step records."""
    if not records:
        return {"label": label, "error": "No steps found"}

    llm = np.array([r["llm_exec_time"] for r in records])
    summ = np.array([r["summ_exec_time"] for r in records])
    prefill = np.array([r["prefill_exec_time"] for r in records])
    env = np.array([r["env_exec_time"] for r in records])
    total = np.array([r["total_step_time"] for r in records])
    prompt_tok = np.array([r["token_usage_prompt"] for r in records])
    compl_tok = np.array([r["token_usage_completion"] for r in records])

    # Split into compression and non-compression steps
    comp_mask = summ > 0
    non_comp_mask = ~comp_mask

    return {
        "label": label,
        "num_steps": len(records),
        "num_compression_steps": int(comp_mask.sum()),
        "num_non_compression_steps": int(non_comp_mask.sum()),
        "llm_exec_time": _stats(llm),
        "summ_exec_time": _stats(summ),
        "summ_exec_time_compression_only": _stats(summ[comp_mask]) if comp_mask.any() else _stats(np.array([])),
        "prefill_exec_time": _stats(prefill),
        "env_exec_time": _stats(env),
        "total_step_time": _stats(total),
        "total_step_time_compression": _stats(total[comp_mask]) if comp_mask.any() else _stats(np.array([])),
        "total_step_time_non_compression": _stats(total[non_comp_mask]) if non_comp_mask.any() else _stats(np.array([])),
        "prompt_tokens": _stats(prompt_tok),
        "completion_tokens": _stats(compl_tok),
    }


# ---------------------------------------------------------------------------
# Per-step-index comparison (how metrics evolve over the trajectory)
# ---------------------------------------------------------------------------

def compute_per_step_idx(records: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    """Aggregate metrics by step_idx across all instances."""
    by_idx = defaultdict(lambda: {"llm": [], "summ": [], "env": [], "total": []})
    for r in records:
        idx = r["step_idx"]
        by_idx[idx]["llm"].append(r["llm_exec_time"])
        by_idx[idx]["summ"].append(r["summ_exec_time"])
        by_idx[idx]["env"].append(r["env_exec_time"])
        by_idx[idx]["total"].append(r["total_step_time"])

    result = {}
    for idx in sorted(by_idx.keys()):
        d = by_idx[idx]
        result[idx] = {
            "llm_mean": float(np.mean(d["llm"])),
            "summ_mean": float(np.mean(d["summ"])),
            "env_mean": float(np.mean(d["env"])),
            "total_mean": float(np.mean(d["total"])),
            "n": len(d["llm"]),
        }
    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(
    async_summary: Dict[str, Any],
    sync_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the comparison metrics between async and sync runs."""
    if "error" in async_summary or "error" in sync_summary:
        return {"error": "Missing data in one or both summaries"}

    async_total = async_summary["total_step_time"]["mean"]
    sync_total = sync_summary["total_step_time"]["mean"]
    async_llm = async_summary["llm_exec_time"]["mean"]
    sync_llm = sync_summary["llm_exec_time"]["mean"]
    async_summ = async_summary["summ_exec_time"]["mean"]
    sync_summ = sync_summary["summ_exec_time"]["mean"]

    # Time saved by async interleaving
    async_benefit_abs = sync_total - async_total
    async_benefit_pct = async_benefit_abs / sync_total if sync_total > 0 else 0

    # On compression steps specifically
    async_comp_total = async_summary["total_step_time_compression"]["mean"]
    sync_comp_total = sync_summary["total_step_time_compression"]["mean"]
    comp_benefit_abs = sync_comp_total - async_comp_total
    comp_benefit_pct = comp_benefit_abs / sync_comp_total if sync_comp_total > 0 else 0

    # Theoretical maximum async benefit:
    # If memory finishes within agent time, benefit = summ_time
    # If memory exceeds agent time, benefit = llm_time (partial overlap)
    theoretical_max_benefit = min(async_summ, async_llm) if async_summ > 0 else 0

    return {
        "async_avg_step_time": async_total,
        "sync_avg_step_time": sync_total,
        "async_benefit_seconds": async_benefit_abs,
        "async_benefit_percent": async_benefit_pct,
        "async_avg_llm_time": async_llm,
        "sync_avg_llm_time": sync_llm,
        "async_avg_summ_time": async_summ,
        "sync_avg_summ_time": sync_summ,
        "compression_step_async_total": async_comp_total,
        "compression_step_sync_total": sync_comp_total,
        "compression_step_benefit_seconds": comp_benefit_abs,
        "compression_step_benefit_percent": comp_benefit_pct,
        "theoretical_max_async_benefit": theoretical_max_benefit,
        "async_num_steps": async_summary["num_steps"],
        "sync_num_steps": sync_summary["num_steps"],
    }


# ---------------------------------------------------------------------------
# Matching async/sync JSONL files by config
# ---------------------------------------------------------------------------

def parse_config_from_filename(filename: str) -> Optional[str]:
    """Extract the config key from a benchmark JSONL filename.

    Async files:  in512_out1024_m2048_s64_w16_n16.jsonl
    Sync files:   sync_in512_out1024_m2048_s64_w16_n16.jsonl

    Returns the normalized config key (without sync_ prefix).
    """
    stem = Path(filename).stem
    # Strip sync_ prefix for matching
    if stem.startswith("sync_"):
        stem = stem[5:]
    # Check if it looks like a benchmark config
    if re.match(r"in\d+_out\d+_", stem):
        return stem
    return None


def match_files(
    async_files: List[str], sync_files: List[str]
) -> List[Tuple[str, str, str]]:
    """Match async and sync JSONL files by config key.

    Returns list of (config_key, async_path, sync_path).
    """
    async_by_config = {}
    for f in async_files:
        key = parse_config_from_filename(f)
        if key:
            async_by_config[key] = f

    matched = []
    for f in sync_files:
        key = parse_config_from_filename(f)
        if key and key in async_by_config:
            matched.append((key, async_by_config[key], f))

    return sorted(matched, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_step_time_comparison(
    async_per_idx: Dict[int, Dict[str, float]],
    sync_per_idx: Dict[int, Dict[str, float]],
    config_key: str,
    output_path: str,
):
    """Plot per-step total_step_time for async vs sync."""
    if not HAS_MATPLOTLIB:
        return

    common_idxs = sorted(set(async_per_idx.keys()) & set(sync_per_idx.keys()))
    if not common_idxs:
        return

    x = np.array(common_idxs)
    async_total = np.array([async_per_idx[i]["total_mean"] for i in common_idxs])
    sync_total = np.array([sync_per_idx[i]["total_mean"] for i in common_idxs])
    async_llm = np.array([async_per_idx[i]["llm_mean"] for i in common_idxs])
    async_summ = np.array([async_per_idx[i]["summ_mean"] for i in common_idxs])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total step time comparison
    ax = axes[0]
    ax.plot(x, async_total, "o-", label="Async (interleaved)", markersize=4)
    ax.plot(x, sync_total, "s-", label="Sync (sequential)", markersize=4)
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Total Step Time (s)")
    ax.set_title(f"Step Time: Async vs Sync\n{config_key}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: async benefit (time saved)
    ax = axes[1]
    benefit = sync_total - async_total
    colors = ["green" if b >= 0 else "red" for b in benefit]
    ax.bar(x, benefit, color=colors, alpha=0.7, width=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Time Saved by Async (s)")
    ax.set_title(f"Async Benefit per Step\n{config_key}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_breakdown(
    async_per_idx: Dict[int, Dict[str, float]],
    sync_per_idx: Dict[int, Dict[str, float]],
    config_key: str,
    output_path: str,
):
    """Stacked bar chart showing LLM/memory/env time breakdown."""
    if not HAS_MATPLOTLIB:
        return

    common_idxs = sorted(set(async_per_idx.keys()) & set(sync_per_idx.keys()))
    if not common_idxs:
        return

    x = np.arange(len(common_idxs))
    width = 0.35

    async_llm = np.array([async_per_idx[i]["llm_mean"] for i in common_idxs])
    async_summ = np.array([async_per_idx[i]["summ_mean"] for i in common_idxs])
    async_env = np.array([async_per_idx[i]["env_mean"] for i in common_idxs])
    sync_llm = np.array([sync_per_idx[i]["llm_mean"] for i in common_idxs])
    sync_summ = np.array([sync_per_idx[i]["summ_mean"] for i in common_idxs])
    sync_env = np.array([sync_per_idx[i]["env_mean"] for i in common_idxs])

    fig, ax = plt.subplots(figsize=(max(10, len(common_idxs) * 0.4), 6))

    # Async bars
    ax.bar(x - width / 2, async_llm, width, label="Async: LLM", color="#2196F3")
    ax.bar(x - width / 2, async_summ, width, bottom=async_llm,
           label="Async: Memory", color="#2196F3", alpha=0.4, hatch="//")
    ax.bar(x - width / 2, async_env, width, bottom=async_llm + async_summ,
           label="Async: Env", color="#2196F3", alpha=0.2)

    # Sync bars
    ax.bar(x + width / 2, sync_llm, width, label="Sync: LLM", color="#FF9800")
    ax.bar(x + width / 2, sync_summ, width, bottom=sync_llm,
           label="Sync: Memory", color="#FF9800", alpha=0.4, hatch="//")
    ax.bar(x + width / 2, sync_env, width, bottom=sync_llm + sync_summ,
           label="Sync: Env", color="#FF9800", alpha=0.2)

    ax.set_xlabel("Step Index")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Time Breakdown: Async vs Sync\n{config_key}")
    ax.set_xticks(x)
    ax.set_xticklabels(common_idxs, fontsize=7)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_grid_summary(
    comparisons: List[Dict[str, Any]],
    config_keys: List[str],
    output_path: str,
):
    """Bar chart summarizing async benefit across all grid configs."""
    if not HAS_MATPLOTLIB or not comparisons:
        return

    labels = config_keys
    async_times = [c["async_avg_step_time"] for c in comparisons]
    sync_times = [c["sync_avg_step_time"] for c in comparisons]
    benefits_pct = [c["async_benefit_percent"] * 100 for c in comparisons]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(labels) * 1.5), 5))

    x = np.arange(len(labels))
    width = 0.35

    # Left: absolute times
    ax = axes[0]
    ax.bar(x - width / 2, async_times, width, label="Async", color="#2196F3")
    ax.bar(x + width / 2, sync_times, width, label="Sync", color="#FF9800")
    ax.set_ylabel("Avg Step Time (s)")
    ax.set_title("Avg Step Time: Async vs Sync")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Right: benefit percentage
    ax = axes[1]
    colors = ["green" if b >= 0 else "red" for b in benefits_pct]
    ax.bar(x, benefits_pct, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Async Benefit (%)")
    ax.set_title("Time Saved by Async Interleaving")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def print_comparison(
    config_key: str,
    async_summary: Dict[str, Any],
    sync_summary: Dict[str, Any],
    comp: Dict[str, Any],
):
    """Print a single async vs sync comparison."""
    if "error" in comp:
        print(f"\n  {config_key}: {comp['error']}")
        return

    print(f"\n  Config: {config_key}")
    print(f"  {'':>30}  {'Async':>12}  {'Sync':>12}  {'Diff':>12}  {'Benefit':>10}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")

    async_t = comp["async_avg_step_time"]
    sync_t = comp["sync_avg_step_time"]
    diff = comp["async_benefit_seconds"]
    pct = comp["async_benefit_percent"]
    print(f"  {'Avg total step time (s)':>30}  {async_t:>12.3f}  {sync_t:>12.3f}  {diff:>+12.3f}  {pct:>9.1%}")

    async_llm = comp["async_avg_llm_time"]
    sync_llm = comp["sync_avg_llm_time"]
    llm_diff = sync_llm - async_llm
    print(f"  {'Avg LLM time (s)':>30}  {async_llm:>12.3f}  {sync_llm:>12.3f}  {llm_diff:>+12.3f}")

    async_summ = comp["async_avg_summ_time"]
    sync_summ = comp["sync_avg_summ_time"]
    summ_diff = sync_summ - async_summ
    print(f"  {'Avg memory time (s)':>30}  {async_summ:>12.3f}  {sync_summ:>12.3f}  {summ_diff:>+12.3f}")

    ac = comp["compression_step_async_total"]
    sc = comp["compression_step_sync_total"]
    cd = comp["compression_step_benefit_seconds"]
    cp = comp["compression_step_benefit_percent"]
    print(f"  {'Compression step time (s)':>30}  {ac:>12.3f}  {sc:>12.3f}  {cd:>+12.3f}  {cp:>9.1%}")

    n_async = comp["async_num_steps"]
    n_sync = comp["sync_num_steps"]
    print(f"  {'Num steps':>30}  {n_async:>12}  {n_sync:>12}")

    print(f"\n  Interpretation:")
    if pct > 0.01:
        print(f"    Async interleaving saves {pct:.1%} of per-step time ({diff:.3f}s per step)")
        print(f"    On compression steps specifically, saves {cp:.1%} ({cd:.3f}s)")
    elif pct > -0.01:
        print(f"    Async and sync have similar step times (difference: {diff:.3f}s, {pct:.1%})")
    else:
        print(f"    Sync is faster by {-pct:.1%} — async overhead exceeds overlap benefit")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare async vs sync COMEM benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--async_dir", type=str, default=None,
                        help="Directory with async benchmark .jsonl files")
    parser.add_argument("--sync_dir", type=str, default=None,
                        help="Directory with sync benchmark .jsonl files")
    parser.add_argument("--async_jsonl", type=str, nargs="*", default=None,
                        help="Specific async .jsonl file(s)")
    parser.add_argument("--sync_jsonl", type=str, nargs="*", default=None,
                        help="Specific sync .jsonl file(s)")
    parser.add_argument("--output_dir", type=str, default="./analysis_results/sync_vs_async",
                        help="Output directory for plots and summary")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save full results to JSON")

    args = parser.parse_args()

    # Collect files
    async_files = []
    sync_files = []

    if args.async_dir:
        async_files.extend(find_jsonl_files(args.async_dir))
    if args.sync_dir:
        sync_files.extend(find_jsonl_files(args.sync_dir))
    if args.async_jsonl:
        async_files.extend(args.async_jsonl)
    if args.sync_jsonl:
        sync_files.extend(args.sync_jsonl)

    if not async_files or not sync_files:
        print("ERROR: Provide both async and sync data via --async_dir/--sync_dir or --async_jsonl/--sync_jsonl")
        sys.exit(1)

    # Match files by config
    matched = match_files(async_files, sync_files)

    # If no config-based matching works, fall back to positional pairing
    if not matched and len(async_files) == len(sync_files):
        print("  No config-based matching; falling back to positional pairing.")
        for i, (af, sf) in enumerate(zip(async_files, sync_files)):
            matched.append((f"config_{i}", af, sf))

    if not matched:
        print(f"ERROR: Could not match async ({len(async_files)} files) "
              f"and sync ({len(sync_files)} files) results.")
        print(f"  Async files: {async_files}")
        print(f"  Sync files: {sync_files}")
        sys.exit(1)

    print(f"Matched {len(matched)} config(s) for comparison.")

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each matched pair
    all_comparisons = []
    all_config_keys = []
    all_results = {}

    print_section("ASYNC vs SYNC COMEM BENCHMARK COMPARISON")

    for config_key, async_path, sync_path in matched:
        print(f"\n  Loading {config_key}...")
        print(f"    Async: {async_path}")
        print(f"    Sync:  {sync_path}")

        async_trajs = load_trajectories(async_path)
        sync_trajs = load_trajectories(sync_path)

        async_records = extract_step_metrics(async_trajs)
        sync_records = extract_step_metrics(sync_trajs)

        async_summary = compute_summary(async_records, f"async_{config_key}")
        sync_summary = compute_summary(sync_records, f"sync_{config_key}")

        comp = compare(async_summary, sync_summary)

        print_comparison(config_key, async_summary, sync_summary, comp)

        # Per-step-index plots
        async_per_idx = compute_per_step_idx(async_records)
        sync_per_idx = compute_per_step_idx(sync_records)

        if HAS_MATPLOTLIB:
            safe_key = config_key.replace("/", "_")
            plot_step_time_comparison(
                async_per_idx, sync_per_idx, config_key,
                os.path.join(args.output_dir, f"step_time_{safe_key}.png"),
            )
            plot_time_breakdown(
                async_per_idx, sync_per_idx, config_key,
                os.path.join(args.output_dir, f"breakdown_{safe_key}.png"),
            )

        all_comparisons.append(comp)
        all_config_keys.append(config_key)
        all_results[config_key] = {
            "async_summary": async_summary,
            "sync_summary": sync_summary,
            "comparison": comp,
        }

    # Grid summary
    valid_comparisons = [c for c in all_comparisons if "error" not in c]
    valid_keys = [k for k, c in zip(all_config_keys, all_comparisons) if "error" not in c]

    if valid_comparisons:
        print_section("GRID SUMMARY")
        print(f"\n  {'Config':<45} {'Async(s)':>10} {'Sync(s)':>10} {'Saved(s)':>10} {'Benefit':>10}")
        print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for key, comp in zip(valid_keys, valid_comparisons):
            print(f"  {key:<45} "
                  f"{comp['async_avg_step_time']:>10.3f} "
                  f"{comp['sync_avg_step_time']:>10.3f} "
                  f"{comp['async_benefit_seconds']:>+10.3f} "
                  f"{comp['async_benefit_percent']:>9.1%}")

        # Aggregated
        all_benefits = [c["async_benefit_percent"] for c in valid_comparisons]
        all_abs = [c["async_benefit_seconds"] for c in valid_comparisons]
        print(f"\n  Overall async benefit:")
        print(f"    Mean:   {np.mean(all_benefits):.1%} ({np.mean(all_abs):.3f}s)")
        print(f"    Median: {np.median(all_benefits):.1%} ({np.median(all_abs):.3f}s)")
        print(f"    Range:  {np.min(all_benefits):.1%} to {np.max(all_benefits):.1%}")

        # Save CSV summary
        csv_path = os.path.join(args.output_dir, "sync_vs_async_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "config", "async_step_time", "sync_step_time",
                "benefit_seconds", "benefit_percent",
                "async_llm_time", "sync_llm_time",
                "async_summ_time", "sync_summ_time",
                "comp_async_time", "comp_sync_time",
                "comp_benefit_seconds", "comp_benefit_percent",
            ])
            writer.writeheader()
            for key, comp in zip(valid_keys, valid_comparisons):
                writer.writerow({
                    "config": key,
                    "async_step_time": round(comp["async_avg_step_time"], 4),
                    "sync_step_time": round(comp["sync_avg_step_time"], 4),
                    "benefit_seconds": round(comp["async_benefit_seconds"], 4),
                    "benefit_percent": round(comp["async_benefit_percent"], 4),
                    "async_llm_time": round(comp["async_avg_llm_time"], 4),
                    "sync_llm_time": round(comp["sync_avg_llm_time"], 4),
                    "async_summ_time": round(comp["async_avg_summ_time"], 4),
                    "sync_summ_time": round(comp["sync_avg_summ_time"], 4),
                    "comp_async_time": round(comp["compression_step_async_total"], 4),
                    "comp_sync_time": round(comp["compression_step_sync_total"], 4),
                    "comp_benefit_seconds": round(comp["compression_step_benefit_seconds"], 4),
                    "comp_benefit_percent": round(comp["compression_step_benefit_percent"], 4),
                })
        print(f"\n  CSV summary saved to {csv_path}")

        # Grid summary plot
        if HAS_MATPLOTLIB:
            plot_grid_summary(
                valid_comparisons, valid_keys,
                os.path.join(args.output_dir, "grid_summary.png"),
            )
            print(f"  Plots saved to {args.output_dir}/")

    # Save JSON
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Full results saved to {args.output_json}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
