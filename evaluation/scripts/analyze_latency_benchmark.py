#!/usr/bin/env python3
"""
Post-hoc analysis of COMEM vs Full-Context latency benchmark results.

Reads the .jsonl trajectory files produced by:
  - edit_benchmark.py        (COMEM results)
  - edit_benchmark_full_context.py (Full-Context results)

Produces per-step comparison plots and summary tables for:
  1. Time dimension   – per-step LLM latency, summary latency, total step time
  2. Memory dimension – per-step prompt tokens, GPU KV-cache usage (from vLLM metrics)
  3. Crossover analysis – at what context length COMEM becomes faster

Usage:
    python scripts/analyze_latency_benchmark.py \
        --comem_dir   ./traj_benchmark/latency-bench-h100-20260325_120000 \
        --fc_dir      ./traj_benchmark/fc-latency-bench-h100-20260325_130000 \
        --output_dir  ./analysis_results

    # Or point to individual JSONL files:
    python scripts/analyze_latency_benchmark.py \
        --comem_jsonl ./traj_benchmark/comem/in512_out1024_mdef_s32_w128_n128.jsonl \
        --fc_jsonl    ./traj_benchmark/fc/in512_out1024_s32_w128_n128.jsonl \
        --output_dir  ./analysis_results
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# vLLM Prometheus metrics parser
# ---------------------------------------------------------------------------

# Gauge metrics (instantaneous values — usable directly per step)
VLLM_GAUGE_PATTERNS = {
    "kv_cache_usage_perc": re.compile(
        r'^vllm:kv_cache_usage_perc\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
    "num_requests_running": re.compile(
        r'^vllm:num_requests_running\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
    "num_requests_waiting": re.compile(
        r'^vllm:num_requests_waiting\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
}

# Counter / histogram-sum metrics (cumulative — need delta for per-step values)
VLLM_COUNTER_PATTERNS = {
    "prefix_cache_hits_total": re.compile(
        r'^vllm:prefix_cache_hits_total\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
    "prefix_cache_queries_total": re.compile(
        r'^vllm:prefix_cache_queries_total\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
    "prompt_tokens_total": re.compile(
        r'^vllm:prompt_tokens_total\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
    "generation_tokens_total": re.compile(
        r'^vllm:generation_tokens_total\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
    "num_preemptions_total": re.compile(
        r'^vllm:num_preemptions_total\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
    "request_prefill_kv_computed_tokens_sum": re.compile(
        r'^vllm:request_prefill_kv_computed_tokens_sum\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
    "request_prompt_tokens_sum": re.compile(
        r'^vllm:request_prompt_tokens_sum\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
    "request_prefill_time_seconds_sum": re.compile(
        r'^vllm:request_prefill_time_seconds_sum\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
    "request_prompt_tokens_count": re.compile(
        r'^vllm:request_prompt_tokens_count\b.*?\s+([\d.eE+-]+)', re.MULTILINE
    ),
}


def parse_vllm_metrics(raw_text: Optional[str]) -> Dict[str, Optional[float]]:
    """Parse Prometheus-format vLLM metrics text into a dict of floats.

    Returns both gauge values (directly usable) and raw counter values
    (need delta computation across steps for per-step rates).
    """
    result: Dict[str, Optional[float]] = {}
    if not raw_text:
        return result

    for key, pattern in VLLM_GAUGE_PATTERNS.items():
        match = pattern.search(raw_text)
        result[key] = float(match.group(1)) if match else None

    for key, pattern in VLLM_COUNTER_PATTERNS.items():
        match = pattern.search(raw_text)
        result[f"counter_{key}"] = float(match.group(1)) if match else None

    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectories(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load all trajectory JSON objects from a .jsonl file."""
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
    """Find all .jsonl files in a directory (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.glob("*.jsonl"))


def detect_step_type(traj: Dict[str, Any]) -> str:
    """Detect whether a trajectory uses COMEM (has summ_exec_time) or full-context steps."""
    if not traj.get("trajectory_steps"):
        return "unknown"
    step0 = traj["trajectory_steps"][0]
    if "summ_exec_time" in step0:
        return "comem"
    return "full_context"


# ---------------------------------------------------------------------------
# Per-step extraction
# ---------------------------------------------------------------------------

def extract_per_step_data(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract per-step metrics from a single trajectory.

    For vLLM counter metrics, computes per-step deltas and derived rates
    (e.g., prefix cache hit rate) by diffing consecutive snapshots.
    """
    steps_data = []
    traj_type = detect_step_type(traj)
    prev_counters: Dict[str, float] = {}

    for step in traj["trajectory_steps"]:
        row = {
            "step_idx": step.get("step_idx", step.get("step_count", 0)),
            "type": traj_type,
            # Time metrics
            "llm_exec_time": step.get("llm_exec_time", 0.0),
            "total_step_time": step.get("total_step_time", 0.0),
            "env_exec_time": step.get("env_exec_time", 0.0),
            # Token metrics
            "token_usage_prompt": step.get("token_usage_prompt", 0),
            "token_usage_completion": step.get("token_usage_completion", 0),
            "token_usage_total": step.get("token_usage_total", 0),
        }

        # COMEM-specific
        if traj_type == "comem":
            row["summ_exec_time"] = step.get("summ_exec_time", 0.0)
            row["prefill_exec_time"] = step.get("prefill_exec_time") or 0.0
            row["token_usage_summary"] = step.get("token_usage_summary", 0)
            row["token_usage_summary_prompt"] = step.get("token_usage_summary_prompt", 0)
            # Effective time: agent inference overlaps with summary+prefill
            row["effective_llm_time"] = max(
                row["llm_exec_time"],
                row["summ_exec_time"] + row["prefill_exec_time"],
            )
        else:
            row["summ_exec_time"] = 0.0
            row["prefill_exec_time"] = 0.0
            row["token_usage_summary"] = 0
            row["token_usage_summary_prompt"] = 0
            row["effective_llm_time"] = row["llm_exec_time"]

        # Parse vLLM metrics
        vllm = parse_vllm_metrics(step.get("vllm_metrics"))

        # Store gauge metrics directly
        for k in VLLM_GAUGE_PATTERNS:
            row[f"vllm_{k}"] = vllm.get(k)

        # Compute per-step deltas for counter metrics
        curr_counters = {k: vllm.get(f"counter_{k}") for k in VLLM_COUNTER_PATTERNS}
        for k in VLLM_COUNTER_PATTERNS:
            curr = curr_counters.get(k)
            prev = prev_counters.get(k)
            if curr is not None and prev is not None and curr >= prev:
                row[f"vllm_delta_{k}"] = curr - prev
            else:
                row[f"vllm_delta_{k}"] = None
        prev_counters = curr_counters

        # Derive prefix cache hit rate from deltas
        d_hits = row.get("vllm_delta_prefix_cache_hits_total")
        d_queries = row.get("vllm_delta_prefix_cache_queries_total")
        if d_hits is not None and d_queries is not None and d_queries > 0:
            row["vllm_prefix_cache_hit_rate"] = d_hits / d_queries
        else:
            row["vllm_prefix_cache_hit_rate"] = None

        # Derive KV reuse ratio: fraction of prompt tokens served from cache
        d_computed = row.get("vllm_delta_request_prefill_kv_computed_tokens_sum")
        d_prompt = row.get("vllm_delta_request_prompt_tokens_sum")
        if d_computed is not None and d_prompt is not None and d_prompt > 0:
            row["vllm_kv_reuse_ratio"] = 1.0 - (d_computed / d_prompt)
        else:
            row["vllm_kv_reuse_ratio"] = None

        steps_data.append(row)

    return steps_data


def aggregate_across_trajectories(
    all_trajs: List[Dict[str, Any]],
) -> Dict[int, Dict[str, List[float]]]:
    """Aggregate per-step metrics across multiple trajectory instances.

    Returns: {step_idx: {metric_name: [values across instances]}}
    """
    by_step = defaultdict(lambda: defaultdict(list))

    for traj in all_trajs:
        steps = extract_per_step_data(traj)
        for s in steps:
            idx = s["step_idx"]
            for key, val in s.items():
                if key in ("step_idx", "type"):
                    continue
                if val is not None:
                    by_step[idx][key].append(val)

    return by_step


def compute_step_stats(
    by_step: Dict[int, Dict[str, List[float]]],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Compute mean, std, median, p25, p75 for each metric at each step.

    Returns: {step_idx: {metric_name: {"mean": ..., "std": ..., "median": ..., "p25": ..., "p75": ...}}}
    """
    stats = {}
    for step_idx in sorted(by_step.keys()):
        stats[step_idx] = {}
        for metric, values in by_step[step_idx].items():
            arr = np.array(values)
            stats[step_idx][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "n": len(arr),
            }
    return stats


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

PLOT_STYLE = {
    "comem": {"color": "#2196F3", "marker": "o", "label": "COMEM"},
    "full_context": {"color": "#FF5722", "marker": "s", "label": "Full-Context"},
    "comem_summ": {"color": "#4CAF50", "marker": "^", "label": "COMEM (summary)"},
    "comem_prefill": {"color": "#9C27B0", "marker": "v", "label": "COMEM (prefill)"},
    "comem_effective": {"color": "#00BCD4", "marker": "D", "label": "COMEM (effective)"},
}


def _setup_figure(title: str, xlabel: str, ylabel: str, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    return fig, ax


def _plot_metric_line(
    ax, steps: List[int], stats: Dict, metric: str, style: Dict,
    use_fill: bool = True,
):
    """Plot mean line with optional p25-p75 shading."""
    means = [stats[s][metric]["mean"] for s in steps if metric in stats[s]]
    valid_steps = [s for s in steps if metric in stats[s]]
    if not means:
        return
    ax.plot(valid_steps, means, **{k: v for k, v in style.items() if k != "label"},
            label=style["label"], linewidth=2, markersize=5)
    if use_fill:
        p25 = [stats[s][metric]["p25"] for s in valid_steps]
        p75 = [stats[s][metric]["p75"] for s in valid_steps]
        ax.fill_between(valid_steps, p25, p75, alpha=0.15, color=style["color"])


def _add_crossover_annotation(ax, steps, comem_vals, fc_vals):
    """Find and annotate crossover point where COMEM becomes faster."""
    for i in range(1, len(steps)):
        if i < len(comem_vals) and i < len(fc_vals):
            # Check if FC was faster before and COMEM is faster now
            if comem_vals[i - 1] >= fc_vals[i - 1] and comem_vals[i] < fc_vals[i]:
                # Linear interpolation for crossover step
                x_cross = steps[i - 1] + (steps[i] - steps[i - 1]) * (
                    (comem_vals[i - 1] - fc_vals[i - 1])
                    / ((comem_vals[i - 1] - fc_vals[i - 1]) - (comem_vals[i] - fc_vals[i]))
                )
                y_cross = comem_vals[i - 1] + (comem_vals[i] - comem_vals[i - 1]) * (
                    (x_cross - steps[i - 1]) / (steps[i] - steps[i - 1])
                )
                ax.axvline(x=x_cross, color="gray", linestyle="--", alpha=0.7)
                ax.annotate(
                    f"Crossover\n(step {x_cross:.1f})",
                    xy=(x_cross, y_cross),
                    xytext=(x_cross + 1, y_cross * 1.2),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    color="gray",
                )
                return x_cross
    return None


# ---------------------------------------------------------------------------
# Main plot functions
# ---------------------------------------------------------------------------

def plot_time_comparison(
    comem_stats: Dict, fc_stats: Dict, output_dir: str, step_config: str = "",
):
    """Plot 1: Time dimension — per-step LLM latency comparison."""
    comem_steps = sorted(comem_stats.keys())
    fc_steps = sorted(fc_stats.keys())
    common_steps = sorted(set(comem_steps) & set(fc_steps))

    if not common_steps:
        # Use all available steps even if not perfectly matched
        max_step = max(max(comem_steps, default=0), max(fc_steps, default=0))
        all_steps_range = range(max_step + 1)
    else:
        all_steps_range = common_steps

    # --- Plot 1a: LLM exec time ---
    fig, ax = _setup_figure(
        f"Per-Step LLM Inference Time{' — ' + step_config if step_config else ''}",
        "Step Index", "LLM Exec Time (s)"
    )

    if comem_steps:
        _plot_metric_line(ax, comem_steps, comem_stats, "llm_exec_time", PLOT_STYLE["comem"])
    if fc_steps:
        _plot_metric_line(ax, fc_steps, fc_stats, "llm_exec_time", PLOT_STYLE["full_context"])

    # Annotate crossover
    if common_steps:
        comem_vals = [comem_stats[s]["llm_exec_time"]["mean"] for s in common_steps
                      if "llm_exec_time" in comem_stats[s]]
        fc_vals = [fc_stats[s]["llm_exec_time"]["mean"] for s in common_steps
                   if "llm_exec_time" in fc_stats[s]]
        _add_crossover_annotation(ax, common_steps, comem_vals, fc_vals)

    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"time_llm_exec{step_config}.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, f"time_llm_exec{step_config}.pdf"))
    plt.close(fig)

    # --- Plot 1b: Effective step time (including summary overhead for COMEM) ---
    fig, ax = _setup_figure(
        f"Per-Step Effective Time (with interleaved summary){' — ' + step_config if step_config else ''}",
        "Step Index", "Effective Step Time (s)"
    )

    if comem_steps:
        _plot_metric_line(ax, comem_steps, comem_stats, "effective_llm_time", PLOT_STYLE["comem_effective"])
        _plot_metric_line(ax, comem_steps, comem_stats, "llm_exec_time", PLOT_STYLE["comem"])
        _plot_metric_line(ax, comem_steps, comem_stats, "summ_exec_time", PLOT_STYLE["comem_summ"])
    if fc_steps:
        _plot_metric_line(ax, fc_steps, fc_stats, "llm_exec_time", PLOT_STYLE["full_context"])

    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"time_effective{step_config}.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, f"time_effective{step_config}.pdf"))
    plt.close(fig)

    # --- Plot 1c: Total step time ---
    fig, ax = _setup_figure(
        f"Per-Step Total Wall Time{' — ' + step_config if step_config else ''}",
        "Step Index", "Total Step Time (s)"
    )

    if comem_steps:
        _plot_metric_line(ax, comem_steps, comem_stats, "total_step_time", PLOT_STYLE["comem"])
    if fc_steps:
        _plot_metric_line(ax, fc_steps, fc_stats, "total_step_time", PLOT_STYLE["full_context"])

    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"time_total_step{step_config}.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, f"time_total_step{step_config}.pdf"))
    plt.close(fig)


def _has_metric(stats: Dict, steps: List[int], metric: str) -> bool:
    """Check if a metric has any non-None values in the stats."""
    return any(
        metric in stats[s] and stats[s][metric]["n"] > 0
        for s in steps
    )


def plot_memory_comparison(
    comem_stats: Dict, fc_stats: Dict, output_dir: str, step_config: str = "",
):
    """Plot 2: Memory dimension — prompt tokens, KV cache usage, prefix cache hit rate."""
    comem_steps = sorted(comem_stats.keys())
    fc_steps = sorted(fc_stats.keys())

    # --- Plot 2a: Prompt tokens per step ---
    fig, ax = _setup_figure(
        f"Per-Step Prompt Tokens{' — ' + step_config if step_config else ''}",
        "Step Index", "Prompt Tokens"
    )

    if comem_steps:
        _plot_metric_line(ax, comem_steps, comem_stats, "token_usage_prompt", PLOT_STYLE["comem"])
    if fc_steps:
        _plot_metric_line(ax, fc_steps, fc_stats, "token_usage_prompt", PLOT_STYLE["full_context"])

    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"memory_prompt_tokens{step_config}.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, f"memory_prompt_tokens{step_config}.pdf"))
    plt.close(fig)

    # --- Plot 2b: GPU KV-cache usage percentage (gauge) ---
    has_comem_kv = _has_metric(comem_stats, comem_steps, "vllm_kv_cache_usage_perc")
    has_fc_kv = _has_metric(fc_stats, fc_steps, "vllm_kv_cache_usage_perc")

    if has_comem_kv or has_fc_kv:
        fig, ax = _setup_figure(
            f"Per-Step GPU KV-Cache Usage{' — ' + step_config if step_config else ''}",
            "Step Index", "KV-Cache Usage (fraction)"
        )

        if has_comem_kv:
            _plot_metric_line(ax, comem_steps, comem_stats, "vllm_kv_cache_usage_perc", PLOT_STYLE["comem"])
        if has_fc_kv:
            _plot_metric_line(ax, fc_steps, fc_stats, "vllm_kv_cache_usage_perc", PLOT_STYLE["full_context"])

        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"memory_kv_cache_usage{step_config}.png"), dpi=150)
        fig.savefig(os.path.join(output_dir, f"memory_kv_cache_usage{step_config}.pdf"))
        plt.close(fig)
    else:
        print("  Note: No vllm:kv_cache_usage_perc found, skipping KV cache usage plot.")

    # --- Plot 2c: Prefix cache hit rate (derived from counter deltas) ---
    has_comem_pchr = _has_metric(comem_stats, comem_steps, "vllm_prefix_cache_hit_rate")
    has_fc_pchr = _has_metric(fc_stats, fc_steps, "vllm_prefix_cache_hit_rate")

    if has_comem_pchr or has_fc_pchr:
        fig, ax = _setup_figure(
            f"Per-Step Prefix Cache Hit Rate{' — ' + step_config if step_config else ''}",
            "Step Index", "Prefix Cache Hit Rate"
        )

        if has_comem_pchr:
            _plot_metric_line(ax, comem_steps, comem_stats, "vllm_prefix_cache_hit_rate", PLOT_STYLE["comem"])
        if has_fc_pchr:
            _plot_metric_line(ax, fc_steps, fc_stats, "vllm_prefix_cache_hit_rate", PLOT_STYLE["full_context"])

        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"memory_prefix_cache_hit_rate{step_config}.png"), dpi=150)
        fig.savefig(os.path.join(output_dir, f"memory_prefix_cache_hit_rate{step_config}.pdf"))
        plt.close(fig)
    else:
        print("  Note: No prefix cache hit rate data found, skipping prefix cache plot.")

    # --- Plot 2d: KV reuse ratio (1 - computed/total prompt tokens) ---
    has_comem_kvr = _has_metric(comem_stats, comem_steps, "vllm_kv_reuse_ratio")
    has_fc_kvr = _has_metric(fc_stats, fc_steps, "vllm_kv_reuse_ratio")

    if has_comem_kvr or has_fc_kvr:
        fig, ax = _setup_figure(
            f"Per-Step KV Reuse Ratio{' — ' + step_config if step_config else ''}",
            "Step Index", "KV Reuse Ratio (1 = fully cached)"
        )

        if has_comem_kvr:
            _plot_metric_line(ax, comem_steps, comem_stats, "vllm_kv_reuse_ratio", PLOT_STYLE["comem"])
        if has_fc_kvr:
            _plot_metric_line(ax, fc_steps, fc_stats, "vllm_kv_reuse_ratio", PLOT_STYLE["full_context"])

        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"memory_kv_reuse_ratio{step_config}.png"), dpi=150)
        fig.savefig(os.path.join(output_dir, f"memory_kv_reuse_ratio{step_config}.pdf"))
        plt.close(fig)
    else:
        print("  Note: No KV reuse ratio data found, skipping KV reuse plot.")

    # --- Plot 2e: Cumulative token budget ---
    fig, ax = _setup_figure(
        f"Cumulative Prompt Tokens{' — ' + step_config if step_config else ''}",
        "Step Index", "Cumulative Prompt Tokens"
    )

    if comem_steps:
        comem_cumulative = np.cumsum([
            comem_stats[s]["token_usage_prompt"]["mean"]
            for s in comem_steps if "token_usage_prompt" in comem_stats[s]
        ])
        valid_comem = [s for s in comem_steps if "token_usage_prompt" in comem_stats[s]]
        ax.plot(valid_comem, comem_cumulative, **{k: v for k, v in PLOT_STYLE["comem"].items() if k != "label"},
                label="COMEM (agent only)", linewidth=2, markersize=5)

        # Also show cumulative including summary prompt tokens
        if any("token_usage_summary_prompt" in comem_stats[s] for s in comem_steps):
            comem_cumulative_total = np.cumsum([
                comem_stats[s]["token_usage_prompt"]["mean"]
                + comem_stats[s].get("token_usage_summary_prompt", {"mean": 0})["mean"]
                for s in comem_steps if "token_usage_prompt" in comem_stats[s]
            ])
            ax.plot(valid_comem, comem_cumulative_total, color="#2196F3", marker="o",
                    label="COMEM (agent + summary model)", linewidth=2, markersize=5,
                    linestyle="--", alpha=0.7)

    if fc_steps:
        fc_cumulative = np.cumsum([
            fc_stats[s]["token_usage_prompt"]["mean"]
            for s in fc_steps if "token_usage_prompt" in fc_stats[s]
        ])
        valid_fc = [s for s in fc_steps if "token_usage_prompt" in fc_stats[s]]
        ax.plot(valid_fc, fc_cumulative, **{k: v for k, v in PLOT_STYLE["full_context"].items() if k != "label"},
                label="Full-Context", linewidth=2, markersize=5)

    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"memory_cumulative_tokens{step_config}.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, f"memory_cumulative_tokens{step_config}.pdf"))
    plt.close(fig)


def plot_latency_vs_context(
    comem_stats: Dict, fc_stats: Dict, output_dir: str, step_config: str = "",
):
    """Plot 3: Latency vs context length — shows crossover threshold."""
    comem_steps = sorted(comem_stats.keys())
    fc_steps = sorted(fc_stats.keys())

    fig, ax = _setup_figure(
        f"LLM Latency vs Prompt Token Count{' — ' + step_config if step_config else ''}",
        "Prompt Tokens", "LLM Exec Time (s)"
    )

    if comem_steps:
        x = [comem_stats[s]["token_usage_prompt"]["mean"] for s in comem_steps
             if "token_usage_prompt" in comem_stats[s] and "llm_exec_time" in comem_stats[s]]
        y = [comem_stats[s]["llm_exec_time"]["mean"] for s in comem_steps
             if "token_usage_prompt" in comem_stats[s] and "llm_exec_time" in comem_stats[s]]
        if x and y:
            ax.plot(x, y, **{k: v for k, v in PLOT_STYLE["comem"].items() if k != "label"},
                    label="COMEM", linewidth=2, markersize=5)
            # Annotate step indices
            valid_comem = [s for s in comem_steps
                           if "token_usage_prompt" in comem_stats[s] and "llm_exec_time" in comem_stats[s]]
            for i, s in enumerate(valid_comem):
                if i % max(1, len(valid_comem) // 8) == 0:  # annotate ~8 points
                    ax.annotate(f"s{s}", (x[i], y[i]), fontsize=7, alpha=0.6,
                                xytext=(5, 5), textcoords="offset points")

    if fc_steps:
        x = [fc_stats[s]["token_usage_prompt"]["mean"] for s in fc_steps
             if "token_usage_prompt" in fc_stats[s] and "llm_exec_time" in fc_stats[s]]
        y = [fc_stats[s]["llm_exec_time"]["mean"] for s in fc_steps
             if "token_usage_prompt" in fc_stats[s] and "llm_exec_time" in fc_stats[s]]
        if x and y:
            ax.plot(x, y, **{k: v for k, v in PLOT_STYLE["full_context"].items() if k != "label"},
                    label="Full-Context", linewidth=2, markersize=5)
            valid_fc = [s for s in fc_steps
                        if "token_usage_prompt" in fc_stats[s] and "llm_exec_time" in fc_stats[s]]
            for i, s in enumerate(valid_fc):
                if i % max(1, len(valid_fc) // 8) == 0:
                    ax.annotate(f"s{s}", (x[i], y[i]), fontsize=7, alpha=0.6,
                                xytext=(5, 5), textcoords="offset points")

    ax.legend(fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"latency_vs_context{step_config}.png"), dpi=150)
    fig.savefig(os.path.join(output_dir, f"latency_vs_context{step_config}.pdf"))
    plt.close(fig)


def _plot_panel(ax, comem_stats, fc_stats, comem_steps, fc_steps, metric, ylabel):
    """Helper to plot a single metric panel with mean + p25/p75 fill."""
    ax.set_xlabel("Step Index", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)

    for stats, steps, color, marker, label in [
        (comem_stats, comem_steps, "#2196F3", "o", "COMEM"),
        (fc_stats, fc_steps, "#FF5722", "s", "Full-Context"),
    ]:
        valid = [s for s in steps if metric in stats[s] and stats[s][metric]["n"] > 0]
        if not valid:
            continue
        means = [stats[s][metric]["mean"] for s in valid]
        p25 = [stats[s][metric]["p25"] for s in valid]
        p75 = [stats[s][metric]["p75"] for s in valid]
        ax.plot(valid, means, color=color, marker=marker, label=label, linewidth=2, markersize=5)
        ax.fill_between(valid, p25, p75, alpha=0.15, color=color)

    ax.legend(fontsize=10)


def plot_combined_time_memory(
    comem_stats: Dict, fc_stats: Dict, output_dir: str, step_config: str = "",
):
    """Plot 4: Combined multi-panel figure (Time + Memory) for paper.

    Produces a 2-panel (Time + Prompt Tokens) version and, if vLLM KV cache
    data is available, also a 3-panel version adding GPU KV-cache usage.
    """
    comem_steps = sorted(comem_stats.keys())
    fc_steps = sorted(fc_stats.keys())

    has_kv = (
        _has_metric(comem_stats, comem_steps, "vllm_kv_cache_usage_perc")
        or _has_metric(fc_stats, fc_steps, "vllm_kv_cache_usage_perc")
    )
    has_pchr = (
        _has_metric(comem_stats, comem_steps, "vllm_prefix_cache_hit_rate")
        or _has_metric(fc_stats, fc_steps, "vllm_prefix_cache_hit_rate")
    )

    # Determine number of panels
    panels = ["time", "tokens"]
    if has_kv:
        panels.append("kv_cache")
    if has_pchr:
        panels.append("prefix_cache")
    n = len(panels)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    panel_idx = 0

    # Panel: Time
    ax = axes[panel_idx]
    ax.set_title(f"({chr(97 + panel_idx)}) Per-Step LLM Inference Time", fontsize=13, fontweight="bold")
    _plot_panel(ax, comem_stats, fc_stats, comem_steps, fc_steps, "llm_exec_time", "LLM Exec Time (s)")
    panel_idx += 1

    # Panel: Prompt Tokens
    ax = axes[panel_idx]
    ax.set_title(f"({chr(97 + panel_idx)}) Per-Step Prompt Tokens", fontsize=13, fontweight="bold")
    _plot_panel(ax, comem_stats, fc_stats, comem_steps, fc_steps, "token_usage_prompt", "Prompt Tokens")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
    panel_idx += 1

    # Panel: KV Cache Usage (if available)
    if has_kv:
        ax = axes[panel_idx]
        ax.set_title(f"({chr(97 + panel_idx)}) GPU KV-Cache Usage", fontsize=13, fontweight="bold")
        _plot_panel(ax, comem_stats, fc_stats, comem_steps, fc_steps, "vllm_kv_cache_usage_perc", "KV-Cache Usage")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        panel_idx += 1

    # Panel: Prefix Cache Hit Rate (if available)
    if has_pchr:
        ax = axes[panel_idx]
        ax.set_title(f"({chr(97 + panel_idx)}) Prefix Cache Hit Rate", fontsize=13, fontweight="bold")
        _plot_panel(ax, comem_stats, fc_stats, comem_steps, fc_steps, "vllm_prefix_cache_hit_rate", "Hit Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        panel_idx += 1

    fig.suptitle(
        f"COMEM vs Full-Context: Time and Memory Dimensions{' — ' + step_config if step_config else ''}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"combined_time_memory{step_config}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"combined_time_memory{step_config}.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary CSV / table
# ---------------------------------------------------------------------------

def write_summary_csv(
    comem_stats: Dict, fc_stats: Dict, output_dir: str, step_config: str = "",
):
    """Write a per-step CSV comparing COMEM vs full-context metrics."""
    import csv

    all_steps = sorted(set(comem_stats.keys()) | set(fc_stats.keys()))
    metrics = [
        "llm_exec_time", "summ_exec_time", "prefill_exec_time",
        "effective_llm_time", "total_step_time",
        "token_usage_prompt", "token_usage_completion", "token_usage_total",
        "token_usage_summary", "token_usage_summary_prompt",
        "vllm_kv_cache_usage_perc", "vllm_num_requests_running",
        "vllm_prefix_cache_hit_rate", "vllm_kv_reuse_ratio",
        "vllm_delta_prefix_cache_hits_total", "vllm_delta_prefix_cache_queries_total",
        "vllm_delta_num_preemptions_total",
    ]

    csv_path = os.path.join(output_dir, f"per_step_comparison{step_config}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = ["step_idx"]
        for m in metrics:
            header.extend([f"comem_{m}_mean", f"comem_{m}_std", f"fc_{m}_mean", f"fc_{m}_std"])
        writer.writerow(header)

        for s in all_steps:
            row = [s]
            for m in metrics:
                # COMEM
                if s in comem_stats and m in comem_stats[s]:
                    row.extend([
                        f"{comem_stats[s][m]['mean']:.4f}",
                        f"{comem_stats[s][m]['std']:.4f}",
                    ])
                else:
                    row.extend(["", ""])
                # Full-context
                if s in fc_stats and m in fc_stats[s]:
                    row.extend([
                        f"{fc_stats[s][m]['mean']:.4f}",
                        f"{fc_stats[s][m]['std']:.4f}",
                    ])
                else:
                    row.extend(["", ""])
            writer.writerow(row)

    print(f"  Summary CSV saved to: {csv_path}")
    return csv_path


def extract_traj_level_stats(trajs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute trajectory-level aggregate statistics across instances.

    For each trajectory: total wall time, total LLM time, total summary time,
    total prompt tokens, total completion tokens, num steps.

    Returns: {metric_name: {"mean": ..., "std": ..., "median": ..., "min": ..., "max": ..., "n": ...}}
    """
    if not trajs:
        return {}

    records = defaultdict(list)
    for traj in trajs:
        steps = traj.get("trajectory_steps", [])
        if not steps:
            continue

        n_steps = len(steps)
        total_llm = sum(s.get("llm_exec_time", 0) for s in steps)
        total_summ = sum(s.get("summ_exec_time", 0) for s in steps)
        total_step = sum(s.get("total_step_time", 0) for s in steps)
        total_prompt = sum(s.get("token_usage_prompt", 0) for s in steps)
        total_completion = sum(s.get("token_usage_completion", 0) for s in steps)
        total_summary_tokens = sum(s.get("token_usage_summary", 0) for s in steps)

        # Wall time from trajectory timestamps
        t_start = traj.get("traj_start_time")
        t_end = traj.get("traj_finish_time")
        wall_time = (t_end - t_start) if (t_start is not None and t_end is not None) else total_step

        records["num_steps"].append(n_steps)
        records["wall_time"].append(wall_time)
        records["total_llm_time"].append(total_llm)
        records["total_summ_time"].append(total_summ)
        records["total_step_time"].append(total_step)
        records["total_prompt_tokens"].append(total_prompt)
        records["total_completion_tokens"].append(total_completion)
        records["total_summary_tokens"].append(total_summary_tokens)
        records["avg_prompt_tokens_per_step"].append(total_prompt / n_steps if n_steps > 0 else 0)
        records["avg_llm_time_per_step"].append(total_llm / n_steps if n_steps > 0 else 0)

    stats = {}
    for key, values in records.items():
        arr = np.array(values)
        stats[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr),
        }
    return stats


def print_traj_summary(
    comem_trajs: List[Dict[str, Any]],
    fc_trajs: List[Dict[str, Any]],
    output_dir: str,
    step_config: str = "",
):
    """Print and save trajectory-level summary table."""
    comem_stats = extract_traj_level_stats(comem_trajs)
    fc_stats = extract_traj_level_stats(fc_trajs)

    if not comem_stats and not fc_stats:
        return

    print("\n" + "=" * 120)
    print(f"TRAJECTORY-LEVEL SUMMARY{' — ' + step_config if step_config else ''}")
    print("=" * 120)

    metrics_display = [
        ("num_steps",                 "Steps/Traj",          "{:.1f}"),
        ("wall_time",                 "Wall Time (s)",       "{:.2f}"),
        ("total_llm_time",            "Total LLM Time (s)",  "{:.2f}"),
        ("total_summ_time",           "Total Summ Time (s)", "{:.2f}"),
        ("total_step_time",           "Total Step Time (s)", "{:.2f}"),
        ("avg_llm_time_per_step",     "Avg LLM/Step (s)",    "{:.3f}"),
        ("total_prompt_tokens",       "Total Prompt Tok",    "{:.0f}"),
        ("total_completion_tokens",   "Total Compl Tok",     "{:.0f}"),
        ("total_summary_tokens",      "Total Summ Tok",      "{:.0f}"),
        ("avg_prompt_tokens_per_step","Avg Prompt/Step",      "{:.0f}"),
    ]

    header = f"{'Metric':<25} | {'COMEM (mean +/- std)':>25} | {'FC (mean +/- std)':>25} | {'Speedup':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for key, label, fmt in metrics_display:
        c = comem_stats.get(key, {})
        f = fc_stats.get(key, {})
        c_str = (fmt + " +/- " + fmt).format(c["mean"], c["std"]) if c else "N/A"
        f_str = (fmt + " +/- " + fmt).format(f["mean"], f["std"]) if f else "N/A"

        # Speedup for time metrics
        speedup_str = ""
        if key in ("wall_time", "total_llm_time", "total_step_time", "avg_llm_time_per_step"):
            if c and f and c["mean"] > 0:
                speedup_str = f"{f['mean'] / c['mean']:.2f}x"
        elif key in ("total_prompt_tokens", "avg_prompt_tokens_per_step"):
            if c and f and c["mean"] > 0:
                speedup_str = f"{f['mean'] / c['mean']:.2f}x"

        print(f"{label:<25} | {c_str:>25} | {f_str:>25} | {speedup_str:>10}")
        rows.append({"metric": label, "comem_mean": c.get("mean", ""), "comem_std": c.get("std", ""),
                      "fc_mean": f.get("mean", ""), "fc_std": f.get("std", "")})

    print("=" * 120)

    # Also save as CSV
    import csv
    csv_path = os.path.join(output_dir, f"traj_level_summary{step_config}.csv")
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["metric", "comem_mean", "comem_std", "fc_mean", "fc_std"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Trajectory summary CSV saved to: {csv_path}")


def print_crossover_analysis(comem_stats: Dict, fc_stats: Dict):
    """Print crossover analysis to stdout."""
    common_steps = sorted(set(comem_stats.keys()) & set(fc_stats.keys()))
    if not common_steps:
        print("  No overlapping steps between COMEM and Full-Context for crossover analysis.")
        return

    print("\n" + "=" * 80)
    print("CROSSOVER ANALYSIS")
    print("=" * 80)

    print(f"\n{'Step':>6} | {'COMEM LLM(s)':>14} | {'FC LLM(s)':>14} | {'Speedup':>10} | {'COMEM Prompt':>14} | {'FC Prompt':>14} | {'Token Ratio':>12}")
    print("-" * 100)

    crossover_step = None
    for s in common_steps:
        c_llm = comem_stats[s].get("llm_exec_time", {}).get("mean", float("nan"))
        f_llm = fc_stats[s].get("llm_exec_time", {}).get("mean", float("nan"))
        c_prompt = comem_stats[s].get("token_usage_prompt", {}).get("mean", float("nan"))
        f_prompt = fc_stats[s].get("token_usage_prompt", {}).get("mean", float("nan"))

        speedup = f_llm / c_llm if c_llm > 0 else float("nan")
        token_ratio = f_prompt / c_prompt if c_prompt > 0 else float("nan")

        marker = " <-- crossover" if crossover_step is None and speedup > 1.0 and s > common_steps[0] else ""
        if crossover_step is None and speedup > 1.0 and s > common_steps[0]:
            crossover_step = s

        print(f"{s:>6} | {c_llm:>14.3f} | {f_llm:>14.3f} | {speedup:>9.2f}x | {c_prompt:>14.0f} | {f_prompt:>14.0f} | {token_ratio:>11.2f}x{marker}")

    if crossover_step is not None:
        fc_tokens_at_cross = fc_stats[crossover_step].get("token_usage_prompt", {}).get("mean", 0)
        print(f"\n  Crossover at step {crossover_step} (~{fc_tokens_at_cross:.0f} full-context prompt tokens)")
        print(f"  COMEM becomes faster when full-context exceeds ~{fc_tokens_at_cross:.0f} tokens.")
    else:
        # Check if COMEM is always faster or always slower
        first_speedup = (
            fc_stats[common_steps[0]].get("llm_exec_time", {}).get("mean", 0)
            / max(comem_stats[common_steps[0]].get("llm_exec_time", {}).get("mean", 1), 1e-9)
        )
        if first_speedup > 1.0:
            print("\n  COMEM is faster than Full-Context at ALL measured steps.")
        else:
            print("\n  Full-Context is faster than COMEM at ALL measured steps.")
            print("  (COMEM overhead does not pay off within the measured step range.)")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Grid-level analysis (when multiple step configs exist)
# ---------------------------------------------------------------------------

def process_grid_directory(
    comem_dir: str, fc_dir: str, output_dir: str,
):
    """Process all JSONL files in grid benchmark directories.

    Matches COMEM and full-context files by step count (sN in filename).
    """
    comem_files = find_jsonl_files(comem_dir)
    fc_files = find_jsonl_files(fc_dir)

    if not comem_files and not fc_files:
        print(f"No .jsonl files found in {comem_dir} or {fc_dir}")
        return

    print(f"Found {len(comem_files)} COMEM files, {len(fc_files)} full-context files")

    # Extract step count from filename pattern like "..._s32_..."
    step_pattern = re.compile(r'_s(\d+)_')

    def get_step_count(filepath):
        match = step_pattern.search(os.path.basename(filepath))
        return int(match.group(1)) if match else None

    # Group files by step count
    comem_by_steps = {}
    for f in comem_files:
        sc = get_step_count(f)
        if sc is not None:
            comem_by_steps[sc] = f

    fc_by_steps = {}
    for f in fc_files:
        sc = get_step_count(f)
        if sc is not None:
            fc_by_steps[sc] = f

    all_step_counts = sorted(set(comem_by_steps.keys()) | set(fc_by_steps.keys()))
    print(f"Step counts found: {all_step_counts}")

    for sc in all_step_counts:
        config_label = f"_s{sc}"
        print(f"\n--- Processing step config: {sc} steps ---")

        comem_trajs = []
        fc_trajs = []

        if sc in comem_by_steps:
            print(f"  Loading COMEM: {os.path.basename(comem_by_steps[sc])}")
            comem_trajs = load_trajectories(comem_by_steps[sc])
            print(f"  Loaded {len(comem_trajs)} COMEM trajectories")

        if sc in fc_by_steps:
            print(f"  Loading FC: {os.path.basename(fc_by_steps[sc])}")
            fc_trajs = load_trajectories(fc_by_steps[sc])
            print(f"  Loaded {len(fc_trajs)} full-context trajectories")

        comem_by_step = aggregate_across_trajectories(comem_trajs) if comem_trajs else {}
        fc_by_step = aggregate_across_trajectories(fc_trajs) if fc_trajs else {}

        comem_stats = compute_step_stats(comem_by_step)
        fc_stats = compute_step_stats(fc_by_step)

        plot_time_comparison(comem_stats, fc_stats, output_dir, config_label)
        plot_memory_comparison(comem_stats, fc_stats, output_dir, config_label)
        plot_latency_vs_context(comem_stats, fc_stats, output_dir, config_label)
        plot_combined_time_memory(comem_stats, fc_stats, output_dir, config_label)
        write_summary_csv(comem_stats, fc_stats, output_dir, config_label)
        print_traj_summary(comem_trajs, fc_trajs, output_dir, config_label)
        print_crossover_analysis(comem_stats, fc_stats)

    # Also produce a multi-config overlay if there are multiple step counts
    if len(all_step_counts) > 1:
        plot_multi_config_summary(comem_by_steps, fc_by_steps, all_step_counts, output_dir)


def plot_multi_config_summary(
    comem_by_steps: Dict[int, str],
    fc_by_steps: Dict[int, str],
    step_counts: List[int],
    output_dir: str,
):
    """Plot summary across multiple step configs: avg latency vs step count."""
    comem_avg_latencies = []
    fc_avg_latencies = []
    comem_total_tokens = []
    fc_total_tokens = []

    for sc in step_counts:
        if sc in comem_by_steps:
            trajs = load_trajectories(comem_by_steps[sc])
            all_llm = [s.get("llm_exec_time", 0) for t in trajs for s in t.get("trajectory_steps", [])]
            all_prompt = [s.get("token_usage_prompt", 0) for t in trajs for s in t.get("trajectory_steps", [])]
            comem_avg_latencies.append(np.mean(all_llm) if all_llm else 0)
            comem_total_tokens.append(np.mean(all_prompt) if all_prompt else 0)
        else:
            comem_avg_latencies.append(None)
            comem_total_tokens.append(None)

        if sc in fc_by_steps:
            trajs = load_trajectories(fc_by_steps[sc])
            all_llm = [s.get("llm_exec_time", 0) for t in trajs for s in t.get("trajectory_steps", [])]
            all_prompt = [s.get("token_usage_prompt", 0) for t in trajs for s in t.get("trajectory_steps", [])]
            fc_avg_latencies.append(np.mean(all_llm) if all_llm else 0)
            fc_total_tokens.append(np.mean(all_prompt) if all_prompt else 0)
        else:
            fc_avg_latencies.append(None)
            fc_total_tokens.append(None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: avg LLM latency vs step count
    ax1.set_title("Avg LLM Latency vs Trajectory Length", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Steps per Trajectory", fontsize=12)
    ax1.set_ylabel("Avg LLM Exec Time (s)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    valid_comem = [(sc, lat) for sc, lat in zip(step_counts, comem_avg_latencies) if lat is not None]
    valid_fc = [(sc, lat) for sc, lat in zip(step_counts, fc_avg_latencies) if lat is not None]

    if valid_comem:
        ax1.plot([x[0] for x in valid_comem], [x[1] for x in valid_comem],
                 color="#2196F3", marker="o", label="COMEM", linewidth=2, markersize=7)
    if valid_fc:
        ax1.plot([x[0] for x in valid_fc], [x[1] for x in valid_fc],
                 color="#FF5722", marker="s", label="Full-Context", linewidth=2, markersize=7)
    ax1.legend(fontsize=11)

    # Right: avg prompt tokens vs step count
    ax2.set_title("Avg Prompt Tokens vs Trajectory Length", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Steps per Trajectory", fontsize=12)
    ax2.set_ylabel("Avg Prompt Tokens", fontsize=12)
    ax2.grid(True, alpha=0.3)

    valid_comem_t = [(sc, tok) for sc, tok in zip(step_counts, comem_total_tokens) if tok is not None]
    valid_fc_t = [(sc, tok) for sc, tok in zip(step_counts, fc_total_tokens) if tok is not None]

    if valid_comem_t:
        ax2.plot([x[0] for x in valid_comem_t], [x[1] for x in valid_comem_t],
                 color="#2196F3", marker="o", label="COMEM", linewidth=2, markersize=7)
    if valid_fc_t:
        ax2.plot([x[0] for x in valid_fc_t], [x[1] for x in valid_fc_t],
                 color="#FF5722", marker="s", label="Full-Context", linewidth=2, markersize=7)
    ax2.legend(fontsize=11)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))

    fig.suptitle("Multi-Config Summary: COMEM vs Full-Context", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "multi_config_summary.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "multi_config_summary.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Multi-config summary plot saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze COMEM vs Full-Context latency benchmark results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Directory mode (grid benchmark)
    parser.add_argument("--comem_dir", type=str, default=None,
                        help="Directory containing COMEM benchmark .jsonl files")
    parser.add_argument("--fc_dir", type=str, default=None,
                        help="Directory containing full-context benchmark .jsonl files")

    # Single-file mode
    parser.add_argument("--comem_jsonl", type=str, default=None,
                        help="Single COMEM benchmark .jsonl file")
    parser.add_argument("--fc_jsonl", type=str, default=None,
                        help="Single full-context benchmark .jsonl file")

    parser.add_argument("--output_dir", type=str, default="./analysis_results",
                        help="Directory to save plots and CSVs (default: ./analysis_results)")

    args = parser.parse_args()

    # Validate inputs
    has_dirs = args.comem_dir or args.fc_dir
    has_files = args.comem_jsonl or args.fc_jsonl

    if not has_dirs and not has_files:
        parser.error("Provide either --comem_dir/--fc_dir or --comem_jsonl/--fc_jsonl")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    if has_dirs:
        # Grid mode: process all files in directories
        process_grid_directory(
            comem_dir=args.comem_dir or "",
            fc_dir=args.fc_dir or "",
            output_dir=args.output_dir,
        )
    else:
        # Single file mode
        comem_trajs = []
        fc_trajs = []

        if args.comem_jsonl:
            print(f"Loading COMEM trajectories from: {args.comem_jsonl}")
            comem_trajs = load_trajectories(args.comem_jsonl)
            print(f"  Loaded {len(comem_trajs)} trajectories")

        if args.fc_jsonl:
            print(f"Loading full-context trajectories from: {args.fc_jsonl}")
            fc_trajs = load_trajectories(args.fc_jsonl)
            print(f"  Loaded {len(fc_trajs)} trajectories")

        comem_by_step = aggregate_across_trajectories(comem_trajs) if comem_trajs else {}
        fc_by_step = aggregate_across_trajectories(fc_trajs) if fc_trajs else {}

        comem_stats = compute_step_stats(comem_by_step)
        fc_stats = compute_step_stats(fc_by_step)

        plot_time_comparison(comem_stats, fc_stats, args.output_dir)
        plot_memory_comparison(comem_stats, fc_stats, args.output_dir)
        plot_latency_vs_context(comem_stats, fc_stats, args.output_dir)
        plot_combined_time_memory(comem_stats, fc_stats, args.output_dir)
        write_summary_csv(comem_stats, fc_stats, args.output_dir)
        print_traj_summary(comem_trajs, fc_trajs, args.output_dir)
        print_crossover_analysis(comem_stats, fc_stats)

    print(f"\nDone. All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
