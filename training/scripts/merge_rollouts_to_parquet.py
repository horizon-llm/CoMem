#!/usr/bin/env python3
# Copyright 2025
# Merge all generated rollout JSONL files into a single parquet file for GRPO training.
#
# Usage examples:
#   1. Merge all rollouts from a directory:
#      python merge_rollouts_to_parquet.py --input-dir outputs/my_rollouts --output-file data/rollouts.parquet
#
#   2. Merge specific iteration range:
#      python merge_rollouts_to_parquet.py --input-dir outputs/my_rollouts --start-iter 0 --end-iter 10 --output-file data/rollouts_0_10.parquet
#
#   3. Filter only active/valid steps:
#      python merge_rollouts_to_parquet.py --input-dir outputs/my_rollouts --filter-active --output-file data/rollouts_active.parquet
#
#   4. Split data into train/eval sets:
#      python merge_rollouts_to_parquet.py --input-dir outputs/my_rollouts --output-file data/rollouts_train.parquet --eval-output-file data/rollouts_eval.parquet
#
#   5. Filter steps beyond a maximum:
#      python merge_rollouts_to_parquet.py --input-dir outputs/my_rollouts --max-steps 10 --output-file data/rollouts_max10.parquet
#
#   6. Filter steps with min/max range (default min_steps=1 excludes step 0):
#      python merge_rollouts_to_parquet.py --input-dir outputs/my_rollouts --min-steps 2 --max-steps 10 --output-file data/rollouts_2_10.parquet
#
#   7. Specify exact number of train and eval records:
#      python merge_rollouts_to_parquet.py --input-dir outputs/my_rollouts --train-records 8000 --eval-records 2000 --eval-output-file data/rollouts_eval.parquet --output-file data/rollouts_train.parquet

import re
import json
import os
import json
import glob
import argparse
import random
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

def parse_turns(raw: str) -> List[Dict[str, str]]:
    """
    Return a list of {"role": "user"|"assistant", "content": "<block>"}.
    - Tolerates inputs wrapped like: `"prompt": "user\\n...\\nassistant\\n..."`
    - Splits on lines that are exactly `user` or `assistant` (case-insensitive).
    """
    text = raw

    # If wrapped in a JSON string field "prompt": "<escaped>", unescape it
    m = re.search(r'"prompt"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.S)
    if m:
        try:
            text = json.loads(f'"{m.group(1)}"')
        except Exception:
            text = m.group(1).replace("\\n", "\n")

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Split into [preamble, role1, block1, role2, block2, ...]
    parts = re.split(r'(?mi)^\s*(user|assistant)\s*$', text)
    if len(parts) < 3:
        return []

    turns = []
    # Skip parts[0] (preamble before first role marker)
    for i in range(1, len(parts) - 1, 2):
        role = parts[i].strip().lower()
        block = parts[i + 1].strip()
        if role in ("user", "assistant"):
            turns.append({"role": role, "content": block})
    if turns[-1]["content"] == "": # Remove trailing empty block
        turns = turns[:-1]
    return turns

def parse_conv_strings(raw: str) -> List[Dict[str, str]]:
    """
    Parse conversation strings in the format "role: content\\nrole: content".
    This format is used by conv_strings field from multi_agent_rollout_loop_main_api_mem.py.

    Returns a list of {"role": "user"|"assistant", "content": "<block>"}.
    """
    text = raw.strip()
    if not text:
        return []

    turns = []
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match "role: content" pattern
        match = re.match(r'^\s*(user|assistant)\s*:\s*(.*)$', line, flags=re.IGNORECASE)
        if match:
            role = match.group(1).lower()
            content = match.group(2).strip()
            turns.append({"role": role, "content": content})

    return turns

def find_rollout_files(input_dir: str, start_iter: Optional[int] = None, end_iter: Optional[int] = None) -> List[str]:
    """
    Find all rollout JSONL files in the input directory.

    Args:
        input_dir: Directory containing rollout JSONL files
        start_iter: Starting iteration (inclusive), None for no lower bound
        end_iter: Ending iteration (exclusive), None for no upper bound

    Returns:
        List of file paths sorted by iteration number
    """
    pattern = os.path.join(input_dir, "rollouts_*.jsonl")
    all_files = glob.glob(pattern)

    # Extract iteration numbers and filter
    files_with_iters = []
    for f in all_files:
        basename = os.path.basename(f)
        try:
            iter_num = int(basename.split("_")[1].split(".")[0])
            if start_iter is not None and iter_num < start_iter:
                continue
            if end_iter is not None and iter_num >= end_iter:
                continue
            files_with_iters.append((iter_num, f))
        except (IndexError, ValueError):
            print(f"Warning: Skipping file with unexpected name format: {f}")
            continue

    # Sort by iteration number
    files_with_iters.sort(key=lambda x: x[0])
    return [f for _, f in files_with_iters]


def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of records."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def merge_rollouts(
    input_dir: str,
    output_file: str,
    start_iter: Optional[int] = None,
    end_iter: Optional[int] = None,
    filter_active: bool = False,
    filter_valid: bool = False,
    min_steps: int = 1,
    max_steps: Optional[int] = None,
    train_records: Optional[int] = None,
    eval_records: Optional[int] = None,
    compression: str = "snappy",
    eval_output_file: Optional[str] = None,
    random_seed: int = 42,
    data_source: Optional[str] = None,
) -> None:
    """
    Merge all rollout JSONL files into a single parquet file.

    Args:
        input_dir: Directory containing rollout JSONL files
        output_file: Path to output parquet file (for training data if eval split is used)
        start_iter: Starting iteration (inclusive)
        end_iter: Ending iteration (exclusive)
        filter_active: If True, only include rows where active=True
        filter_valid: If True, only include rows where is_action_valid=True
        min_steps: Minimum step number to include (default: 1, which excludes step 0)
        max_steps: If specified, only include steps up to max_steps (steps > max_steps are excluded)
        train_records: If specified, randomly sample up to this many records for training set
        eval_records: If specified, randomly sample up to this many records for evaluation set
        compression: Compression codec for parquet (snappy, gzip, brotli, etc.)
        eval_output_file: Path to output parquet file for evaluation data. If specified, data will be split into train/eval sets.
        random_seed: Random seed for train/eval split and sampling (default: 42)
        data_source: Optional label to add as data_source column to all records
    """
    # Find all rollout files
    rollout_files = find_rollout_files(input_dir, start_iter, end_iter)

    if not rollout_files:
        raise ValueError(f"No rollout files found in {input_dir}")

    print(f"Found {len(rollout_files)} rollout files")
    print(f"Iteration range: {os.path.basename(rollout_files[0])} to {os.path.basename(rollout_files[-1])}")

    # Load all records
    all_records = []
    for i, filepath in enumerate(rollout_files):
        print(f"Loading {i+1}/{len(rollout_files)}: {os.path.basename(filepath)}", end="\r")
        records = load_jsonl_file(filepath)

        # Add iteration number to each record for reference
        iter_num = int(os.path.basename(filepath).split("_")[1].split(".")[0])
        for record in records:
            record["iteration"] = iter_num

        all_records.extend(records)

    print(f"\nLoaded {len(all_records)} total records")

    # Parse turns for all records
    print("Parsing user/assistant turns from prompts...")
    parse_source_counts = {"conv_strings": 0, "prompt": 0, "none": 0}
    for record in all_records:
        if "conv_strings" in record and record["conv_strings"]:
            record["turns"] = parse_conv_strings(record["conv_strings"])
            parse_source_counts["conv_strings"] += 1
        elif "prompt" in record and record["prompt"]:
            record["turns"] = parse_turns(record["prompt"])
            parse_source_counts["prompt"] += 1
            # Validate turn count for prompt-based parsing
            if "step" in record:
                assert len(record["turns"]) == record["step"] * 2 + 1, f"Record step {record['step']} but got {len(record['turns'])} turns: {record['turns']}"
        else:
            record["turns"] = []
            parse_source_counts["none"] += 1

    print(f"Parsed turns for {len(all_records)} records:")
    print(f"  - From 'conv_strings': {parse_source_counts['conv_strings']}")
    print(f"  - From 'prompt': {parse_source_counts['prompt']}")
    print(f"  - No source found: {parse_source_counts['none']}")

    # Add data source column if specified
    for record in all_records:
        record["data_source"] = data_source if data_source is not None else "unknown"
    print(f"Added data_source='{data_source}' to all records")

    # Apply filters if requested
    if filter_active:
        all_records = [r for r in all_records if r.get("active", True)]
        print(f"After filtering active=True: {len(all_records)} records")

    if filter_valid:
        all_records = [r for r in all_records if r.get("is_action_valid", True)]
        print(f"After filtering is_action_valid=True: {len(all_records)} records")

    # Filter by step range
    all_records = [r for r in all_records if r.get("step", 0) >= min_steps]
    print(f"After filtering step >= {min_steps}: {len(all_records)} records")

    if max_steps is not None:
        all_records = [r for r in all_records if r.get("step", 0) <= max_steps]
        print(f"After filtering step <= {max_steps}: {len(all_records)} records")

    if not all_records:
        raise ValueError("No records remaining after filtering")

    # Convert to DataFrame
    print("Converting to DataFrame...")
    df = pd.DataFrame(all_records)

    # Print schema info
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)

    # Print statistics
    print(f"\nStatistics:")
    num_prompt_response_pairs = len(df)
    num_trajectories = df["traj_uid"].nunique() if "traj_uid" in df.columns else "N/A"
    print(f"  Number of trajectories: {num_trajectories}")
    print(f"  Number of prompt-response pairs: {num_prompt_response_pairs}")

    # Split into train/eval if requested
    if eval_output_file is not None:
        # Calculate eval proportion from requested records if both are specified
        if train_records is not None and eval_records is not None:
            total_requested = train_records + eval_records
            eval_proportion = eval_records / total_requested
            print(f"\nCalculated eval proportion from requested records: {eval_proportion:.3f} ({eval_records}/{total_requested})")
        else:
            # Default split: 80% train, 20% eval
            eval_proportion = 0.2
            print(f"\nUsing default eval proportion: {eval_proportion:.3f}")

        print(f"Splitting data: {eval_proportion*100:.1f}% for eval, {(1-eval_proportion)*100:.1f}% for train")
        print(f"Random seed: {random_seed}")

        # Split by trajectory to avoid data leakage
        if "traj_uid" in df.columns:
            unique_trajs = df["traj_uid"].unique()
            train_trajs, eval_trajs = train_test_split(
                unique_trajs,
                test_size=eval_proportion,
                random_state=random_seed
            )
            df_train = df[df["traj_uid"].isin(train_trajs)].copy()
            df_eval = df[df["traj_uid"].isin(eval_trajs)].copy()
            print(f"Split by trajectory: {len(train_trajs)} train trajectories, {len(eval_trajs)} eval trajectories")
        else:
            # If no trajectory ID, split by records
            df_train, df_eval = train_test_split(
                df,
                test_size=eval_proportion,
                random_state=random_seed
            )
            print(f"Split by records (no traj_uid found)")

        print(f"Train set: {len(df_train)} records")
        print(f"Eval set: {len(df_eval)} records")

        # Sample train records if specified
        if train_records is not None:
            if len(df_train) > train_records:
                print(f"Randomly sampling {train_records} train records from {len(df_train)} records (seed: {random_seed})")
                df_train = df_train.sample(n=train_records, random_state=random_seed)
                print(f"Train set after sampling: {len(df_train)} records")
            elif len(df_train) < train_records:
                print(f"Warning: Requested {train_records} train records but only {len(df_train)} available")

        # Sample eval records if specified
        if eval_records is not None:
            if len(df_eval) > eval_records:
                print(f"Randomly sampling {eval_records} eval records from {len(df_eval)} records (seed: {random_seed})")
                df_eval = df_eval.sample(n=eval_records, random_state=random_seed)
                print(f"Eval set after sampling: {len(df_eval)} records")
            elif len(df_eval) < eval_records:
                print(f"Warning: Requested {eval_records} eval records but only {len(df_eval)} available")

        # Save train set
        _save_parquet(df_train, output_file, compression)

        # Save eval set
        _save_parquet(df_eval, eval_output_file, compression)
    else:
        # No split, save all data to output_file
        _save_parquet(df, output_file, compression)


def _save_parquet(df: pd.DataFrame, output_file: str, compression: str) -> None:
    """Helper function to save DataFrame to parquet and verify."""
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to parquet
    print(f"\nSaving to parquet: {output_file}")
    df.to_parquet(
        output_file,
        engine="pyarrow",
        compression=compression,
        index=False,
    )

    # Print file size
    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)
    print(f"Saved {len(df)} records to {output_file}")
    print(f"File size: {file_size_mb:.2f} MB")

    # Verify the file can be read back
    print(f"Verifying parquet file...")
    df_verify = pd.read_parquet(output_file)
    print(f"Verification successful: {len(df_verify)} records")


def main():
    parser = argparse.ArgumentParser(
        description="Merge rollout JSONL files into a single parquet file for GRPO training."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing rollout JSONL files (e.g., outputs/my_rollouts)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output parquet file (e.g., data/rollouts.parquet)",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=None,
        help="Starting iteration (inclusive). If not specified, starts from the first file.",
    )
    parser.add_argument(
        "--end-iter",
        type=int,
        default=None,
        help="Ending iteration (exclusive). If not specified, includes all files.",
    )
    parser.add_argument(
        "--filter-active",
        action="store_true",
        help="Only include records where active=True",
    )
    parser.add_argument(
        "--filter-valid",
        action="store_true",
        help="Only include records where is_action_valid=True",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="Minimum step number to include (default: 1, which excludes step 0)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum step number to include. Steps beyond this value will be filtered out.",
    )
    parser.add_argument(
        "--train-records",
        type=int,
        default=None,
        help="Maximum number of training records to include. If more records exist in the train set, randomly samples this many records.",
    )
    parser.add_argument(
        "--eval-records",
        type=int,
        default=None,
        help="Maximum number of evaluation records to include. If more records exist in the eval set, randomly samples this many records.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=["snappy", "gzip", "brotli", "lz4", "zstd", "none"],
        help="Compression codec for parquet file (default: snappy)",
    )
    parser.add_argument(
        "--eval-output-file",
        type=str,
        default=None,
        help="Path to output parquet file for evaluation data. If specified, data will be split into train/eval sets.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for train/eval split (default: 42)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default=None,
        help="Add data source column with this value to all records.",
    )

    args = parser.parse_args()

    merge_rollouts(
        input_dir=args.input_dir,
        output_file=args.output_file,
        start_iter=args.start_iter,
        end_iter=args.end_iter,
        filter_active=args.filter_active,
        filter_valid=args.filter_valid,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        train_records=args.train_records,
        eval_records=args.eval_records,
        compression=args.compression,
        eval_output_file=args.eval_output_file,
        random_seed=args.random_seed,
        data_source=args.data_source,
    )

    print("\nMerge complete!")


if __name__ == "__main__":
    main()
