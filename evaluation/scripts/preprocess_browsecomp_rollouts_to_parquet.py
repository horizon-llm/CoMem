import json
import os
import glob
import argparse
from typing import List, Dict, Any
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


# Fields to remove from trajectory steps to avoid parquet serialization issues
FIELDS_TO_REMOVE_FROM_STEP = ['vllm_metrics']


def clean_trajectory_steps(trajectory_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean trajectory steps by removing problematic fields."""
    cleaned_steps = []
    for step in trajectory_steps:
        step_copy = step.copy()
        for field in FIELDS_TO_REMOVE_FROM_STEP:
            step_copy.pop(field, None)
        if 'info' in step_copy and isinstance(step_copy['info'], dict):
            info_copy = step_copy['info'].copy()
            for field in FIELDS_TO_REMOVE_FROM_STEP:
                info_copy.pop(field, None)
            step_copy['info'] = info_copy
        cleaned_steps.append(step_copy)
    return cleaned_steps


def load_trajectory_files(input_path: str) -> List[Dict[str, Any]]:
    """Load trajectory JSON files from a directory or glob pattern.

    Supports:
      - A directory path: loads all task_*.json files in it
      - A glob pattern: loads matching files
      - A single JSONL file: loads records line by line
    """
    records = []

    if os.path.isdir(input_path):
        pattern = os.path.join(input_path, "task_*.json")
        files = sorted(glob.glob(pattern))
        print(f"Found {len(files)} trajectory files in {input_path}")
        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                records.append(json.load(f))
    elif input_path.endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"Loaded {len(records)} records from {input_path}")
    else:
        files = sorted(glob.glob(input_path))
        print(f"Found {len(files)} files matching pattern {input_path}")
        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                records.append(json.load(f))

    return records


def _save_parquet(df: pd.DataFrame, output_file: str, compression: str) -> None:
    """Save DataFrame to parquet and verify."""
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving to parquet: {output_file}")
    df.to_parquet(
        output_file,
        engine="pyarrow",
        compression=compression,
        index=False,
    )

    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)
    print(f"Saved {len(df)} records to {output_file}")
    print(f"File size: {file_size_mb:.2f} MB")

    print(f"Verifying parquet file...")
    try:
        df_verify = pd.read_parquet(output_file)
        print(f"Verification successful: {len(df_verify)} records")
    except Exception as e:
        print(f"Verification with pd.read_parquet failed: {e}")
        print("Attempting to identify problematic columns...")
        parquet_file = pq.ParquetFile(output_file)
        print(f"Parquet schema:\n{parquet_file.schema_arrow}")
        for col in df.columns:
            try:
                _ = pd.read_parquet(output_file, columns=[col])
                print(f"  Column '{col}': OK")
            except Exception as col_e:
                print(f"  Column '{col}': FAILED - {col_e}")
        raise


def preprocess_records(args, records: List[Dict[str, Any]]) -> None:
    """Preprocess browsecomp trajectory records and save to parquet."""
    processed = []
    for record in records:
        trajectory_steps = record.get("trajectory_steps", [])
        trajectory_steps = clean_trajectory_steps(trajectory_steps)

        if len(trajectory_steps) < args.min_steps + 1:
            continue

        task_id = record.get("env_args", {}).get("task_id", "")
        ground_truth = record.get("env_args", {}).get("ground_truth", "")

        end = len(trajectory_steps)
        if args.max_steps is not None:
            end = min(end, args.max_steps)

        for step_idx in range(args.min_steps - 1, end):
            cleaned_steps = trajectory_steps[: step_idx + 1]
            processed_record = {
                "docker_image": f"browsecomp/{task_id}",
                "trajectory_steps": cleaned_steps,
                "system_prompt": record.get("system_prompt", ""),
                "user_prompt": record.get("user_prompt", ""),
                "data_source": record.get("exp_name", "browsecomp"),
                "extra_info": {"tools_kwargs": {
                    "docker_image": f"browsecomp/{task_id}",
                    "trajectory_steps": cleaned_steps,
                    "system_prompt": record.get("system_prompt", ""),
                    "user_prompt": record.get("user_prompt", ""),
                    "problem_statement": record.get("problem_statement", ""),
                    "task_id": task_id,
                    "ground_truth": ground_truth,
                }},
            }
            processed.append(processed_record)

    print(f"Preprocessed {len(processed)} records from {len(records)} trajectories.")

    df = pd.DataFrame(processed)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)

    num_prompt_response_pairs = len(df)
    num_trajectories = df['docker_image'].nunique()
    print(f"\nStatistics:")
    print(f"  Number of trajectories: {num_trajectories}")
    print(f"  Number of prompt-response pairs: {num_prompt_response_pairs}")

    if args.eval_output_file is not None:
        if args.train_records is not None and args.eval_records is not None:
            total_requested = args.train_records + args.eval_records
            eval_proportion = args.eval_records / total_requested
        else:
            eval_proportion = 0.2

        print(f"\nSplitting data: {eval_proportion*100:.1f}% eval, {(1-eval_proportion)*100:.1f}% train")

        # Split by trajectory (task_id) to avoid data leakage
        unique_trajs = df["docker_image"].unique()
        train_trajs, eval_trajs = train_test_split(
            unique_trajs,
            test_size=eval_proportion,
            random_state=args.random_seed,
        )
        df_train = df[df["docker_image"].isin(train_trajs)].copy()
        df_eval = df[df["docker_image"].isin(eval_trajs)].copy()
        print(f"Split by trajectory: {len(train_trajs)} train, {len(eval_trajs)} eval")

        if args.train_records is not None and len(df_train) > args.train_records:
            df_train = df_train.sample(n=args.train_records, random_state=args.random_seed)
            print(f"Train set after sampling: {len(df_train)} records")

        if args.eval_records is not None and len(df_eval) > args.eval_records:
            df_eval = df_eval.sample(n=args.eval_records, random_state=args.random_seed)
            print(f"Eval set after sampling: {len(df_eval)} records")

        print(f"Train set: {len(df_train)} records")
        print(f"Eval set: {len(df_eval)} records")

        _save_parquet(df_train, args.output_file, args.compression)
        _save_parquet(df_eval, args.eval_output_file, args.compression)
    else:
        if args.train_records is not None and len(df) > args.train_records:
            df = df.sample(n=args.train_records, random_state=args.random_seed)
            print(f"Dataset after sampling: {len(df)} records")

        _save_parquet(df, args.output_file, args.compression)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess browsecomp trajectory files into parquet for GRPO training."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to directory with task_*.json files, a glob pattern, or a JSONL file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output parquet file",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=1,
        help="Minimum step number to include (default: 1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum step number to include (default: None for all)",
    )
    parser.add_argument(
        "--train-records",
        type=int,
        default=None,
        help="Maximum number of training records to include",
    )
    parser.add_argument(
        "--eval-records",
        type=int,
        default=None,
        help="Maximum number of evaluation records to include",
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
        help="Path to eval parquet file. If specified, data is split into train/eval.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for train/eval split (default: 42)",
    )

    args = parser.parse_args()

    records = load_trajectory_files(args.input_path)
    if not records:
        print("No records found. Exiting.")
        return

    preprocess_records(args, records)
    print("Preprocessing completed.")


if __name__ == "__main__":
    main()
