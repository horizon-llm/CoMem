import re
import json
import os
import json
import glob
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

error_string = "Error executing command in pod: ApiException()"

# Fields to remove from trajectory steps to avoid parquet serialization issues
# vllm_metrics is a very long string that causes "Nested data conversions not implemented
# for chunked array outputs" error when reading parquet files with pd.read_parquet()
FIELDS_TO_REMOVE_FROM_STEP = ['vllm_metrics']


def clean_trajectory_steps(trajectory_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean trajectory steps by removing problematic fields.

    This prevents PyArrow serialization issues with nested data structures
    that have inconsistent schemas across records.
    """
    cleaned_steps = []
    for step in trajectory_steps:
        step_copy = step.copy()
        # Remove problematic fields from step level
        for field in FIELDS_TO_REMOVE_FROM_STEP:
            step_copy.pop(field, None)
        # Also remove from info dict if present
        if 'info' in step_copy and isinstance(step_copy['info'], dict):
            info_copy = step_copy['info'].copy()
            for field in FIELDS_TO_REMOVE_FROM_STEP:
                info_copy.pop(field, None)
            step_copy['info'] = info_copy
        cleaned_steps.append(step_copy)
    return cleaned_steps

def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of records."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def apply_oversampling_to_df(
    df: pd.DataFrame,
    threshold: int,
    target_ratio: float,
    target_size: int,
    random_seed: int,
    set_name: str = "dataset"
) -> pd.DataFrame:
    """
    Oversample records with step count > threshold to achieve target_ratio while maintaining target_size.

    This ensures all records remain unique (no duplication). The function intelligently samples
    short and long records to achieve both the target ratio and target size.

    Args:
        df: DataFrame with processed records
        threshold: Step count threshold (e.g., 40)
        target_ratio: Target proportion of long records in final dataset (e.g., 0.5 for 50%)
        target_size: Desired total number of records in final dataset
        random_seed: Random seed for reproducibility
        set_name: Name of the dataset (for logging)

    Returns:
        DataFrame with oversampling applied
    """
    # Add step count column
    df['_step_count'] = df['trajectory_steps'].apply(len)

    # Separate short and long records
    df_short = df[df['_step_count'] <= threshold].copy()
    df_long = df[df['_step_count'] > threshold].copy()

    num_short = len(df_short)
    num_long = len(df_long)
    total = num_short + num_long

    print(f"\nOversampling statistics for {set_name}:")
    print(f"  Records with ≤{threshold} steps: {num_short}")
    print(f"  Records with >{threshold} steps: {num_long}")
    print(f"  Original total: {total}")
    print(f"  Original ratio of long records: {num_long/total:.3f}" if total > 0 else "  N/A")

    if num_long == 0:
        print(f"  Warning: No records with >{threshold} steps found. Returning first {target_size} records.")
        result = df.sample(n=min(target_size, len(df)), random_state=random_seed)
        return result.drop(columns=['_step_count'])

    if target_ratio >= 1.0 or target_ratio <= 0:
        print(f"  Warning: target_ratio must be between 0 and 1, got {target_ratio}. Skipping oversampling.")
        result = df.sample(n=min(target_size, len(df)), random_state=random_seed)
        return result.drop(columns=['_step_count'])

    # Calculate how many long and short records we need
    # target_ratio = num_long_needed / target_size
    # num_long_needed = target_ratio * target_size
    # num_short_needed = (1 - target_ratio) * target_size

    num_long_needed = int(target_ratio * target_size)
    num_short_needed = target_size - num_long_needed

    print(f"  Target total size: {target_size}")
    print(f"  Target ratio: {target_ratio:.3f}")
    print(f"  Long records needed: {num_long_needed}")
    print(f"  Short records needed: {num_short_needed}")

    # Sample long records (prefer all if available)
    if num_long >= num_long_needed:
        df_long_sampled = df_long.sample(n=num_long_needed, random_state=random_seed)
        print(f"  Sampling {num_long_needed} from {num_long} long records")
    else:
        df_long_sampled = df_long
        print(f"  Warning: Only {num_long} long records available, needed {num_long_needed}")
        # Adjust short records to fill the gap
        num_short_needed = target_size - num_long
        print(f"  Adjusted short records needed: {num_short_needed}")

    # Sample short records
    if num_short >= num_short_needed:
        df_short_sampled = df_short.sample(n=num_short_needed, random_state=random_seed)
        print(f"  Sampling {num_short_needed} from {num_short} short records")
    else:
        df_short_sampled = df_short
        print(f"  Warning: Only {num_short} short records available, needed {num_short_needed}")

    # Combine and shuffle
    result = pd.concat([df_short_sampled, df_long_sampled], ignore_index=True)
    result = result.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    actual_long = len(df_long_sampled)
    actual_total = len(result)
    print(f"  Final dataset size: {actual_total} records")
    print(f"  Final ratio of long records: {actual_long/actual_total:.3f}")

    # Remove temporary column
    result = result.drop(columns=['_step_count'])

    return result

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
    try:
        df_verify = pd.read_parquet(output_file)
        print(f"Verification successful: {len(df_verify)} records")
    except Exception as e:
        print(f"Verification with pd.read_parquet failed: {e}")
        print("Attempting to identify problematic columns...")
        # Read parquet schema to see column types
        parquet_file = pq.ParquetFile(output_file)
        print(f"Parquet schema:\n{parquet_file.schema_arrow}")
        # Try reading columns one by one
        for col in df.columns:
            try:
                _ = pd.read_parquet(output_file, columns=[col])
                print(f"  Column '{col}': OK")
            except Exception as col_e:
                print(f"  Column '{col}': FAILED - {col_e}")
                # For nested columns, try to identify which keys vary
                if col in ['trajectory_steps', 'extra_info']:
                    print(f"    Analyzing nested structure of '{col}'...")
                    sample_keys = set()
                    for idx, val in df[col].head(10).items():
                        if isinstance(val, list) and len(val) > 0:
                            for item in val[:3]:
                                if isinstance(item, dict):
                                    sample_keys.update(item.keys())
                                    if 'info' in item and isinstance(item['info'], dict):
                                        print(f"      Row {idx} step info keys: {list(item['info'].keys())}")
                        elif isinstance(val, dict):
                            sample_keys.update(val.keys())
                    print(f"    Sample keys found: {sample_keys}")
        raise

def preprocess_records(args, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preprocess records to extract relevant fields."""
    processed = []
    for record in records:
        trajectory_steps = record.get("trajectory_steps", [])
        # Clean trajectory steps to remove problematic fields (e.g., vllm_metrics)
        trajectory_steps = clean_trajectory_steps(trajectory_steps)
        for step in trajectory_steps:
            if error_string in step.get('observation', ''):
                continue
        if len(trajectory_steps) < args.min_steps+1:
            continue  # Skip records with insufficient steps
        for step_idx in range(args.min_steps - 1, min(len(trajectory_steps), args.max_steps) if args.max_steps is not None else len(trajectory_steps)):
            cleaned_steps = trajectory_steps[: step_idx + 1]
            processed_record = {
                "docker_image": record.get("docker_image", ""),
                "trajectory_steps": cleaned_steps,
                "system_prompt": record.get("system_prompt", ""),
                "user_prompt": record.get("user_prompt", ""),
                "data_source": record.get("exp_name", ""),
                "extra_info": {"tools_kwargs": {
                    "docker_image": record.get("docker_image", ""),
                    "trajectory_steps": cleaned_steps,
                    "system_prompt": record.get("system_prompt", ""),
                    "user_prompt": record.get("user_prompt", ""),
                    "problem_statement": record.get("problem_statement", ""),
                    }},
            }
            processed.append(processed_record)

    print(f"Preprocessed {len(processed)} records from {len(records)} trajectories.")

    print("Converting to DataFrame...")
    df = pd.DataFrame(processed)

    # Print schema info
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)

    # Print statistics
    print(f"\nStatistics:")
    num_prompt_response_pairs = len(df)
    num_trajectories = df['docker_image'].nunique()
    print(f"  Number of trajectories: {num_trajectories}")
    print(f"  Number of prompt-response pairs: {num_prompt_response_pairs}")

    # Split into train/eval if requested
    if args.eval_output_file is not None:
        # Calculate eval proportion from requested records if both are specified
        if args.train_records is not None and args.eval_records is not None:
            total_requested = args.train_records + args.eval_records
            eval_proportion = args.eval_records / total_requested
            print(f"Calculated eval proportion from requested records: {eval_proportion:.3f} (train: {args.train_records}, eval: {args.eval_records})")
        else:
            # Default split: 80% train, 20% eval
            eval_proportion = 0.2
            print(f"\nUsing default eval proportion: {eval_proportion:.3f}")
        
        print(f"Splitting data: {eval_proportion*100:.1f}% for eval, {(1-eval_proportion)*100:.1f}% for train")
        print(f"Random seed: {args.random_seed}")

        # Split by trajectory to avoid data leakage
        if "docker_image" in df.columns:
            unique_trajs = df["docker_image"].unique()
            train_trajs, eval_trajs = train_test_split(
                unique_trajs,
                test_size=eval_proportion,
                random_state=args.random_seed,
            )
            df_train = df[df["docker_image"].isin(train_trajs)].copy()
            df_eval = df[df["docker_image"].isin(eval_trajs)].copy()
            print(f"Split by trajectory: {len(train_trajs)} train trajectories, {len(eval_trajs)} eval trajectories")
        else:
            # If no trajectory ID, split by records
            df_train, df_eval = train_test_split(
                df,
                test_size=eval_proportion,
                random_state=args.random_seed,
            )
            print(f"Split by records: {len(df_train)} train records, {len(df_eval)} eval records")
        
        print(f"Train set: {len(df_train)} records")
        print(f"Eval set: {len(df_eval)} records")

        # Sample train records if specified, with optional oversampling
        if args.train_records is not None:
            if args.oversample_threshold is not None and args.oversample_ratio is not None:
                # Apply oversampling while respecting target size
                df_train = apply_oversampling_to_df(
                    df_train,
                    args.oversample_threshold,
                    args.oversample_ratio,
                    args.train_records,
                    args.random_seed,
                    "train set"
                )
            else:
                # Regular sampling without oversampling
                if len(df_train) > args.train_records:
                    print(f"Randomly sampling {args.train_records} train records from {len(df_train)} records (seed: {args.random_seed})")
                    df_train = df_train.sample(n=args.train_records, random_state=args.random_seed)
                    print(f"Train set after sampling: {len(df_train)} records")
                elif len(df_train) < args.train_records:
                    print(f"Warning: Requested {args.train_records} train records but only {len(df_train)} available")
        elif args.oversample_threshold is not None and args.oversample_ratio is not None:
            # Oversampling requested but no target size specified
            print(f"Warning: Oversampling requires --train-records to be specified. Skipping oversampling for train set.")

        # Sample eval records if specified, with optional oversampling
        if args.eval_records is not None:
            if args.oversample_threshold is not None and args.oversample_ratio is not None:
                # Apply oversampling while respecting target size
                df_eval = apply_oversampling_to_df(
                    df_eval,
                    args.oversample_threshold,
                    args.oversample_ratio,
                    args.eval_records,
                    args.random_seed,
                    "eval set"
                )
            else:
                # Regular sampling without oversampling
                if len(df_eval) > args.eval_records:
                    print(f"Randomly sampling {args.eval_records} eval records from {len(df_eval)} records (seed: {args.random_seed})")
                    df_eval = df_eval.sample(n=args.eval_records, random_state=args.random_seed)
                    print(f"Eval set after sampling: {len(df_eval)} records")
                elif len(df_eval) < args.eval_records:
                    print(f"Warning: Requested {args.eval_records} eval records but only {len(df_eval)} available")
        elif args.oversample_threshold is not None and args.oversample_ratio is not None:
            # Oversampling requested but no target size specified
            print(f"Warning: Oversampling requires --eval-records to be specified. Skipping oversampling for eval set.")
        
        # Save train set
        _save_parquet(df_train, args.output_file, args.compression)

        # Save eval set
        _save_parquet(df_eval, args.eval_output_file, args.compression)
    else:
        # No split, save all data to output_file
        # Apply oversampling if train_records is specified (used as target size)
        if args.train_records is not None and args.oversample_threshold is not None and args.oversample_ratio is not None:
            df = apply_oversampling_to_df(
                df,
                args.oversample_threshold,
                args.oversample_ratio,
                args.train_records,
                args.random_seed,
                "dataset"
            )
        elif args.train_records is not None:
            # Regular sampling without oversampling
            if len(df) > args.train_records:
                print(f"Randomly sampling {args.train_records} records from {len(df)} records (seed: {args.random_seed})")
                df = df.sample(n=args.train_records, random_state=args.random_seed)
                print(f"Dataset after sampling: {len(df)} records")
            elif len(df) < args.train_records:
                print(f"Warning: Requested {args.train_records} records but only {len(df)} available")

        _save_parquet(df, args.output_file, args.compression)

def main():
    parser = argparse.ArgumentParser(
        description="Merge rollout JSONL files into a single parquet file for GRPO training."
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to input JSONL file or glob pattern (e.g., data/rollouts_*.jsonl)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output parquet file (e.g., data/rollouts.parquet)",
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
        help="Maximum step number to include (default: None, which includes all steps)",
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
    parser.add_argument(
        "--oversample-threshold",
        type=int,
        default=None,
        help="Step count threshold for oversampling (e.g., 40). Records with more than this many steps will be oversampled.",
    )
    parser.add_argument(
        "--oversample-ratio",
        type=float,
        default=None,
        help="Target ratio of long records in the final dataset (e.g., 0.5 for 50%). Must be between 0 and 1.",
    )

    args = parser.parse_args()

    # Validate oversampling arguments
    if (args.oversample_threshold is None) != (args.oversample_ratio is None):
        parser.error("Both --oversample-threshold and --oversample-ratio must be specified together")

    if args.oversample_ratio is not None:
        if args.oversample_ratio <= 0 or args.oversample_ratio >= 1:
            parser.error("--oversample-ratio must be between 0 and 1")

        # Warn if oversampling is specified without record counts
        if args.train_records is None and args.eval_records is None:
            print("Warning: Oversampling parameters specified but no --train-records or --eval-records specified.")
            print("Oversampling will be skipped. To use oversampling, specify target record counts.")

    all_records = load_jsonl_file(args.input_jsonl)
    preprocess_records(args, all_records)

    print("Preprocessing completed.")


if __name__ == "__main__":
    main()