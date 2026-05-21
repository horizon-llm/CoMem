#!/usr/bin/env python3
# Copyright 2025
# Standalone test/validation script: runs validation on test data and saves results without training.
#
# Usage examples:
#   1. Test a checkpoint on validation data:
#      python generate_env_test.py --checkpoint checkpoints/my_exp/my_run/global_step_100 --output-dir outputs/test_results
#
#   2. Test with custom config overrides:
#      python generate_env_test.py --checkpoint checkpoints/my_exp/my_run/global_step_100 \
#          --overrides data.val_batch_size=64 env.response_agent.temperature=0.0

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List

import ray
import torch
import numpy as np
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl.utils.fs import copy_to_local
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.model import compute_position_id_with_mask
from verl.utils.device import is_cuda_available
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl import DataProto

# Env + rollout utilities
from agent_system.environments import make_envs, make_conv_envs
from agent_system.multi_turn_rollout import TrajectoryCollector, ConvTrajectoryCollector, MultiAgentTrajectoryCollector, MainAPIMemTrajectoryCollector, MainAPIRespAgentTrajectoryCollector, SumAPIMemTrajectoryCollector, SumAPIRespTrajectoryCollector, InterleavedAPIMemTrajectoryCollector, SumAPIRespRecentLiteLLMTrajectoryCollector

# Data loading
from verl.utils.dataset import RLHFDataset
from verl.utils.dataset.rl_dataset import collate_fn
from torch.utils.data import DataLoader


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _default_output_dir(base: str = "outputs/test_results") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, stamp)


def _collector_from_config(config, tokenizer, processor):
    traj_collection_type = config.env.get("traj_collection_type", "sum")
    print(f"[Test] Using trajectory collector type: {traj_collection_type}")
    if traj_collection_type == "mainapimem":
        return MainAPIMemTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    elif traj_collection_type == "interleavedapimem":
        return InterleavedAPIMemTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    elif traj_collection_type == "mainapiresp":
        return MainAPIRespAgentTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    elif traj_collection_type == "multi-agent":
        return MultiAgentTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    elif traj_collection_type == "sumapimem":
        return SumAPIMemTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    elif traj_collection_type == "sumapiresp":
        return SumAPIRespTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    elif traj_collection_type == "conv":
        return ConvTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    elif  traj_collection_type == "sumapiresprecentlitellm":
        return SumAPIRespRecentLiteLLMTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    else:
        return TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)


def _make_envs_from_config(config):
    env_manager_type = config.env.get("env_manager_type", "vanilla")
    if env_manager_type == "conv":
        return make_conv_envs(config)
    return make_envs(config)


def _save_config_snapshot(config, output_dir: str):
    """Dump the fully resolved config so test logs always capture parameters."""
    try:
        cfg_yaml = OmegaConf.to_yaml(config, resolve=True)
    except Exception:
        cfg_yaml = OmegaConf.to_yaml(config)
    snapshot_path = os.path.join(output_dir, "resolved_config.yaml")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        f.write(cfg_yaml)
    print(f"[Test] Saved resolved config -> {snapshot_path}")


def _init_ray_if_needed(config):
    if not ray.is_initialized():
        # Try to connect to an existing cluster; if fails, start local
        try:
            ray.init(
                runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true", "RAY_DEBUG": "legacy"}},
                num_cpus=config.ray_init.num_cpus,
                address="auto",
            )
        except Exception:
            ray.init(
                runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true", "RAY_DEBUG": "legacy"}},
                num_cpus=config.ray_init.num_cpus,
            )


def _init_rollout_wg(config, checkpoint_path: str = None, role_name="actor_rollout"):
    """Initialize rollout worker group using the same logic as RayPPOTrainer.init_workers()
    
    Returns:
        Tuple[RayWorkerGroup, RayResourcePool]: The worker group and resource pool for cleanup.
    """
    print(f"[Test] Creating resource pool...")
    # Create resource pool with max_colocate_count=1 (same as trainer)
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        use_gpu=True,
        max_colocate_count=1,
        name_prefix="test_rollout"
    )

    print(f"[Test] Creating placement groups...")
    # Create placement groups (this is what the trainer does internally)
    resource_pool.get_placement_groups(device_name="cuda" if is_cuda_available else "npu")

    print(f"[Test] Creating Ray class wrapper...")
    ray_cls = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config.actor_rollout_ref,
        role=role_name
    )

    print(f"[Test] Creating RayWorkerGroup...")
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls,
        device_name="cuda" if is_cuda_available else "npu"
    )

    print(f"[Test] Initializing model...")
    wg.init_model()

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[Test] Loading checkpoint from: {checkpoint_path}")
        wg.load_checkpoint(checkpoint_path)

    print(f"[Test] Worker group initialization complete!")
    return wg, resource_pool


def _decode_texts(tokenizer, ids_batch: np.ndarray | Any) -> List[str]:
    try:
        if isinstance(ids_batch, torch.Tensor):
            ids_batch = ids_batch.detach().cpu()
        texts = tokenizer.batch_decode(ids_batch, skip_special_tokens=True)
    except Exception:
        # Fallback to per-sample decoding if batch_decode not applicable
        texts = []
        for i in range(len(ids_batch)):
            ids = ids_batch[i]
            if hasattr(ids, "detach"):
                ids = ids.detach().cpu().tolist()
            texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


def _create_val_dataloader(config, tokenizer, processor):
    """Create validation dataloader"""
    val_files = config.data.val_files
    if isinstance(val_files, str):
        val_files = [val_files]

    val_dataset = RLHFDataset(
        data_files=val_files,
        tokenizer=tokenizer,
        config=config.data,
        processor=processor,
    )

    batch_size = config.data.val_batch_size or config.data.train_batch_size

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return val_dataloader


def run_validation(
    config,
    checkpoint_path: str = None,
    output_dir: str = None,
) -> Dict[str, Any]:
    """
    Run validation on test data and save results.

    Args:
        config: Configuration object
        checkpoint_path: Path to checkpoint directory
        output_dir: Directory to save outputs

    Returns:
        Dictionary with validation metrics
    """
    # Resolve config and prepare components
    OmegaConf.resolve(config)

    _init_ray_if_needed(config)

    # Prepare tokenizer/processor
    local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Envs and collector
    _train_envs, val_envs = _make_envs_from_config(config)
    traj_collector = _collector_from_config(config, tokenizer, processor)

    # Rollout worker group (with checkpoint loading)
    wg, resource_pool = _init_rollout_wg(config, checkpoint_path=checkpoint_path, role_name="actor_rollout")

    # Create validation dataloader
    val_dataloader = _create_val_dataloader(config, tokenizer, processor)

    # Create reward function (using same logic as main_ppo.py)
    reward_manager_name = config.reward_model.get("reward_manager", "episode")
    if reward_manager_name == 'episode':
        from agent_system.reward_manager.episode import EpisodeRewardManager
        reward_manager_cls = EpisodeRewardManager
    else:
        raise NotImplementedError

    # Use function-based RM for validation (same as in main_ppo.py line 158)
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

    if output_dir is None:
        output_dir = _default_output_dir()
    _ensure_dir(output_dir)
    _save_config_snapshot(config, output_dir)

    print(f"[Test] Writing outputs to: {output_dir}")
    print(f"[Test] Running validation on {len(val_dataloader)} batches")

    # Run validation
    reward_tensor_lst = []
    data_source_lst = []
    success_rate_dict = {}
    total_rollout_time = 0.0

    # Lists to collect samples for saving
    all_samples = []

    val_progress = tqdm(enumerate(val_dataloader), desc="Validation", total=len(val_dataloader))
    for batch_idx, test_data in val_progress:
        test_batch = DataProto.from_single_dict(test_data)

        # repeat test batch for validation sampling
        val_n = config.actor_rollout_ref.rollout.val_kwargs.get("n", 1)
        test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

        # Pop keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
        if "multi_modal_data" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "response" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("response")
        if "turns" in test_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("turns")
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        test_gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": config.actor_rollout_ref.rollout.val_kwargs.get("do_sample", False),
            "validate": True,
        }

        # Run agent-environment loop and log runtime
        rollout_start = time.time()
        test_output_gen_batch = traj_collector.multi_turn_loop(
            gen_batch=test_gen_batch,
            actor_rollout_wg=wg,
            envs=val_envs,
            is_train=False,
        )
        rollout_duration = time.time() - rollout_start
        total_rollout_time += rollout_duration
        print(f"[Test] Batch {batch_idx} multi_turn_rollout runtime: {rollout_duration:.2f}s")

        del test_batch
        test_batch = test_output_gen_batch

        early_stop_reason_batch = test_batch.non_tensor_batch.get("early_stop_reason")
        early_stop_reason_list = None
        if early_stop_reason_batch is not None:
            try:
                early_stop_reason_array = np.array(early_stop_reason_batch, dtype=object)
                early_stop_reason_list = early_stop_reason_array.tolist()
                unique_early_stop_reasons = sorted({str(reason) for reason in early_stop_reason_array.flatten().tolist() if reason})
            except Exception:
                early_stop_reason_list = list(early_stop_reason_batch)
                unique_early_stop_reasons = sorted({str(reason) for reason in early_stop_reason_list if reason})

            if unique_early_stop_reasons:
                print(
                    f"[Test] Early stop reason(s) detected in batch {batch_idx}: "
                    + ", ".join(unique_early_stop_reasons)
                )

        # Store input texts
        input_ids = test_batch.batch["input_ids"]
        input_texts = _decode_texts(tokenizer, input_ids)

        # Store generated outputs
        output_ids = test_output_gen_batch.batch["responses"]
        output_texts = _decode_texts(tokenizer, output_ids)

        # Evaluate using reward_function
        if val_reward_fn is not None:
            result = val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            scores = [0.0] * len(input_texts)

        # Collect success rates (excluding step_success_rates_list which is handled separately)
        for k in test_batch.non_tensor_batch.keys():
            if 'success_rate' in k and k != 'step_success_rates_list':
                if k not in success_rate_dict:
                    success_rate_dict[k] = []
                success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])

        # Collect step_success_rates_list (per-step success rates)
        if 'step_success_rates_list' in test_batch.non_tensor_batch:
            # Find the longest list (from the last step which has all accumulated rates)
            import ast
            step_rates_batch = test_batch.non_tensor_batch['step_success_rates_list']
            longest_rates = []
            for step_rates_str in step_rates_batch:
                if isinstance(step_rates_str, str):
                    try:
                        step_rates = ast.literal_eval(step_rates_str)
                        if len(step_rates) > len(longest_rates):
                            longest_rates = step_rates
                    except (ValueError, SyntaxError):
                        pass

            if longest_rates:
                if 'step_success_rates_list' not in success_rate_dict:
                    success_rate_dict['step_success_rates_list'] = longest_rates
                else:
                    # Keep the longest list across all batches
                    existing = success_rate_dict['step_success_rates_list']
                    if len(longest_rates) > len(existing):
                        success_rate_dict['step_success_rates_list'] = longest_rates

        # Collect step_times_list (per-step timing)
        if 'step_times_list' in test_batch.non_tensor_batch:
            # Find the longest list (from the last step which has all accumulated times)
            import ast
            step_times_batch = test_batch.non_tensor_batch['step_times_list']
            longest_times = []
            for step_times_str in step_times_batch:
                if isinstance(step_times_str, str):
                    try:
                        step_times = ast.literal_eval(step_times_str)
                        if len(step_times) > len(longest_times):
                            longest_times = step_times
                    except (ValueError, SyntaxError):
                        pass

            if longest_times:
                if 'step_times_list' not in success_rate_dict:
                    success_rate_dict['step_times_list'] = longest_times
                else:
                    # Keep the longest list across all batches
                    existing = success_rate_dict['step_times_list']
                    if len(longest_times) > len(existing):
                        success_rate_dict['step_times_list'] = longest_times

        # Save samples
        for i in range(len(input_texts)):
            sample = {
                "batch_idx": batch_idx,
                "sample_idx": i,
                "input": input_texts[i],
                "output": output_texts[i],
                "score": scores[i],
                "model": config.actor_rollout_ref.model.path,
                "api_model": config.env.response_agent.model_name if "response_agent" in config.env else "N/A",
                "traj_collection_type": config.env.traj_collection_type,
            }
            if early_stop_reason_list is not None and i < len(early_stop_reason_list):
                sample['early_stop_reason'] = early_stop_reason_list[i]
            # Add any additional metadata
            for k in test_batch.non_tensor_batch.keys():
                if k not in ["raw_prompt", "raw_prompt_ids", "data_source", "multi_modal_data", "tools_kwargs", "response", "turns"]:
                    try:
                        sample[k] = test_batch.non_tensor_batch[k][i]
                        if hasattr(sample[k], "item"):
                            sample[k] = sample[k].item()
                    except Exception:
                        pass
            all_samples.append(sample)

    # Save all samples to JSONL
    jsonl_path = os.path.join(output_dir, "validation_results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[Test] Saved {len(all_samples)} samples -> {jsonl_path}")

    # Compute metrics
    metric_dict = {}

    if val_reward_fn is not None and len(reward_tensor_lst) > 0:
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)

        # Evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = float(np.mean(rewards))

    # Success rates
    for k, v in success_rate_dict.items():
        if k == 'step_success_rates_list':
            # Store the full list of per-step success rates
            metric_dict['val/step_success_rates_list'] = v
        elif k == 'step_times_list':
            # Store the full list of per-step times
            metric_dict['val/step_times_list'] = v
        else:
            metric_dict[f'val/{k}'] = float(np.mean(v))

    # Runtime metrics
    metric_dict['val/total_rollout_time'] = float(total_rollout_time)
    metric_dict['val/avg_rollout_time_per_batch'] = float(total_rollout_time / len(val_dataloader)) if len(val_dataloader) > 0 else 0.0

    # Save metrics
    metrics_path = os.path.join(output_dir, "validation_metrics.json")
    serializable_metrics = {}
    for k, v in metric_dict.items():
        if isinstance(v, np.generic):
            serializable_metrics[k] = v.item()
        elif isinstance(v, list):
            serializable_metrics[k] = v  # Keep lists as-is
        else:
            serializable_metrics[k] = v
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
    print(f"[Test] Saved metrics -> {metrics_path}")

    # Print metrics
    print("\n" + "="*50)
    print("Validation Metrics:")
    print("="*50)
    for k, v in metric_dict.items():
        if isinstance(v, list):
            # For lists, print the full list
            print(f"{k}: {v}")
            # Also print a formatted version showing step index
            if 'step_success_rates' in k:
                print(f"  Per-step breakdown:")
                for step_idx, rate in enumerate(v):
                    print(f"    Step {step_idx}: {rate:.4f}")
            elif 'step_times' in k:
                print(f"  Per-step timing:")
                for step_idx, step_time in enumerate(v):
                    print(f"    Step {step_idx}: {step_time:.4f}s")
                print(f"  Total time (sum of steps): {sum(v):.4f}s")
                print(f"  Average time per step: {np.mean(v):.4f}s")
        else:
            print(f"{k}: {v:.4f}")
    print("="*50 + "\n")

    # Cleanup
    try:
        val_envs.close()
    except Exception:
        pass

    # Cleanup worker group and placement groups to allow re-runs
    print("[Test] Cleaning up worker group and placement groups...")
    try:
        del wg
    except Exception:
        pass
    try:
        for pg in resource_pool.get_placement_groups():
            ray.util.remove_placement_group(pg)
        # Wait for placement groups to be removed
        time.sleep(10)
    except Exception as e:
        print(f"[Test] Warning: Failed to remove placement groups: {e}")

    return metric_dict


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run validation on test data and save results without training.")
    parser.add_argument("--config", type=str, default="verl/trainer/config/ppo_multi_agent_trainer.yaml", help="Path to base YAML config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint directory (e.g., checkpoints/exp/run/global_step_100).")
    parser.add_argument("--overrides", nargs="*", default=[], help="OmegaConf style key=value overrides.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save test results (default: outputs/test_results/TIMESTAMP).")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of times to run validation (results will be averaged).")

    args = parser.parse_args()

    print("[Test] CLI arguments:")
    print(f"  --config        = {args.config}")
    print(f"  --checkpoint    = {args.checkpoint if args.checkpoint else 'None'}")
    print(f"  --output-dir    = {args.output_dir if args.output_dir else 'Auto (timestamped)'}")
    print(f"  --num-runs      = {args.num_runs}")
    if args.overrides:
        print("  --overrides     =")
        for override in args.overrides:
            print(f"    - {override}")
    else:
        print("  --overrides     = None")

    # Load and override config
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    # Safety: ensure we won't inadvertently train or save
    with open_dict(cfg):
        if "trainer" not in cfg:
            cfg.trainer = {}
        cfg.trainer.save_freq = -1
        cfg.trainer.test_freq = -1
        # Force validation mode
        cfg.trainer.val_only = True

    # Run validation multiple times and aggregate results
    all_run_metrics = []
    for run_idx in range(args.num_runs):
        if args.num_runs > 1:
            print(f"\n{'='*60}")
            print(f"[Test] Starting run {run_idx + 1}/{args.num_runs}")
            print(f"{'='*60}")
            # Create run-specific output directory
            run_output_dir = os.path.join(args.output_dir, f"run_{run_idx + 1}") if args.output_dir else None
        else:
            run_output_dir = args.output_dir

        metrics = run_validation(
            cfg,
            checkpoint_path=args.checkpoint,
            output_dir=run_output_dir,
        )
        all_run_metrics.append(metrics)

    # Aggregate metrics across runs
    if args.num_runs > 1:
        print(f"\n{'='*60}")
        print(f"[Test] Aggregating results across {args.num_runs} runs")
        print(f"{'='*60}")
        
        aggregated_metrics = {}
        aggregated_std = {}
        
        # Collect all keys
        all_keys = set()
        for m in all_run_metrics:
            all_keys.update(m.keys())
        
        for key in all_keys:
            values = [m.get(key) for m in all_run_metrics if key in m]
            if not values:
                continue
            
            # Handle list values (like step_success_rates_list)
            if isinstance(values[0], list):
                # Average element-wise across runs
                max_len = max(len(v) for v in values)
                averaged_list = []
                std_list = []
                for i in range(max_len):
                    step_values = [v[i] for v in values if i < len(v)]
                    averaged_list.append(float(np.mean(step_values)))
                    std_list.append(float(np.std(step_values)))
                aggregated_metrics[key] = averaged_list
                aggregated_std[key] = std_list
            else:
                # Average scalar values
                aggregated_metrics[key] = float(np.mean(values))
                aggregated_std[key] = float(np.std(values))
        
        # Print aggregated results
        print("\n" + "="*60)
        print(f"Aggregated Validation Metrics (mean ± std over {args.num_runs} runs):")
        print("="*60)
        for k in sorted(aggregated_metrics.keys()):
            v = aggregated_metrics[k]
            std = aggregated_std.get(k, 0)
            if isinstance(v, list):
                print(f"{k}: [list of {len(v)} values]")
                if 'step_success_rates' in k:
                    print(f"  Per-step breakdown (mean ± std):")
                    for step_idx, (rate, rate_std) in enumerate(zip(v, std)):
                        print(f"    Step {step_idx}: {rate:.4f} ± {rate_std:.4f}")
                elif 'step_times' in k:
                    print(f"  Per-step timing (mean ± std):")
                    for step_idx, (step_time, time_std) in enumerate(zip(v, std)):
                        print(f"    Step {step_idx}: {step_time:.4f}s ± {time_std:.4f}s")
                    print(f"  Total time (sum of steps): {sum(v):.4f}s ± {sum(std):.4f}s")
                    print(f"  Average time per step: {np.mean(v):.4f}s ± {np.mean(std):.4f}s")
            else:
                print(f"{k}: {v:.4f} ± {std:.4f}")
        print("="*60 + "\n")
        
        # Save aggregated metrics
        if args.output_dir:
            agg_metrics_path = os.path.join(args.output_dir, "aggregated_metrics.json")
            agg_data = {
                "num_runs": args.num_runs,
                "mean": aggregated_metrics,
                "std": aggregated_std,
            }
            with open(agg_metrics_path, "w", encoding="utf-8") as f:
                json.dump(agg_data, f, indent=2, ensure_ascii=False)
            print(f"[Test] Saved aggregated metrics -> {agg_metrics_path}")
        
        final_metrics = aggregated_metrics
    else:
        final_metrics = metrics

    print(f"[Test] Completed validation.")
    print(f"[Test] Overall success rate: {final_metrics.get('val/success_rate', 'N/A')}")


if __name__ == "__main__":
    main()
