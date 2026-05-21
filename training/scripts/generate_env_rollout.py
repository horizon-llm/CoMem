#!/usr/bin/env python3
# Copyright 2025
# Standalone rollout generator: runs env rollouts and saves trajectories without any parameter updates.
#
# Usage examples:
#   1. Generate 10 iterations from scratch:
#      python generate_env_rollout.py --num-iters 10 --output-dir outputs/my_rollouts
#
#   2. Continue generation from iteration 5 to 15:
#      python generate_env_rollout.py --num-iters 15 --start-iter 5 --output-dir outputs/my_rollouts
#
#   3. Generate more iterations, automatically skipping existing files:
#      python generate_env_rollout.py --num-iters 20 --skip-existing --output-dir outputs/my_rollouts
#
#   4. Restart generation but skip already completed iterations:
#      python generate_env_rollout.py --num-iters 100 --skip-existing --output-dir outputs/my_rollouts

import os
import json
import time
import glob
from datetime import datetime
from typing import Dict, Any, List

import ray
import numpy as np
from omegaconf import OmegaConf, open_dict

from verl.utils.fs import copy_to_local
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.model import compute_position_id_with_mask
from verl.utils.device import is_cuda_available
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl import DataProto

# Env + rollout utilities
from agent_system.environments import make_envs, make_conv_envs
from agent_system.multi_turn_rollout import TrajectoryCollector, ConvTrajectoryCollector, MultiAgentTrajectoryCollector, MainAPIMemTrajectoryCollector


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _default_output_dir(base: str = "outputs/rollouts") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, stamp)


def _build_minimal_gen_batch(config, tokenizer, batch_size: int) -> DataProto:
    """
    Build a minimal DataProto to satisfy TrajectoryCollector.vanilla_multi_turn_loop.
    It only needs a non-empty batch for length checks and meta info for generation.
    """
    # Minimal dummy tensors just to define batch dimension
    import torch
    input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
    attention_mask = torch.ones((batch_size, 1), dtype=torch.long)
    position_ids = compute_position_id_with_mask(attention_mask)

    non_tensors = {
        # Placeholders; data_source is useful for downstream analytics
        "raw_prompt": np.array([[{"role": "user", "content": ""}]] * batch_size, dtype=object),
        "data_source": np.array([config.env.env_name] * batch_size, dtype=object),
    }

    meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        # Follow rollout validation kwargs for deterministic/non-deterministic behavior
        "do_sample": bool(OmegaConf.select(config, "actor_rollout_ref.rollout.val_kwargs.do_sample", default=False)),
        "validate": False,
    }

    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        non_tensors=non_tensors,
        meta_info=meta_info,
    )


def _collector_from_config(config, tokenizer, processor):
    traj_collection_type = config.env.get("traj_collection_type", "vanilla")
    if traj_collection_type == "mainapimem":
        return MainAPIMemTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    elif traj_collection_type == "multi-agent":
        return MultiAgentTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    elif traj_collection_type == "conv":
        return ConvTrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    else:
        return TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)


def _make_envs_from_config(config):
    env_manager_type = config.env.get("env_manager_type", "vanilla")
    if env_manager_type == "conv":
        return make_conv_envs(config)
    return make_envs(config)


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


def _init_rollout_wg(config, role_name="rollout") -> RayWorkerGroup:
    ray_cls = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config.actor_rollout_ref, role=role_name)
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls, device_name="cuda" if is_cuda_available else "npu")
    wg.init_model()
    return wg


def _decode_texts(tokenizer, ids_batch: np.ndarray | Any) -> List[str]:
    # ids_batch can be a torch tensor inside DataProto batch; access via .cpu().tolist() is fine
    try:
        import torch
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


def _dataproto_to_jsonl_rows(tokenizer, dp: DataProto, config) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    b = dp.batch
    nt = dp.non_tensor_batch

    prompts_txt = _decode_texts(tokenizer, b["prompts"]) if "prompts" in b.keys() else [None] * len(dp)
    responses_txt = _decode_texts(tokenizer, b["responses"]) if "responses" in b.keys() else [None] * len(dp)

    # Optional fields
    rewards = nt.get("rewards", np.array([None] * len(dp)))
    is_active = nt.get("active_masks", np.array([True] * len(dp)))
    traj_uid = nt.get("traj_uid", np.array([None] * len(dp)))
    uid = nt.get("uid", np.array([None] * len(dp)))
    step = nt.get("step", np.array([None] * len(dp)))
    is_action_valid = nt.get("is_action_valid", np.array([True] * len(dp)))
    episode_rewards = nt.get("episode_rewards", np.array([None] * len(dp)))
    episode_lengths = nt.get("episode_lengths", np.array([None] * len(dp)))
    conv_strings = nt.get("conv_strings", np.array([None] * len(dp)))

    # Success metrics (could contain multiple keys)
    success_keys = [k for k in nt.keys() if k.endswith("_success_rate") or k == "success_rate"]

    for i in range(len(dp)):
        row: Dict[str, Any] = {
            "uid": uid[i].item() if hasattr(uid[i], "item") else uid[i],
            "traj_uid": traj_uid[i].item() if hasattr(traj_uid[i], "item") else traj_uid[i],
            "step": step[i].item() if hasattr(step[i], "item") else step[i],
            "prompt": prompts_txt[i],
            "response": responses_txt[i],
            "reward": float(rewards[i]) if rewards[i] is not None else None,
            "active": bool(is_active[i]) if is_active[i] is not None else None,
            "is_action_valid": bool(is_action_valid[i]) if is_action_valid[i] is not None else None,
            "episode_rewards": float(episode_rewards[i]) if episode_rewards[i] is not None else None,
            "episode_lengths": int(episode_lengths[i]) if episode_lengths[i] is not None else None,
            "model_name": config.actor_rollout_ref.model.path,
            "conv_string": conv_strings[i],
        }
        for k in success_keys:
            v = nt[k][i]
            try:
                v = float(v)
            except Exception:
                pass
            row[k] = v
        rows.append(row)
    return rows


def generate_rollouts(
    config,
    num_iters: int = 1,
    output_dir: str | None = None,
    save_dataproto: bool = False,
    start_iter: int = 0,
    skip_existing: bool = False,
) -> str:
    """
    Generate rollouts for multiple iterations.
    
    Args:
        config: Configuration object
        num_iters: Total number of iterations to generate
        output_dir: Directory to save outputs
        save_dataproto: Whether to save DataProto files
        start_iter: Starting iteration number (for continuation)
        skip_existing: If True, skip iterations that already have output files
    
    Returns:
        Path to output directory
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
    envs, _val_envs = _make_envs_from_config(config)
    traj_collector = _collector_from_config(config, tokenizer, processor)

    # Rollout worker group
    wg = _init_rollout_wg(config, role_name="actor_rollout")

    # Minimal gen batch aligned with env batch size
    env_batch_size = config.data.train_batch_size
    gen_batch = _build_minimal_gen_batch(config, tokenizer, env_batch_size)

    if output_dir is None:
        output_dir = _default_output_dir()
    _ensure_dir(output_dir)

    print(f"[Rollout] Writing outputs to: {output_dir}")
    
    # Check for existing files if skip_existing is enabled
    existing_iters = set()
    if skip_existing:
        existing_files = glob.glob(os.path.join(output_dir, "rollouts_*.jsonl"))
        for f in existing_files:
            basename = os.path.basename(f)
            try:
                iter_num = int(basename.split("_")[1].split(".")[0])
                existing_iters.add(iter_num)
            except (IndexError, ValueError):
                pass
        if existing_iters:
            print(f"[Rollout] Found existing iterations: {sorted(existing_iters)}")

    all_jsonl_paths: List[str] = []

    for it in range(start_iter, num_iters):
        # Skip if file already exists and skip_existing is True
        if skip_existing and it in existing_iters:
            jsonl_path = os.path.join(output_dir, f"rollouts_{it:05d}.jsonl")
            print(f"[Rollout] Skipping iteration {it + 1}/{num_iters} (already exists: {jsonl_path})")
            all_jsonl_paths.append(jsonl_path)
            envs.reset() # Reset envs to make sure the seed is correct
            continue
            
        print(f"[Rollout] Iteration {it + 1}/{num_iters}")
        dp: DataProto = traj_collector.multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=wg,
            envs=envs,
            is_train=False,
        )

        # Save JSONL rows
        rows = _dataproto_to_jsonl_rows(tokenizer, dp, config)
        jsonl_path = os.path.join(output_dir, f"rollouts_{it:05d}.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Rollout] Saved {len(rows)} steps -> {jsonl_path}")
        all_jsonl_paths.append(jsonl_path)

        if save_dataproto:
            dp_path = os.path.join(output_dir, f"rollouts_{it:05d}.dp")
            dp.save_to_disk(dp_path)
            print(f"[Rollout] Saved DataProto -> {dp_path}")

    # Cleanup
    try:
        envs.close()
    except Exception:
        pass

    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate env rollouts and save to JSONL without training.")
    parser.add_argument("--config", type=str, default="verl/trainer/config/ppo_memory_trainer.yaml", help="Path to base YAML config.")
    parser.add_argument("--overrides", nargs="*", default=[], help="OmegaConf style key=value overrides.")
    parser.add_argument("--num-iters", type=int, default=1, help="How many rollout iterations to run (each iteration runs batch_size parallel episodes).")
    parser.add_argument("--start-iter", type=int, default=0, help="Starting iteration number (useful for continuing generation).")
    parser.add_argument("--output-dir", type=str, default="outputs/rollouts", help="Directory to save JSONL files.")
    parser.add_argument("--save-dataproto", action="store_true", help="Also save raw DataProto binaries (.dp).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip iterations that already have output files.")

    args = parser.parse_args()

    # Load and override config
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    # Safety: ensure we won't inadvertently train or update
    with open_dict(cfg):
        # No checkpoints, no saving, no testing
        if "trainer" not in cfg:
            cfg.trainer = {}
        cfg.trainer.save_freq = -1
        cfg.trainer.test_freq = -1
        # Force single validation sample settings off
        if "actor_rollout_ref" in cfg and "rollout" in cfg.actor_rollout_ref:
            if "val_kwargs" not in cfg.actor_rollout_ref.rollout:
                cfg.actor_rollout_ref.rollout.val_kwargs = {}
            if "do_sample" not in cfg.actor_rollout_ref.rollout.val_kwargs:
                cfg.actor_rollout_ref.rollout.val_kwargs.do_sample = False
        # Disable reward model usage
        if "reward_model" in cfg:
            cfg.reward_model.enable = False
        # Ensure env grouping default is disabled unless user set it
        if "env" in cfg and "rollout" in cfg.env and cfg.env.rollout.get("n", -1) == -1:
            cfg.env.rollout.n = 1

    out_dir = generate_rollouts(
        cfg, 
        num_iters=args.num_iters, 
        output_dir=args.output_dir, 
        save_dataproto=args.save_dataproto,
        start_iter=args.start_iter,
        skip_existing=args.skip_existing,
    )
    print(f"[Rollout] Completed. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
