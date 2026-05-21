# Post-hoc Sequential Scaling for Existing Trajectories
# This module allows improving existing trajectories by continuing from their final state

import copy
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
from tqdm import tqdm

from r2egym.agenthub.trajectory.trajectory import Trajectory
from r2egym.agenthub.run.edit_sequential import (
    run_sequential_scaling,
    get_verifier_score,
)
from r2egym.agenthub.utils.log import get_logger

logger = get_logger(__name__)


def improve_trajectory(
    trajectory_json: str,
    verifier_model_name: str = "openai/Qwen2.5-Coder-32B-Instruct",
    verifier_threshold: float = 0.7,
    max_iterations: int = 3,
    max_continuation_steps: int = 10,
    max_steps_absolute: int = 60,
    llm_name: str = "gpt-4o",
    temperature: float = 0.0,
    use_fn_calling: bool = True,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
    backend: str = "docker",
    max_reward_calc_time: int = 300,
) -> Optional[str]:
    """
    Improve an existing trajectory using sequential scaling.
    
    If the trajectory already has a high verifier score or reward=1,
    returns the original. Otherwise, continues from the trajectory's
    state to try to improve it.
    
    Args:
        trajectory_json: JSON string of the trajectory to improve
        ... (other args same as run_sequential_scaling)
        
    Returns:
        JSON string of the improved trajectory
    """
    trajectory = Trajectory.model_validate_json(trajectory_json)
    
    # Check if already good enough
    if trajectory.reward == 1:
        logger.info(f"Trajectory already successful (reward=1), skipping: {trajectory.docker_image}")
        return trajectory_json
    
    # Get current verifier score if not present
    if trajectory.verifier_prob is None:
        trajectory.verifier_prob = get_verifier_score(
            trajectory, verifier_model_name, max_tokens
        )
    
    if trajectory.verifier_prob >= verifier_threshold:
        logger.info(
            f"Trajectory already above threshold ({trajectory.verifier_prob:.4f}), "
            f"skipping: {trajectory.docker_image}"
        )
        return trajectory_json
    
    logger.info(
        f"Improving trajectory: {trajectory.docker_image}, "
        f"current score: {trajectory.verifier_prob:.4f}"
    )
    
    # Run sequential scaling starting from this trajectory's dataset entry
    result = run_sequential_scaling(
        ds=trajectory.ds,
        exp_name=trajectory.exp_name,
        max_initial_steps=0,  # Skip initial generation, we already have a trajectory
        max_continuation_steps=max_continuation_steps,
        max_steps_absolute=max_steps_absolute,
        llm_name=llm_name,
        temperature=temperature,
        use_fn_calling=use_fn_calling,
        scaffold=scaffold,
        max_tokens=max_tokens,
        verifier_model_name=verifier_model_name,
        verifier_threshold=verifier_threshold,
        max_iterations=max_iterations,
        backend=backend,
        max_reward_calc_time=max_reward_calc_time,
    )
    
    return result if result else trajectory_json


def improve_trajectories_from_file(
    input_file: str,
    output_file: str,
    verifier_model_name: str = "openai/Qwen2.5-Coder-32B-Instruct",
    verifier_threshold: float = 0.7,
    max_iterations: int = 3,
    max_continuation_steps: int = 10,
    max_steps_absolute: int = 60,
    llm_name: str = "gpt-4o",
    max_workers: int = 4,
    only_improve_failures: bool = True,
    backend: str = "docker",
):
    """
    Improve trajectories from a JSONL file using sequential scaling.
    
    Args:
        input_file: Path to input JSONL file with trajectories
        output_file: Path to output JSONL file for improved trajectories
        verifier_model_name: Name of verifier model
        verifier_threshold: Score threshold to stop scaling
        max_iterations: Maximum continuation iterations
        max_continuation_steps: Steps per continuation
        max_steps_absolute: Absolute max steps
        llm_name: LLM model name
        max_workers: Number of parallel workers
        only_improve_failures: If True, only improve trajectories with reward != 1
        backend: Docker backend type
    """
    # Load trajectories
    trajectories: List[Trajectory] = []
    with open(input_file, "r") as f:
        for line in f:
            try:
                traj = Trajectory.model_validate_json(line)
                trajectories.append(traj)
            except Exception as e:
                logger.warning(f"Failed to parse trajectory: {e}")
    
    logger.info(f"Loaded {len(trajectories)} trajectories from {input_file}")
    
    # Filter trajectories to improve
    if only_improve_failures:
        to_improve = [t for t in trajectories if t.reward != 1]
        already_good = [t for t in trajectories if t.reward == 1]
        logger.info(f"Improving {len(to_improve)} failed trajectories, keeping {len(already_good)} successful")
    else:
        to_improve = trajectories
        already_good = []
    
    # Improve trajectories in parallel
    improved = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                improve_trajectory,
                trajectory_json=t.model_dump_json(),
                verifier_model_name=verifier_model_name,
                verifier_threshold=verifier_threshold,
                max_iterations=max_iterations,
                max_continuation_steps=max_continuation_steps,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                backend=backend,
            ): t.docker_image
            for t in to_improve
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Improving"):
            docker_image = futures[future]
            try:
                result = future.result()
                if result:
                    improved.append(Trajectory.model_validate_json(result))
                    logger.info(f"Improved: {docker_image}")
            except Exception as e:
                logger.error(f"Failed to improve {docker_image}: {e}")
    
    # Combine with already good trajectories
    all_results = already_good + improved
    
    # Write output
    with open(output_file, "w") as f:
        for traj in all_results:
            f.write(traj.model_dump_json() + "\n")
    
    # Report stats
    original_pass = sum(1 for t in trajectories if t.reward == 1)
    new_pass = sum(1 for t in all_results if t.reward == 1)
    
    logger.info(f"Results written to {output_file}")
    logger.info(f"Original pass rate: {original_pass}/{len(trajectories)} = {original_pass/len(trajectories):.2%}")
    logger.info(f"New pass rate: {new_pass}/{len(all_results)} = {new_pass/len(all_results):.2%}")


def select_best_with_sequential_scaling(
    traj_file_glob: str,
    output_file: str,
    verifier_model_name: str = "openai/Qwen2.5-Coder-32B-Instruct",
    verifier_threshold: float = 0.7,
    max_iterations: int = 2,
    max_continuation_steps: int = 5,
    max_workers: int = 4,
    llm_name: str = "gpt-4o",
    backend: str = "docker",
):
    """
    For each instance, select the best trajectory and optionally improve it.
    
    This implements a hybrid approach:
    1. First, select the best trajectory per instance based on verifier score
    2. If the best trajectory is below threshold and didn't pass, try to improve it
    
    Args:
        traj_file_glob: Glob pattern for input trajectory files
        output_file: Output file path
        verifier_model_name: Verifier model name
        verifier_threshold: Threshold for improvement
        max_iterations: Max improvement iterations
        max_continuation_steps: Steps per continuation
        max_workers: Parallel workers
        llm_name: LLM model name
        backend: Docker backend
    """
    from collections import defaultdict
    
    # Load all trajectories
    traj_files = glob.glob(traj_file_glob)
    all_trajs_by_docker: Dict[str, List[Trajectory]] = defaultdict(list)
    
    for traj_file in traj_files:
        with open(traj_file, "r") as f:
            for line in f:
                try:
                    traj = Trajectory.model_validate_json(line)
                    all_trajs_by_docker[traj.docker_image].append(traj)
                except:
                    pass
    
    logger.info(f"Loaded trajectories for {len(all_trajs_by_docker)} instances")
    
    # Select best per instance
    best_trajs = []
    to_improve = []
    
    for docker_image, trajs in all_trajs_by_docker.items():
        # First priority: any trajectory that passed
        passing = [t for t in trajs if t.reward == 1]
        if passing:
            best = max(passing, key=lambda x: x.verifier_prob or 0)
            best_trajs.append(best)
            continue
        
        # Otherwise: highest verifier score
        best = max(trajs, key=lambda x: x.verifier_prob or 0)
        
        if (best.verifier_prob or 0) >= verifier_threshold:
            best_trajs.append(best)
        else:
            to_improve.append(best)
    
    logger.info(f"Best trajectories: {len(best_trajs)} good, {len(to_improve)} to improve")
    
    # Improve low-scoring trajectories
    if to_improve:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    improve_trajectory,
                    trajectory_json=t.model_dump_json(),
                    verifier_model_name=verifier_model_name,
                    verifier_threshold=verifier_threshold,
                    max_iterations=max_iterations,
                    max_continuation_steps=max_continuation_steps,
                    llm_name=llm_name,
                    backend=backend,
                ): t.docker_image
                for t in to_improve
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Improving"):
                docker_image = futures[future]
                try:
                    result = future.result()
                    if result:
                        best_trajs.append(Trajectory.model_validate_json(result))
                except Exception as e:
                    logger.error(f"Failed to improve {docker_image}: {e}")
                    # Keep original
                    orig = next(t for t in to_improve if t.docker_image == docker_image)
                    best_trajs.append(orig)
    
    # Write output
    with open(output_file, "w") as f:
        for traj in best_trajs:
            f.write(traj.model_dump_json() + "\n")
    
    pass_count = sum(1 for t in best_trajs if t.reward == 1)
    logger.info(f"Final pass rate: {pass_count}/{len(best_trajs)} = {pass_count/len(best_trajs):.2%}")
    logger.info(f"Results written to {output_file}")


if __name__ == "__main__":
    fire.Fire({
        "improve": improve_trajectory,
        "improve_file": improve_trajectories_from_file,
        "select_and_improve": select_best_with_sequential_scaling,
    })
