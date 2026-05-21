# Sequential Test-Time Scaling for Trajectory Generation
# This module implements sequential test-time scaling where:
# 1. Generate an initial trajectory
# 2. Score it with the verifier
# 3. If score is below threshold, continue from the trajectory state to improve
# 4. Repeat until max iterations or satisfactory score is reached

import copy
import json
import time
import concurrent.futures
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import docker
from datasets import load_dataset
from fire import Fire

from r2egym.agenthub.agent.agent import Agent, AgentArgs
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.trajectory import Trajectory, TrajectoryStep
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.verifiers.run_ef_verifier import run_model
from r2egym.agenthub.verifiers.prepare_ef_verifier_input import traj2verifier_data
from r2egym.docker_bash_utils.docker_list_tags import fetch_docker_tags
from r2egym.logging import INFO, setup_logging

logger = get_logger(__name__)
file_lock = threading.Lock()


def get_verifier_score(
    trajectory: Trajectory,
    verifier_model_name: str,
    max_tokens: int = 65536,
) -> float:
    """
    Get the verifier score for a trajectory.
    
    Args:
        trajectory: The trajectory to score
        verifier_model_name: Name of the verifier model
        max_tokens: Maximum tokens for the verifier input
        
    Returns:
        The probability that the trajectory is successful (0-1)
    """
    try:
        data_entry, success = traj2verifier_data(
            trajectory.model_dump(), max_tokens=max_tokens
        )
        if not success:
            logger.warning("Failed to convert trajectory to verifier format")
            return 0.0
        
        yes_prob = run_model((data_entry, verifier_model_name))
        return yes_prob
    except Exception as e:
        logger.error(f"Error getting verifier score: {e}")
        return 0.0


def continue_trajectory_from_state(
    agent: Agent,
    env: RepoEnv,
    previous_trajectory: Trajectory,
    additional_steps: int = 10,
    temperature: float = 0.0,
    max_steps_absolute: int = 50,
    use_fn_calling: bool = True,
    scaffold: str = "r2egym",
    max_token_limit: int = 65536,
    continuation_prompt: Optional[str] = None,
) -> Trajectory:
    """
    Continue a trajectory from its current state.
    
    This function:
    1. Reconstructs the agent's conversation history from the previous trajectory
    2. Adds a continuation prompt encouraging the agent to improve/fix the solution
    3. Runs the agent for additional steps
    4. Merges the new steps with the previous trajectory
    
    Args:
        agent: The agent to run
        env: The environment (should already have the state from previous trajectory)
        previous_trajectory: The trajectory to continue from
        additional_steps: Number of additional steps to run
        temperature: Temperature for LLM sampling
        max_steps_absolute: Absolute max steps limit
        use_fn_calling: Whether to use function calling
        scaffold: The scaffold to use
        max_token_limit: Maximum token limit
        continuation_prompt: Custom prompt to add for continuation
        
    Returns:
        A new Trajectory that extends the previous one
    """
    # Set up agent configuration
    agent.scaffold = scaffold
    agent.use_fn_calling = use_fn_calling
    agent.llm_timeout = agent.other_args.get("timeout", 3000)
    
    # Reconstruct conversation history from previous trajectory
    agent.history = copy.deepcopy(previous_trajectory.conversation_history)
    agent.trajectory_steps = [
        TrajectoryStep(**step) if isinstance(step, dict) else step
        for step in previous_trajectory.trajectory_steps
    ]
    
    # Add continuation prompt
    if continuation_prompt is None:
        continuation_prompt = """
The previous attempt may not have fully resolved the issue. Please review your work and:
1. Check if the solution properly addresses the problem
2. Verify that all edge cases are handled
3. Ensure the code compiles/runs without errors
4. Make any necessary improvements or fixes

Continue working on the solution.
"""
    
    # Add continuation message to history
    if agent.history and agent.history[-1]["role"] == "user":
        agent.history[-1]["content"] += f"\n\n{continuation_prompt}"
    else:
        agent.history.append({"role": "user", "content": continuation_prompt})
    
    # Run additional steps
    start_time = time.time()
    current_step_count = len(agent.trajectory_steps)
    max_steps = current_step_count + additional_steps
    total_time_traj = previous_trajectory.trajectory_steps[-1].total_time_traj if agent.trajectory_steps else 0
    
    done = False
    exit_reason = "agent"
    
    while not done and len(agent.trajectory_steps) < max_steps:
        steps_remaining = max_steps - len(agent.trajectory_steps)
        stepcount_message = f"Steps Remaining: {steps_remaining}"
        
        if agent.history[-1]["role"] == "user":
            agent.history[-1]["content"] += f"\n{stepcount_message}"
        
        logger.info(stepcount_message)
        
        # Query the LLM
        messages = copy.deepcopy(agent.history)
        try:
            response, llm_exec_time = agent.model_query(messages, temperature)
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            exit_reason = "llm_query_error"
            break
        
        # Get token usage
        if hasattr(response, "usage"):
            usage = response.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
        else:
            prompt_tokens = -1
            completion_tokens = -1
            total_tokens = agent._count_tokens(messages)
        
        # Parse response
        if agent.use_fn_calling:
            thought, action = agent.custom_parser(response)
        else:
            thought, action = agent.parse_response(response.choices[0].message.content)
        
        action_str = action.to_xml_string()
        logger.info(f"THOUGHT:\n{thought}")
        logger.info(f"ACTION:\n{action.to_bashcmd()}")
        
        # Execute action
        try:
            obs, reward, done, info = env.step(action, timeout=90)
        except Exception as e:
            obs = str(e)
            logger.error(f"Error during environment step: {obs}")
            info = {"total_time": 0}
        
        env_exec_time = info.get("total_time", 0)
        total_step_time = llm_exec_time + env_exec_time
        total_time_traj += total_step_time
        step_count = len(agent.trajectory_steps) + 1
        
        # Update history
        if agent.use_fn_calling:
            assistant_response = response.choices[0].message.dict()
            if assistant_response.get("tool_calls"):
                assistant_response["tool_calls"] = assistant_response["tool_calls"][:1]
            agent.history.append(assistant_response)
            try:
                function_name = response.choices[0].message.tool_calls[0].function.name
                function_id = response.choices[0].message.tool_calls[0].id
                agent.history.append({
                    "role": "tool",
                    "content": str(obs),
                    "name": function_name,
                    "tool_call_id": function_id,
                })
            except Exception:
                agent.history.append({"role": "user", "content": str(obs)})
        else:
            assistant_message = f"{thought}\n\n{action.to_xml_string()}"
            agent.history.append({"role": "assistant", "content": assistant_message})
            agent.history.append({"role": "user", "content": str(obs)})
        
        logger.info(f"OBSERVATION:\n{obs}")
        
        # Check termination conditions
        if done:
            exit_reason = "agent" if steps_remaining > 0 else "max_step_limit"
        elif total_tokens >= max_token_limit:
            exit_reason = "token_limit"
            done = True
        elif step_count >= max_steps_absolute:
            exit_reason = "abs_step_limit"
            done = True
        
        # Create trajectory step
        trajectory_step = TrajectoryStep(
            step_idx=step_count - 1,
            thought=thought,
            action=action.to_xml_string(),
            observation=str(obs),
            done=done,
            info=info,
            token_usage_prompt=prompt_tokens,
            token_usage_completion=completion_tokens,
            token_usage_total=total_tokens,
            llm_exec_time=llm_exec_time,
            env_exec_time=env_exec_time,
            total_step_time=total_step_time,
            total_time_traj=total_time_traj,
            step_count=step_count,
        )
        agent.trajectory_steps.append(trajectory_step)
    
    # Get output patch
    output_patch = env.runtime.get_patch()
    
    # Create new trajectory
    conversation_history = json.loads(json.dumps(agent.history, default=str))
    
    new_trajectory = Trajectory(
        trajectory_steps=[step.model_dump() for step in agent.trajectory_steps],
        problem_statement=previous_trajectory.problem_statement,
        docker_image=env.runtime.docker_image,
        agent_args=previous_trajectory.agent_args,
        env_args=previous_trajectory.env_args,
        system_prompt=previous_trajectory.system_prompt,
        user_prompt=previous_trajectory.user_prompt,
        conversation_history=conversation_history,
        max_steps=max_steps,
        max_steps_absolute=max_steps_absolute,
        max_token_limit=max_token_limit,
        max_llm_time=agent.llm_timeout,
        max_exec_time=90,
        max_total_time=previous_trajectory.max_total_time,
        exit_reason=exit_reason,
        output_patch=output_patch,
        ds=previous_trajectory.ds,
        exp_name=previous_trajectory.exp_name,
    )
    
    return new_trajectory


def run_sequential_scaling(
    ds: Dict,
    exp_name: Optional[str] = None,
    # Agent parameters
    max_initial_steps: int = 40,
    max_continuation_steps: int = 10,
    max_steps_absolute: int = 60,
    llm_name: str = "gpt-4o",
    temperature: float = 0.0,
    use_fn_calling: bool = True,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
    # Sequential scaling parameters
    verifier_model_name: str = "openai/Qwen2.5-Coder-32B-Instruct",
    verifier_threshold: float = 0.7,
    max_iterations: int = 3,
    # Environment parameters
    backend: str = "docker",
    max_reward_calc_time: int = 300,
) -> Optional[str]:
    """
    Run sequential test-time scaling on a single instance.
    
    This function:
    1. Generates an initial trajectory
    2. Scores it with the verifier
    3. If below threshold, continues from that state with additional steps
    4. Repeats until max_iterations or threshold is reached
    5. Returns the best trajectory (highest verifier score)
    
    Args:
        ds: Dataset entry with docker_image and other info
        exp_name: Experiment name
        max_initial_steps: Maximum steps for initial trajectory
        max_continuation_steps: Steps to add in each continuation
        max_steps_absolute: Absolute maximum steps
        llm_name: LLM model name
        temperature: LLM temperature
        use_fn_calling: Whether to use function calling
        scaffold: Scaffold type
        max_tokens: Maximum tokens
        verifier_model_name: Name of verifier model
        verifier_threshold: Score threshold to stop scaling
        max_iterations: Maximum number of continuation iterations
        backend: Docker backend
        max_reward_calc_time: Maximum time for reward calculation
        
    Returns:
        JSON string of the best trajectory, or None on error
    """
    logger_instance = setup_logging(
        name=ds["docker_image"].replace("/", "_"),
        log_file=f"run_logs/{exp_name}/{ds['docker_image'].replace('/', '_')}_sequential.log",
        console=True,
        level=INFO,
    )
    
    logger_instance.info(f"Starting sequential scaling on: {ds['docker_image']}")
    logger_instance.info(f"Verifier threshold: {verifier_threshold}, Max iterations: {max_iterations}")
    
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize environment
    env_args = EnvArgs(ds=ds)
    env = RepoEnv(env_args, logger=logger_instance, backend=backend)
    
    # Initialize agent
    if use_fn_calling:
        agent_args = AgentArgs.from_yaml(
            Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_fn_calling.yaml")
        )
    else:
        agent_args = AgentArgs.from_yaml(
            Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_non_fn_calling.yaml")
        )
    agent_args.llm_name = llm_name
    agent = Agent(name="EditAgent", args=agent_args, logger=logger_instance)
    
    # Track all trajectories for best-of-n selection
    all_trajectories: List[Tuple[Trajectory, float]] = []
    
    try:
        # Generate initial trajectory
        logger_instance.info("Generating initial trajectory...")
        trajectory = agent.run(
            env,
            max_steps=max_initial_steps,
            temperature=temperature,
            max_steps_absolute=max_steps_absolute,
            use_fn_calling=use_fn_calling,
            scaffold=scaffold,
            max_token_limit=max_tokens,
        )
        
        # Calculate reward
        reward_start = time.time()
        reward, test_output = env.runtime._calculate_reward(
            get_test_output=True, timeout=max_reward_calc_time
        )
        reward_calc_time = time.time() - reward_start
        
        trajectory.reward = reward
        trajectory.test_output = test_output
        trajectory.ds = ds
        trajectory.exp_name = exp_name
        trajectory.reward_calc_time = reward_calc_time
        
        # Get verifier score
        logger_instance.info("Getting verifier score for initial trajectory...")
        verifier_score = get_verifier_score(trajectory, verifier_model_name, max_tokens)
        trajectory.verifier_prob = verifier_score
        all_trajectories.append((trajectory, verifier_score))
        
        logger_instance.info(f"Iteration 0: reward={reward}, verifier_score={verifier_score:.4f}")
        
        # Sequential improvement loop
        current_trajectory = trajectory
        for iteration in range(1, max_iterations + 1):
            # Check if we've reached the threshold or already succeeded
            if verifier_score >= verifier_threshold or reward == 1:
                logger_instance.info(
                    f"Stopping: verifier_score={verifier_score:.4f} >= threshold={verifier_threshold} "
                    f"or reward={reward}"
                )
                break
            
            # Check if we've hit step limits
            if len(current_trajectory.trajectory_steps) >= max_steps_absolute:
                logger_instance.info(f"Stopping: reached max_steps_absolute={max_steps_absolute}")
                break
            
            logger_instance.info(f"Iteration {iteration}: Continuing trajectory...")
            
            # Increase temperature slightly for diversity
            iter_temperature = min(temperature + 0.1 * iteration, 0.3)
            
            # Create continuation prompt based on current state
            if reward == 0:
                continuation_prompt = f"""
Your previous solution did not pass the tests. The verifier confidence is {verifier_score:.2%}.

Please review your changes and:
1. Check for any bugs or logical errors in your implementation
2. Verify that you're modifying the correct files
3. Ensure your changes handle all edge cases
4. Test your solution mentally before submitting

Continue improving the solution.
"""
            else:
                continuation_prompt = f"""
The verifier confidence is {verifier_score:.2%}, which is below the target.

Please review and improve your solution:
1. Are there any edge cases you might have missed?
2. Is the implementation complete?
3. Are there any potential issues with the code?

Continue refining the solution.
"""
            
            # Continue from current state
            new_trajectory = continue_trajectory_from_state(
                agent=agent,
                env=env,
                previous_trajectory=current_trajectory,
                additional_steps=max_continuation_steps,
                temperature=iter_temperature,
                max_steps_absolute=max_steps_absolute,
                use_fn_calling=use_fn_calling,
                scaffold=scaffold,
                max_token_limit=max_tokens,
                continuation_prompt=continuation_prompt,
            )
            
            # Calculate reward for new trajectory
            reward_start = time.time()
            reward, test_output = env.runtime._calculate_reward(
                get_test_output=True, timeout=max_reward_calc_time
            )
            reward_calc_time = time.time() - reward_start
            
            new_trajectory.reward = reward
            new_trajectory.test_output = test_output
            new_trajectory.reward_calc_time = reward_calc_time
            
            # Get verifier score
            verifier_score = get_verifier_score(new_trajectory, verifier_model_name, max_tokens)
            new_trajectory.verifier_prob = verifier_score
            all_trajectories.append((new_trajectory, verifier_score))
            
            logger_instance.info(
                f"Iteration {iteration}: reward={reward}, verifier_score={verifier_score:.4f}, "
                f"steps={len(new_trajectory.trajectory_steps)}"
            )
            
            current_trajectory = new_trajectory
        
        # Select best trajectory based on verifier score
        best_trajectory, best_score = max(all_trajectories, key=lambda x: x[1])
        logger_instance.info(
            f"Best trajectory: verifier_score={best_score:.4f}, "
            f"reward={best_trajectory.reward}, steps={len(best_trajectory.trajectory_steps)}"
        )
        
        # Close environment
        env.close()
        
        return best_trajectory.model_dump_json()
        
    except Exception as e:
        logger_instance.error(f"Error during sequential scaling: {e}")
        env.close()
        
        # Return best trajectory so far if we have any
        if all_trajectories:
            best_trajectory, _ = max(all_trajectories, key=lambda x: x[1])
            return best_trajectory.model_dump_json()
        return None


def run_sequential_scaling_multiple(
    dataset: str,
    split: str,
    k: int = 1,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx: int = 0,
    max_workers: Optional[int] = None,
    # Agent parameters
    max_initial_steps: int = 40,
    max_continuation_steps: int = 10,
    max_steps_absolute: int = 60,
    llm_name: str = "gpt-4o",
    temperature: float = 0.0,
    use_fn_calling: bool = True,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
    # Sequential scaling parameters
    verifier_model_name: str = "openai/Qwen2.5-Coder-32B-Instruct",
    verifier_threshold: float = 0.7,
    max_iterations: int = 3,
    # Environment parameters
    backend: str = "docker",
    max_reward_calc_time: int = 300,
    use_existing: bool = True,
):
    """
    Run sequential test-time scaling on multiple instances.
    
    Args:
        dataset: HuggingFace dataset name
        split: Dataset split
        k: Number of instances to process
        traj_dir: Directory to save trajectories
        exp_name: Experiment name
        start_idx: Starting index
        max_workers: Maximum parallel workers
        ... (other args same as run_sequential_scaling)
    """
    # Load dataset
    ds = load_dataset(dataset, split=split)
    ds = ds.shuffle(seed=42)
    
    selected_idx = range(start_idx, min(start_idx + k, len(ds)))
    ds_selected = [ds[i] for i in selected_idx]
    
    logger.info(f"Dataset: {dataset}, Split: {split}, Processing {len(ds_selected)} instances")
    
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_sequential"
    
    # Setup output directory
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"
    
    # Filter existing if needed
    if use_existing and jsonl_file.exists():
        with open(jsonl_file) as f:
            existing_dockers = []
            for line in f:
                try:
                    existing_dockers.append(
                        Trajectory.load_from_model_dump_json(line).ds["docker_image"]
                    )
                except:
                    pass
        
        ds_selected = [
            entry for entry in ds_selected
            if entry["docker_image"] not in existing_dockers
        ]
    
    logger.info(f"Processing {len(ds_selected)} instances after filtering")
    
    # Process instances
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(
                run_sequential_scaling,
                ds=ds_entry,
                exp_name=exp_name,
                max_initial_steps=max_initial_steps,
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
            ): ds_entry["docker_image"]
            for ds_entry in ds_selected
        }
        
        with open(jsonl_file, "a") as f:
            for future in concurrent.futures.as_completed(future_to_image):
                docker_image = future_to_image[future]
                try:
                    result = future.result()
                    if result is not None:
                        with file_lock:
                            f.write(result + "\n")
                            f.flush()
                        logger.info(f"Completed: {docker_image}")
                except Exception as e:
                    logger.error(f"Exception for {docker_image}: {e}")
    
    logger.info(f"Sequential scaling completed. Results saved to {jsonl_file}")


if __name__ == "__main__":
    Fire({
        "run": run_sequential_scaling,
        "run_multiple": run_sequential_scaling_multiple,
    })
