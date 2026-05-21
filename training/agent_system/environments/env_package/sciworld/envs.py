import os
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray
import copy
import json

from agent_system.environments.env_package.sciworld.agentenv_sciworld import SciworldEnvClient

@ray.remote(num_cpus=0.2)
class SciworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, env_server_base: str):
        self.env = SciworldEnvClient(env_server_base, data_len=1, timeout=2400)
    
    def step(self, action):
        """Execute a step in the environment"""
        # Avoid stepping a finished episode
        if getattr(self.env, "info", None) and self.env.info.get("done", False):
            info = copy.deepcopy(self.env.info)
            return info["observation"], 0.0, True, info
        
        output = self.env.step(action)
        obs, reward, done = output.state, output.reward, output.done
        info = copy.deepcopy(self.env.info)
        score = info.get('score', 0.0)
        score = score / 10.0 # Normalize score to [-10, 10]
        info['won'] = done and score == 10.0
        reward = info.get('reward', 0.0) / 10.0  # Normalize reward to [-10, 10]
        info['task_score'] = copy.deepcopy(score)
        return obs, reward, done, info

    def reset(self, idx):
        """Reset the environment with given session index"""
        response = self.env.reset(idx)
        info = copy.deepcopy(self.env.info)
        return info['observation'], info
    
    def close(self):
        """Close the environment"""
        self.env.close()

# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class SciworldMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        env_server_base = env_kwargs.get('env_server_base', 'http://0.0.0.0:36001')

        # -------------------------- Ray actors setup --------------------------
        self._workers = []

        for _ in range(self.num_processes):
            worker = SciworldWorker.remote(env_server_base)
            self._workers.append(worker)
        
        if not self.is_train:
            self.item_ids = json.load(open("agent_system/environments/env_package/sciworld/agentenv_sciworld/data/eval/sciworld_test.json", 'r'))
        else:
            self.item_ids = json.load(open("agent_system/environments/env_package/sciworld/agentenv_sciworld/data/train/sciworld_train.json", 'r'))
        self.item_ids = [int(item['item_id'].split("_")[1]) for item in self.item_ids]
    
    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )
        
        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.item_ids, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_sciworld_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    return SciworldMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )