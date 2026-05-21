"""
Dummy environment for latency benchmarking of COMEM agents.

Provides the same interface as RepoEnv but returns instant dummy observations
without requiring Docker containers or real code repositories.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from r2egym.agenthub.action import Action
from r2egym.agenthub.observation import Observation
from r2egym.agenthub.environment.env import EnvArgs
from r2egym.agenthub.agent.commands import ParseCommandBash


def generate_text_with_n_tokens(n_tokens: int, tokenizer) -> str:
    """Generate dummy text that tokenizes to approximately n_tokens tokens.

    Args:
        n_tokens: Target number of tokens.
        tokenizer: A HuggingFace tokenizer instance.

    Returns:
        A string that tokenizes to approximately n_tokens tokens.
    """
    if n_tokens <= 0:
        return ""

    base_unit = (
        "The function processes the input data by iterating through each element "
        "and applying the transformation to compute the result value.\n"
    )
    # Estimate tokens per unit
    tokens_per_unit = len(tokenizer.encode(base_unit, add_special_tokens=False))
    if tokens_per_unit == 0:
        tokens_per_unit = 1

    repetitions = (n_tokens // tokens_per_unit) + 2
    candidate = base_unit * repetitions

    # Encode, truncate to exact token count, decode back
    tokens = tokenizer.encode(candidate, add_special_tokens=False)
    trimmed_tokens = tokens[:n_tokens]
    return tokenizer.decode(trimmed_tokens)


class DummyCommit:
    """Mock commit object that returns empty patches."""

    def get_patch(self, **kwargs) -> str:
        return ""


class DummyRuntime:
    """Mock runtime that provides the interface expected by COMEMAgent.run().

    Replaces DockerRuntime without starting any containers.
    """

    def __init__(
        self,
        problem_statement: str = "This is a dummy problem statement for latency benchmarking.",
        docker_image: str = "dummy/benchmark",
    ):
        self.docker_image = docker_image
        self.commit = DummyCommit()
        self.ds = {
            "docker_image": docker_image,
            "repo": "dummy/benchmark-repo",
            "problem_statement": problem_statement,
        }
        self._problem_statement = problem_statement

    def get_task_instruction(self) -> str:
        return self._problem_statement

    def get_patch(self) -> str:
        return ""

    def close(self):
        pass


class DummyEnv:
    """Lightweight environment for latency benchmarking.

    Satisfies the interface used by COMEMAgent.run() without Docker.
    Returns instant dummy observations so env_exec_time ≈ 0.
    The environment never sets done=True — the agent runs for exactly
    max_steps_absolute steps, controlled by the agent loop.

    Args:
        observation_tokens: Number of tokens in each dummy observation.
        problem_statement_tokens: Number of tokens in the dummy problem statement.
        tokenizer: A HuggingFace tokenizer for generating token-sized text.
    """

    def __init__(
        self,
        observation_tokens: int = 200,
        problem_statement_tokens: int = 500,
        tokenizer=None,
    ):
        self.observation_tokens = observation_tokens
        self.problem_statement_tokens = problem_statement_tokens
        self.tokenizer = tokenizer

        # Generate the dummy texts
        if tokenizer is not None:
            self._observation_text = generate_text_with_n_tokens(
                observation_tokens, tokenizer
            )
            problem_statement = generate_text_with_n_tokens(
                problem_statement_tokens, tokenizer
            )
        else:
            self._observation_text = "Dummy observation output. " * max(1, observation_tokens // 5)
            problem_statement = "Dummy problem statement for benchmarking. " * max(1, problem_statement_tokens // 7)

        self.runtime = DummyRuntime(problem_statement=problem_statement)
        self.args = EnvArgs(ds=self.runtime.ds)
        self.commands: List = []
        self.cmd_parser = ParseCommandBash()

    def reset(self) -> str:
        return "Environment reset"

    def add_commands(self, cmd_files: list) -> None:
        """Parse command files locally. No Docker copy needed."""
        cmds = []
        for cmd_file in cmd_files:
            parsed_commands = self.cmd_parser.parse_command_file(cmd_file)
            cmds.extend(parsed_commands)
        self.commands = cmds

    def step(
        self, action: Action, timeout: int = None
    ) -> Tuple[Observation, int, bool, Dict[str, Any]]:
        """Return a dummy observation instantly.

        Always returns done=False so the agent runs for the full
        max_steps_absolute steps. Skips action validation since
        the LLM may produce random/unparseable actions during benchmarking.
        """
        obs = Observation(self._observation_text, 0, action)
        return obs, 0, False, {"total_time": 0.0}

    def compute_reward(self, timeout: int = None):
        return 0, ""

    def close(self):
        pass
