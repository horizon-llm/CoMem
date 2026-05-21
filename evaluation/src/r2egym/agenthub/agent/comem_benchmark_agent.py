"""
Benchmark agent for COMEM-KSO latency measurement.

Thin subclass of COMEMAgent from comem_agent_kso.py that adds:
1. Input token padding (via _prepare_messages_for_query override)
2. Agent output token control (via _model_query override with min_tokens/max_tokens)
3. Memory output token control (via _start_memory_compression override with min_tokens/max_tokens)

The critical run() loop is NOT overridden — the real COMEM-KSO pipeline
(memory compression, interleaving, KSO scheduling, timing) runs unmodified.
"""

import copy
import time
import requests
from typing import Any, Dict, List, Optional, Tuple

import litellm

from r2egym.agenthub.action import Action
from r2egym.agenthub.agent.comem_agent_kso import COMEMAgent, AgentArgs
from r2egym.agenthub.environment.dummy_env import generate_text_with_n_tokens
from transformers import AutoTokenizer


class BenchmarkCOMEMAgent(COMEMAgent):
    """COMEM-KSO agent with controlled input/output token counts for benchmarking.

    Overrides only 3 methods:
    - _prepare_messages_for_query: pads messages to input_token_target (calls super first)
    - _model_query: plain chat completion with min_tokens/max_tokens (no tools/scaffolds)
    - _start_memory_compression: adds min_tokens control for memory model output

    Everything else (run loop, interleaving, scheduling,
    timing, action parsing) runs unmodified from COMEMAgent.
    """

    def __init__(
        self,
        name: str,
        args: AgentArgs,
        input_token_target: int = 4096,
        output_token_target: int = 2048,
        memory_output_token_target: Optional[int] = None,
        logger=None,
    ):
        super().__init__(name, args, logger)
        self.input_token_target = input_token_target
        self.output_token_target = output_token_target
        # If not set, use the agent args' memory_model_max_gen_tokens (no min_tokens forcing)
        self.memory_output_token_target = memory_output_token_target

        # Load the agent model's tokenizer for padding generation
        model_name = "/".join(args.llm_name.split("/")[1:])  # strip "hosted_vllm/" prefix
        self.padding_tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    def parse_response(self, response_text: str) -> Tuple[str, Action]:
        """Keep the FULL raw model output as the 'thought'.

        Matches the FC benchmark agent's behavior so that history grows
        at the same rate, making latency comparisons fair.
        """
        thought = response_text.strip() if response_text else ""
        action = Action.from_string("")
        return thought, action

    def _prepare_messages_for_query(
        self,
        problem_statement: str,
        summary: str,
        history_to_append: str,
        stepcount_message: str,
    ) -> List[Dict[str, str]]:
        """Prepare messages and pad to input_token_target.

        Calls the parent method first (real message construction) then
        pads the last message to reach the target token count.
        """
        messages = super()._prepare_messages_for_query(
            problem_statement, summary, history_to_append, stepcount_message
        )

        current_tokens = self._count_tokens(messages)
        if current_tokens < self.input_token_target:
            deficit = self.input_token_target - current_tokens
            padding = generate_text_with_n_tokens(deficit, self.padding_tokenizer)
            messages[-1]["content"] += "\n\n[BENCHMARK CONTEXT PADDING]\n" + padding
            self.logger.info(
                f"Padded input from {current_tokens} to ~{self.input_token_target} tokens "
                f"(added {deficit} tokens of padding)"
            )
        else:
            self.logger.info(
                f"Input already at {current_tokens} tokens (target: {self.input_token_target})"
            )

        return messages

    async def _model_query(
        self, messages: List[Dict[str, str]], temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Query the LLM with controlled output token count.

        Simplified version of the parent _model_query — no tools or function
        calling since the benchmark uses random input/output tokens.
        All timing logic remains identical to the parent.
        """
        response = None
        retries = 0

        messages_ = copy.deepcopy(messages)
        vllm_llm_name = "/".join(self.llm_name.split("/")[1:])  # remove openai/ or hosted_vllm/

        start_time = time.perf_counter()
        while retries < self.max_retries:
            try:
                response = await self.agent_client.chat.completions.create(
                    model=vllm_llm_name,
                    messages=messages_,
                    timeout=self.llm_timeout,
                    temperature=temperature,
                    max_tokens=self.output_token_target,
                    extra_body={"min_tokens": self.output_token_target},
                )
                break
            except Exception as e:
                self.logger.error(f"LLM query failed @ {retries}: {e}")
                retries += 1
                if "RateLimitError" in str(e):
                    time.sleep(60)
                if retries >= self.max_retries:
                    raise e

        end_time = time.perf_counter()
        exec_time = end_time - start_time
        vllm_metrics = requests.get(
            self.llm_base_url.replace("/v1", "/metrics")
        ).text
        return response, exec_time, start_time, end_time, vllm_metrics

    async def _start_memory_compression(
        self,
        messages: List[Dict[str, str]],
    ):
        """Memory compression with controlled output token count.

        Same as parent but adds min_tokens/max_tokens control when
        memory_output_token_target is set. If not set, falls back to
        parent behavior (uses args.memory_model_max_gen_tokens as max_tokens only).
        """
        if self.memory_output_token_target is None:
            # No memory output control requested, use parent behavior
            return await super()._start_memory_compression(messages)

        call_params = {
            "model": self.args.memory_model_name,
            "messages": messages,
            "n": 1,
            "temperature": self.args.memory_model_temperature,
            "max_tokens": self.memory_output_token_target,
            "extra_body": {"min_tokens": self.memory_output_token_target},
        }

        start_time = time.perf_counter()
        response = await self.mem_client.chat.completions.create(**call_params)
        end_time = time.perf_counter()
        summary_exec_time = end_time - start_time
        response_text = response.choices[0].message.content.strip()
        finish_reason = response.choices[0].finish_reason

        summary_tokens = self._count_summary_tokens(response_text)

        return response_text, summary_tokens, summary_exec_time, finish_reason, start_time, end_time
