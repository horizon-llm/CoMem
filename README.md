<h1 align="center">CoMem</h1>
<p align="center"><em>Context Management with A Decoupled Long-Context Model</em></p>

`CoMem` is an efficient implementation of _context management for LLM agents_ built on [verl-agent](https://github.com/langfengQ/verl-agent) and [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym).

# News
- [2026.05.26] Code released.

# Installation
We provide the following docker environment:
```bash
docker run --gpus all --shm-size=64g --rm -it --net=host \
 --entrypoint /usr/bin/bash \
 brandonzyw/comem:v1
```
The docker image includes two conda environments:
- `verl-agent` — for training (`conda activate verl-agent`)
- `r2e-gym` — for evaluation (`conda activate r2e-gym`)

# Training

We provide training scripts for the CoMem context compressor in [`training/examples/`](training/examples/).

Before running, set your Weights & Biases API key:
```bash
export WANDB_API_KEY=<your-wandb-api-key>
```

## SFT (Warm-up)
```bash
bash training/examples/sft_sum_trainer/run_qwen3_4b.sh
```

## GRPO (RL Fine-tuning)
```bash
# SWE-bench with GLM-4.7 as the agent LLM
bash training/examples/grpo_sum_trainer/run_swebench_qwen3_4b_glm_pv5_2048_v2_grp16.sh
```
```bash
# SWE-bench with DeepSWE as the agent LLM
bash training/examples/grpo_sum_trainer/run_swebench_qwen3_4b_deepswe_pv5_2048_v2_grp16.sh
```
```bash
# SWE-bench with Qwen3-Coder-480B as the agent LLM
bash training/examples/grpo_sum_trainer/run_swebench_qwen3_4b_qmax_pv5_2048_v2_grp16.sh
```
```bash
# BrowseComp with GLM-4.7 as the agent LLM
bash training/examples/grpo_sum_trainer/run_browsecomp_qwen3_4b_glm_pv5_2048_v2_grp16.sh
```

# Evaluation

We provide evaluation scripts in [`evaluation/scripts/`](evaluation/scripts/).

## Step 1: Start the Agent LLM Server
```bash
bash evaluation/scripts/start_vllm_server_glm.sh      # GLM-4.7
bash evaluation/scripts/start_vllm_server_deepswe.sh   # DeepSWE
bash evaluation/scripts/start_vllm_server_qmax.sh      # Qwen3-Coder-480B
```

## Step 2 (CoMem only): Start the Memory Model Server
```bash
bash evaluation/scripts/start_vllm_server_mem.sh       # Memory model (Qwen3-4B compressor)
```

## Step 3: Run Evaluation

### CoMem (Ours)
```bash
bash evaluation/scripts/eval_comem_kso_glm.sh          # SWE-bench + GLM-4.7
bash evaluation/scripts/eval_comem_kso_deepswe.sh      # SWE-bench + DeepSWE
bash evaluation/scripts/eval_comem_kso_qmax.sh         # SWE-bench + Qwen3-Coder-480B
```

### Full-Context Baseline
```bash
bash evaluation/scripts/eval_full_context_glm.sh       # SWE-bench + GLM-4.7
bash evaluation/scripts/eval_full_context_deepswe.sh   # SWE-bench + DeepSWE
bash evaluation/scripts/eval_full_context_qmax.sh      # SWE-bench + Qwen3-Coder-480B
```

### BrowseComp

Set required environment variables:
```bash
export SERPER_API_KEY=<your-serper-api-key>
export OPENAI_API_KEY=<your-openai-api-key>
```
```bash
bash evaluation/scripts/eval_comem_miroflow_browsecomp_en.sh  # CoMeM + GLM-4.7
bash evaluation/scripts/eval_miroflow_browsecomp_en.sh        # Full-context + GLM-4.7
```

### Latency Benchmarks
```bash
# CoMem latency
bash evaluation/scripts/eval_comem_kso_glm_lat.sh
bash evaluation/scripts/eval_comem_kso_deepswe_lat.sh
bash evaluation/scripts/eval_comem_kso_qmax_lat.sh

# CoMem latency with CPU KV offloading
bash evaluation/scripts/eval_comem_kso_glm_lat_cpuoffload.sh
bash evaluation/scripts/eval_comem_kso_deepswe_lat_cpuoffload.sh
bash evaluation/scripts/eval_comem_kso_qmax_lat_cpuoffload.sh

# Full-context latency
bash evaluation/scripts/eval_full_context_glm_lat.sh
bash evaluation/scripts/eval_full_context_deepswe_lat.sh
bash evaluation/scripts/eval_full_context_qmax_lat.sh

# Full-context latency with CPU KV offloading
bash evaluation/scripts/eval_full_context_glm_lat_cpu.sh
bash evaluation/scripts/eval_full_context_deepswe_lat_cpu.sh
bash evaluation/scripts/eval_full_context_qmax_lat_cpu.sh
```

# Acknowledgement

We gratefully acknowledge the contributions of the [veRL](https://github.com/volcengine/verl) team for providing a solid RL infrastructure.

Special thanks to the [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym)  and [verl-agent](https://github.com/langfengQ/verl-agent) project for their codebase, which inspired early design choices during the development of `CoMem`.

# Citation

If you find `CoMem` useful in your research or applications, we would appreciate it if you could cite our work:

```
@inproceedings{zhang2026comem,
  title={CoMem: Context Management with A Decoupled Long-Context Model},
  author={Zhang, Yuwei and Dong, Chengyu and Jin, Shuowei and Yu, Changlong and Cui, Hejie and Jin, Hongye and Zhang, Xinyang and Bonab, Hamed and Lockard, Colin and Chen, Jianshu and others},
  booktitle={ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems}
}
```

We're excited to share our early results and welcome feedback from the community as we continue to refine and expand CoMem’s capabilities. If you have any questions or feedback, please feel free to contact us at [yuz163@ucsd.edu](mailto:yuz163@ucsd.edu).