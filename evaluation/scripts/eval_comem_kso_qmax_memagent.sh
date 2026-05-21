#!/bin/bash

AWS_REGION_NAME="us-east-1"
MODEL="${MODEL:-hosted_vllm/Qwen/Qwen3-Coder-480B-A35B-Instruct}"

SKIP_EXISTING=${SKIP_EXISTING:-"True"}
MEM_MODEL_TOKENS=${MEM_MODEL_TOKENS:-2048}
RETAIN_MOST_TURNS=${RETAIN_MOST_TURNS:-2}
MAX_STEPS=${MAX_STEPS:-40}
MAX_STEPS_ABSOLUTE=${MAX_STEPS_ABSOLUTE:-100}
PROMPT_VERSION=${PROMPT_VERSION:-5}
MEMORY_CHUNK_SIZE=${MEMORY_CHUNK_SIZE:-5000}
MEMAGENT_MAX_CONTEXT_LEN=${MEMAGENT_MAX_CONTEXT_LEN:-32768}
MEMAGENT_ALLOW_LONG_MAX_MODEL_LEN=${MEMAGENT_ALLOW_LONG_MAX_MODEL_LEN:-0}

EXP_NAME=${EXP_NAME:-"comem-kso-qmax-memagent-iterative-t100-r${RETAIN_MOST_TURNS}"}

# RL-MemoryAgent backend + iterative proxy config
MEM_MODEL=${MEM_MODEL:-"BytedTsinghua-SIA/RL-MemoryAgent-7B"}
MEMAGENT_DEVICES=${MEMAGENT_DEVICES:-"4,5,6,7"}
MEMAGENT_PORT=${MEMAGENT_PORT:-9002}
MEMAGENT_PROXY_PORT=${MEMAGENT_PROXY_PORT:-9001}

export AWS_REGION_NAME="$AWS_REGION_NAME"

export LLM_BASE_URL="http://${ADDRESS:-localhost}:8000/v1"
export MEMORY_LLM_BASE_URL="http://localhost:${MEMAGENT_PROXY_PORT}/v1"

# ==== start servers ==== #
wait_for_servers() {
    local urls=("$@")
    local max_wait=${MAX_WAIT:-3000}
    local check_interval=5

    echo "Waiting for servers to be ready..."
    echo "URLs: ${urls[@]}"

    for url in "${urls[@]}"; do
        local elapsed=0
        echo -n "Checking $url... "

        while [ $elapsed -lt $max_wait ]; do
            if curl -s -f "$url" > /dev/null 2>&1; then
                echo "✓ Ready (${elapsed}s)"
                break
            fi

            sleep $check_interval
            elapsed=$((elapsed + check_interval))

            if [ $elapsed -ge $max_wait ]; then
                echo "✗ Timeout after ${max_wait}s"
                return 1
            fi
        done
    done

    echo "All servers are ready!"
    return 0
}

VLLM_MEM_PID=""
MEM_PROXY_PID=""

cleanup() {
    echo "Cleaning up background processes..."
    kill ${VLLM_MEM_PID} ${MEM_PROXY_PID} 2>/dev/null || true
}
trap cleanup EXIT

# 1) Start RL-MemoryAgent vLLM backend
PYTHON="/root/miniconda3/envs/r2e-gym/bin/python" \
DEVICES="${MEMAGENT_DEVICES}" \
MODEL="${MEM_MODEL}" \
PORT="${MEMAGENT_PORT}" \
MAX_CONTEXT_LEN="${MEMAGENT_MAX_CONTEXT_LEN}" \
VLLM_ALLOW_LONG_MAX_MODEL_LEN="${MEMAGENT_ALLOW_LONG_MAX_MODEL_LEN}" \
nohup bash scripts/start_vllm_server_rl_memoryagent_with_sync.sh > "${EXP_NAME}_vllm_memagent_backend.out" 2>&1 &
VLLM_MEM_PID=$!

# 2) Start iterative memory proxy on MEMORY_LLM_BASE_URL
MEMAGENT_BACKEND_MODEL="${MEM_MODEL}" \
MEMAGENT_BACKEND_API_URL="http://localhost:${MEMAGENT_PORT}/v1/chat/completions" \
MEMAGENT_PROXY_PORT="${MEMAGENT_PROXY_PORT}" \
MEMORY_CHUNK_SIZE="${MEMORY_CHUNK_SIZE}" \
MEMAGENT_MAX_TOKENS="${MEM_MODEL_TOKENS}" \
MEMAGENT_TEMPERATURE="0.0" \
MEMAGENT_FORCE_ITERATIVE="true" \
PYTHON="/root/miniconda3/envs/r2e-gym/bin/python" \
nohup bash scripts/start_memagent_iterative_proxy.sh > "${EXP_NAME}_memagent_iterative_proxy.out" 2>&1 &
MEM_PROXY_PID=$!

wait_for_servers \
  "http://${ADDRESS:-localhost}:8000/v1/models" \
  "http://localhost:${MEMAGENT_PORT}/v1/models" \
  "http://localhost:${MEMAGENT_PROXY_PORT}/v1/models"
# ==== start servers ==== #

# Run evaluation
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $($PYTHON -c 'import sys; print(sys.executable)')"
python src/r2egym/agenthub/run/edit_summ_kso.py runagent_multiple \
  --traj_dir "./traj" \
  --max_workers 64 \
  --start_idx 0 \
  --k 500 \
  --dataset "R2E-Gym/SWE-Bench-Verified" \
  --split "test" \
  --llm_name "$MODEL" \
  --use_fn_calling True \
  --backend "kubernetes" \
  --scaffold "r2egym" \
  --exp_name "$EXP_NAME" \
  --temperature 0.7 \
  --max_steps ${MAX_STEPS} \
  --use_existing ${SKIP_EXISTING} \
  --skip_existing False \
  --remove_api_exceptions True \
  --retain_most_turns ${RETAIN_MOST_TURNS} \
  --memory_model_name "${MEM_MODEL}" \
  --memory_model_temperature 0.0 \
  --memory_model_max_gen_tokens ${MEM_MODEL_TOKENS} \
  --max_steps_absolute ${MAX_STEPS_ABSOLUTE} \
  --max_reward_calc_time 1200 \
  --use_user_prompt True \
  --prompt_version ${PROMPT_VERSION}
