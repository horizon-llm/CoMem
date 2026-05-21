#!/bin/bash
# Latency benchmark for COMEM-KSO agent
#
# Sweeps over input tokens, output tokens, memory output tokens, steps, and concurrency.
#
# Usage:
#   bash scripts/run_latency_benchmark.sh
#
# Environment variables (all optional, with defaults):
#   MODEL              - Agent model name (default: hosted_vllm/agentica-org/DeepSWE-Preview)
#   MEM_MODEL          - Memory/summarization model
#   NUM_INSTANCES      - Number of instances per config (default: 32)
#   INPUT_TARGETS      - Comma-separated input token targets (default: "2048,4096,8192,16384")
#   OUTPUT_TARGETS     - Comma-separated output token targets (default: "2048,4096,8192")
#   MEM_OUTPUT_TARGETS - Comma-separated memory output token targets (default: "-1" = use default)
#   STEPS_LIST         - Comma-separated step counts (default: "20")
#   WORKERS_LIST       - Comma-separated worker/concurrency counts (default: "32")
#   DEVICES            - GPU devices for agent model (default: "0,1,2,3")
#   MEM_DEVICES        - GPU devices for memory model (default: "4,5,6,7")

MODEL="${MODEL:-hosted_vllm/zai-org/GLM-4.7}"
MEM_MODEL="${MEM_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"

# Get and format GPU name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | \
    sed 's/NVIDIA //g' | \
    tr '[:upper:]' '[:lower:]' | \
    sed 's/ /-/g' | \
    sed 's/gb/g/g')
GPU_NAME="${GPU_NAME:-unknown-gpu}"

EXP_NAME="latency-bench-${GPU_NAME}-$(date +%Y%m%d_%H%M%S)"

export LLM_BASE_URL="http://${AGENT_ADDRESS:-localhost}:8000/v1"
export MEMORY_LLM_BASE_URL="http://${MEM_ADDRESS:-localhost}:9001/v1"

# ==== Server wait utility ==== #
wait_for_servers() {
    local urls=("$@")
    local max_wait=${MAX_WAIT:-1200}
    local check_interval=5

    echo "Waiting for servers to be ready..."
    echo "URLs: ${urls[@]}"

    for url in "${urls[@]}"; do
        local elapsed=0
        echo -n "Checking $url... "

        while [ $elapsed -lt $max_wait ]; do
            if curl -s -f "$url" > /dev/null 2>&1; then
                echo "Ready (${elapsed}s)"
                break
            fi

            sleep $check_interval
            elapsed=$((elapsed + check_interval))

            if [ $elapsed -ge $max_wait ]; then
                echo "Timeout after ${max_wait}s"
                return 1
            fi
        done
    done

    echo "All servers are ready!"
    return 0
}

wait_for_servers "${LLM_BASE_URL%/v1}/health" "${MEMORY_LLM_BASE_URL%/v1}/health" || exit 1

# ==== Run benchmark ==== #
NUM_INSTANCES=${NUM_INSTANCES:-16}
INPUT_TARGETS="${INPUT_TARGETS:-512}"
OUTPUT_TARGETS="${OUTPUT_TARGETS:-1024}"
MEM_OUTPUT_TARGETS="${MEM_OUTPUT_TARGETS:-2048}"
STEPS_LIST="${STEPS_LIST:-64}"
WORKERS_LIST="${WORKERS_LIST:-16}"

echo ""
echo "=============================================="
echo "COMEM-KSO Latency Benchmark"
echo "=============================================="
echo "Model:              $MODEL"
echo "Memory Model:       $MEM_MODEL"
echo "GPU:                $GPU_NAME"
echo "Instances/cfg:      $NUM_INSTANCES"
echo "Input targets:      $INPUT_TARGETS"
echo "Output targets:     $OUTPUT_TARGETS"
echo "Mem output targets: $MEM_OUTPUT_TARGETS"
echo "Steps list:         $STEPS_LIST"
echo "Workers list:       $WORKERS_LIST"
echo "=============================================="
echo ""

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
python src/r2egym/agenthub/run/edit_benchmark.py run_benchmark_grid \
    --input_token_targets "$INPUT_TARGETS" \
    --output_token_targets "$OUTPUT_TARGETS" \
    --memory_output_token_targets "$MEM_OUTPUT_TARGETS" \
    --max_steps_list "$STEPS_LIST" \
    --max_workers_list "$WORKERS_LIST" \
    --num_instances $NUM_INSTANCES \
    --llm_name "$MODEL" \
    --memory_model_name "$MEM_MODEL" \
    --memory_model_temperature 0.0 \
    --memory_model_max_gen_tokens 2048 \
    --retain_most_turns 3 \
    --use_optimized_schedule False \
    --traj_dir "./traj_benchmark" \
    --exp_name "$EXP_NAME" \
    --latency_test True \
    --use_user_prompt True \
    --prompt_version 5
