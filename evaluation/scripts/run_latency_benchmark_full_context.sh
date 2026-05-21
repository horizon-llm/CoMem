#!/bin/bash
# Latency benchmark for full-context agent (no memory/summarization)
#
# Sweeps over input tokens, output tokens, steps, and concurrency.
# Assumes the vLLM server is already running; waits for it to be ready.
#
# Usage:
#   bash scripts/run_latency_benchmark_full_context.sh
#
# Environment variables (all optional, with defaults):
#   MODEL              - Agent model name (default: hosted_vllm/agentica-org/DeepSWE-Preview)
#   NUM_INSTANCES      - Number of instances per config (default: 32)
#   INPUT_TARGETS      - Comma-separated input token targets (default: "512")
#   OUTPUT_TARGETS     - Comma-separated output token targets (default: "1024")
#   STEPS_LIST         - Comma-separated step counts (default: "32")
#   WORKERS_LIST       - Comma-separated worker/concurrency counts (default: "32")

MODEL="${MODEL:-hosted_vllm/zai-org/GLM-4.7}"

# Get and format GPU name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | \
    sed 's/NVIDIA //g' | \
    tr '[:upper:]' '[:lower:]' | \
    sed 's/ /-/g' | \
    sed 's/gb/g/g')
GPU_NAME="${GPU_NAME:-unknown-gpu}"

EXP_NAME="fc-latency-bench-${GPU_NAME}-$(date +%Y%m%d_%H%M%S)"

export LLM_BASE_URL="http://${AGENT_ADDRESS:-localhost}:8000/v1"

# ==== Server wait utility ==== #
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

wait_for_servers "${LLM_BASE_URL%/v1}/health" || exit 1

# ==== Run benchmark ==== #
NUM_INSTANCES=${NUM_INSTANCES:-16}
INPUT_TARGETS="${INPUT_TARGETS:-512}"
OUTPUT_TARGETS="${OUTPUT_TARGETS:-1024}"
STEPS_LIST="${STEPS_LIST:-64}"
WORKERS_LIST="${WORKERS_LIST:-16}"

echo ""
echo "=============================================="
echo "Full-Context Latency Benchmark"
echo "=============================================="
echo "Model:              $MODEL"
echo "GPU:                $GPU_NAME"
echo "Instances/cfg:      $NUM_INSTANCES"
echo "Input targets:      $INPUT_TARGETS"
echo "Output targets:     $OUTPUT_TARGETS"
echo "Steps list:         $STEPS_LIST"
echo "Workers list:       $WORKERS_LIST"
echo "=============================================="
echo ""

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
python src/r2egym/agenthub/run/edit_benchmark_full_context.py run_benchmark_grid \
    --input_token_targets "$INPUT_TARGETS" \
    --output_token_targets "$OUTPUT_TARGETS" \
    --max_steps_list "$STEPS_LIST" \
    --max_workers_list "$WORKERS_LIST" \
    --num_instances $NUM_INSTANCES \
    --llm_name "$MODEL" \
    --traj_dir "./traj_benchmark" \
    --exp_name "$EXP_NAME" \
    --latency_test True
