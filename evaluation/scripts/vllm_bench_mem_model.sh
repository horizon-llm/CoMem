#!/usr/bin/env bash
# Benchmark any memory model's max throughput.
#
# Usage:
#   MEM_MODEL=Qwen/Qwen3-8B bash scripts/vllm_bench_mem_model.sh
#   MEM_MODEL=Qwen/Qwen3-32B DEVICES=0,1,2,3,4,5,6,7 bash scripts/vllm_bench_mem_model.sh
#
# Environment variables:
#   MEM_MODEL        - Model name (required, e.g. Qwen/Qwen3-4B-Instruct-2507, Qwen/Qwen3-8B, Qwen/Qwen3-32B)
#   DEVICES          - GPU devices (default: 4,5,6,7)
#   KV_OFFLOADING_SIZE - CPU KV offloading size in GB (default: 500)
#   MAX_CONTEXT_LEN  - Max context length (default: 65536)
#   batch_size       - Number of concurrent requests (default: 32)
#   input_length     - Input token length (default: 28206, matching real COMEM workload)
#   output_length    - Output token length (default: 1041, matching real COMEM workload)
#   repeat           - Repeat index for filename (default: 1)

export PATH="$HOME/.local/bin:$PATH"

MEM_MODEL="${MEM_MODEL:?ERROR: Set MEM_MODEL (e.g. Qwen/Qwen3-8B)}"
DEVICES="${DEVICES:-4,5,6,7}"
KV_OFFLOADING_SIZE="${KV_OFFLOADING_SIZE:-500}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-40960}"

# Derive a short model tag for filenames (e.g. "qwen3-4b-instruct-2507")
MODEL_TAG=$(echo "$MEM_MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')

# Start vLLM server
KV_OFFLOADING_SIZE=$KV_OFFLOADING_SIZE PYTHON="${PYTHON:-/root/miniconda3/envs/r2e-gym/bin/python}" \
  DEVICES="$DEVICES" MODEL="$MEM_MODEL" MAX_CONTEXT_LEN="$MAX_CONTEXT_LEN" \
  nohup bash scripts/start_vllm_server_with_sync_cpu.sh > "vllm_mem_model_server.out" 2>&1 &
VLLM_PID=$!
trap "pkill -P $VLLM_PID 2>/dev/null; kill $VLLM_PID 2>/dev/null || true" EXIT

# Get GPU name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | \
    sed 's/NVIDIA //g' | tr '[:upper:]' '[:lower:]' | sed 's/ /-/g' | sed 's/gb/g/g')
GPU_NAME="${GPU_NAME:-unknown-gpu}"

wait_for_servers() {
    local urls=("$@")
    local max_wait=${MAX_WAIT:-600}
    local check_interval=5
    echo "Waiting for servers to be ready..."
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
}

wait_for_servers "http://localhost:9001/v1/models" || exit 1

batch_size=${batch_size:-32}
input_length=${input_length:-28206}
output_length=${output_length:-1041}
repeat=${repeat:-1}

OUTDIR="benchmarks-${MODEL_TAG}-${GPU_NAME}"
mkdir -p "$OUTDIR"
OUTFILE="${OUTDIR}/vllm_bench_batch${batch_size}_inp${input_length}_outp${output_length}_repeat${repeat}.out"

echo "=============================================="
echo "Memory Model Throughput Benchmark"
echo "=============================================="
echo "Model:       $MEM_MODEL"
echo "Devices:     $DEVICES"
echo "GPU:         $GPU_NAME"
echo "Batch size:  $batch_size"
echo "Input len:   $input_length"
echo "Output len:  $output_length"
echo "Output file: $OUTFILE"
echo "=============================================="

vllm bench serve \
  --backend vllm \
  --model "$MEM_MODEL" \
  --endpoint /v1/completions \
  --dataset-name random \
  --num-prompts $batch_size \
  --host localhost \
  --port 9001 \
  --random-output-len $output_length \
  --random-input-len $input_length | tee "$OUTFILE"

