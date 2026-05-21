#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"

PYTHON="/root/miniconda3/envs/r2e-gym/bin/python" DEVICES=0,1 KV_OFFLOADING_SIZE=500 nohup bash scripts/start_vllm_server_cpu.sh > "vllm_server.out" 2>&1 &
VLLM_PID=$!
# Set up trap to kill the server process on script exit (normal or error)
trap "pkill -P $VLLM_PID 2>/dev/null; kill $VLLM_PID 2>/dev/null || true" EXIT

# Get and format GPU name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | \
    sed 's/NVIDIA //g' | \
    tr '[:upper:]' '[:lower:]' | \
    sed 's/ /-/g' | \
    sed 's/gb/g/g')

wait_for_servers() {
    local urls=("$@")
    local max_wait=${MAX_WAIT:-600}  # Default 10 minutes
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

wait_for_servers "http://localhost:8000/v1/models"

batch_size=${batch_size:-32}
input_length=${input_length:-2048}
output_length=${output_length:-1024}
repeat=${repeat:-1}

PYTHON="$CONDA_PREFIX/bin/python" vllm bench serve \
  --backend vllm \
  --model agentica-org/DeepSWE-Preview \
  --endpoint /v1/completions \
  --dataset-name random \
  --num-prompts $batch_size \
  --host localhost \
  --port 8000 \
  --random-output-len $output_length \
  --random-input-len $input_length | tee vllm_bench_cpu_batch${batch_size}_inp${input_length}_outp${output_length}_repeat${repeat}.out
