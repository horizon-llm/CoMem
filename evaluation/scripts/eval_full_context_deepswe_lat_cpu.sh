AWS_REGION_NAME="us-east-2"
MODEL="hosted_vllm/agentica-org/DeepSWE-Preview"

# Get and format GPU name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | \
    sed 's/NVIDIA //g' | \
    tr '[:upper:]' '[:lower:]' | \
    sed 's/ /-/g' | \
    sed 's/gb/g/g')

EXP_NAME="full-context-deepswe-eval-t100-lat-${GPU_NAME}"

export AWS_REGION_NAME="$AWS_REGION_NAME"

export LLM_BASE_URL="http://localhost:8000/v1"

# ==== start servers ==== #
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

PYTHON="/root/miniconda3/envs/r2e-gym/bin/python" DEVICES="0,1,2,3" KV_OFFLOADING_SIZE=500 nohup bash scripts/start_vllm_server_cpu.sh > "vllm_server.out" 2>&1 &
VLLM_PID=$!

# Set up trap to kill the server processes on script exit (normal or error)
trap "pkill -P $VLLM_PID 2>/dev/null; kill $VLLM_PID 2>/dev/null || true" EXIT

wait_for_servers "http://localhost:8000/v1/models"
# ==== start servers ==== #

# Run evaluation
MAX_WORKERS=${MAX_WORKERS:-128}
DATA_SEED=${DATA_SEED:-42}
NUM_SAMPLES=${NUM_SAMPLES:-128}

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
# To test with openhands scaffold, change --scaffold "r2egym" to "openhands"
python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./traj" \
  --max_workers $MAX_WORKERS \
  --data_seed $DATA_SEED \
  --start_idx 0 \
  --k $NUM_SAMPLES \
  --dataset "R2E-Gym/SWE-Bench-Verified" \
  --split "test" \
  --llm_name "$MODEL" \
  --use_fn_calling False \
  --backend "kubernetes" \
  --scaffold "r2egym" \
  --exp_name "$EXP_NAME-tp4cpu-w${MAX_WORKERS}-d${DATA_SEED}-n${NUM_SAMPLES}" \
  --temperature 1.0 \
  --max_steps 40 \
  --use_existing False \
  --skip_existing False \
  --max_steps_absolute 100 \
  --max_reward_calc_time 1200 \
  --latency_test True
