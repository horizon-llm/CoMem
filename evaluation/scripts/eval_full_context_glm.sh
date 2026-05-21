AWS_REGION_NAME="us-east-2"
MODEL="hosted_vllm/zai-org/GLM-4.7"

SKIP_EXISTING=${SKIP_EXISTING:-"True"}
EXP_NAME="full-context-glm-eval-t100"

export AWS_REGION_NAME="$AWS_REGION_NAME"
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

export LLM_BASE_URL="http://localhost:8000/v1"

echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

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

KV_OFFLOADING_SIZE=500 PYTHON="/root/miniconda3/envs/r2e-gym/bin/python" DEVICES="0,1,2,3,4,5,6,7" nohup bash scripts/start_vllm_server_glm_cpu.sh > "vllm_server_glm.out" 2>&1 &
VLLM_PID=$!

# Set up trap to kill the server processes on script exit (normal or error)
trap "pkill -P $VLLM_PID 2>/dev/null; kill $VLLM_PID 2>/dev/null || true" EXIT

wait_for_servers "http://localhost:8000/v1/models"
# ==== start servers ==== #

# To test with openhands scaffold, change --scaffold "r2egym" to "openhands"
python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./traj" \
  --max_workers 32 \
  --start_idx 0 \
  --k 500 \
  --dataset "R2E-Gym/SWE-Bench-Verified" \
  --split "test" \
  --llm_name "$MODEL" \
  --use_fn_calling True \
  --backend "kubernetes" \
  --scaffold "r2egym" \
  --exp_name "$EXP_NAME" \
  --temperature 1.0 \
  --max_steps 40 \
  --use_existing "$SKIP_EXISTING" \
  --skip_existing False \
  --max_steps_absolute 100 \
  --max_reward_calc_time 1200
