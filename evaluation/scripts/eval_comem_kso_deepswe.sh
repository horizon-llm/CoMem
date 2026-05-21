# AWS_REGION_NAME="us-east-2"
MODEL="hosted_vllm/agentica-org/DeepSWE-Preview"

SKIP_EXISTING=${SKIP_EXISTING:-"True"}
EXP_NAME="comem-grpo-qwen3-4b-grp16-l2048-r1-deepswe-sweverified-eval-t100-kso-r2"

export AWS_REGION_NAME="$AWS_REGION_NAME"

export LLM_BASE_URL="http://localhost:8000/v1"
export MEMORY_LLM_BASE_URL="http://localhost:9001/v1"

# ==== download model ==== #
# To switch to SFT model, uncomment below and comment out the GRPO model:
# MEM_MODEL="checkpoints/summary-sft-qwen3-4b"
# HF_MODEL="YWZBrandon/summary-sft-qwen3-4b"
MEM_MODEL="checkpoints/verl_agent_swebench_t100_sum_v1_grpo_qwen3_4b_temp1_pv5_2048_sft_v2_grp16_s150"
HF_MODEL="YWZBrandon/verl_agent_swebench_t100_sum_v1_grpo_qwen3_4b_temp1_pv5_2048_sft_v2_grp16_s150"
if [ ! -d "$MEM_MODEL" ]; then
    hf download $HF_MODEL --local-dir "$MEM_MODEL"
fi

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

PYTHON="/root/miniconda3/envs/verl-agent/bin/python" DEVICES="0,1,2,3" nohup bash scripts/start_vllm_server.sh > "vllm_server.out" 2>&1 &
VLLM_PID=$!
PYTHON="/root/miniconda3/envs/verl-agent/bin/python" DEVICES="4,5,6,7" MODEL=$MEM_MODEL nohup bash scripts/start_vllm_server_with_sync.sh > "vllm_mem_model_server.out" 2>&1 &
VLLM_MEM_PID=$!

# Set up trap to kill the server processes on script exit (normal or error)
trap "pkill -P $VLLM_PID $VLLM_MEM_PID $SYNC_PID 2>/dev/null; kill $VLLM_PID $VLLM_MEM_PID $SYNC_PID 2>/dev/null || true" EXIT

wait_for_servers "http://localhost:8000/v1/models" "http://localhost:9001/v1/models"

# ==== start servers ==== #

# Run evaluation
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
python src/r2egym/agenthub/run/edit_summ.py runagent_multiple \
  --traj_dir "./traj" \
  --max_workers 32 \
  --start_idx 0 \
  --k 500 \
  --dataset "R2E-Gym/SWE-Bench-Verified" \
  --split "test" \
  --llm_name "$MODEL" \
  --use_fn_calling False \
  --backend "kubernetes" \
  --scaffold "r2egym" \
  --exp_name "$EXP_NAME" \
  --temperature 1.0 \
  --max_steps 40 \
  --use_existing ${SKIP_EXISTING} \
  --skip_existing False \
  --remove_api_exceptions True \
  --retain_most_turns 2 \
  --memory_model_name "$MEM_MODEL" \
  --memory_model_temperature 0.0 \
  --memory_model_max_gen_tokens 2048 \
  --max_steps_absolute 100 \
  --max_reward_calc_time 1200
# To test without summary (nosum), add the following flag:
#   --nosummary True
