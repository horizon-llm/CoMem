AWS_REGION_NAME="us-east-2"
MODEL="hosted_vllm/Qwen/Qwen3-Coder-480B-A35B-Instruct"

SKIP_EXISTING=${SKIP_EXISTING:-"True"}
MEM_MODEL_TOKENS=${MEM_MODEL_TOKENS:-2048}
RETAIN_MOST_TURNS=${RETAIN_MOST_TURNS:-2}
MAX_STEPS=${MAX_STEPS:-40}
MAX_STEPS_ABSOLUTE=${MAX_STEPS_ABSOLUTE:-100}
PROMPT_VERSION=${PROMPT_VERSION:-1}

EXP_NAME="comem-grpo-grp16-max${MAX_STEPS}-rewardv2-s35-l${MEM_MODEL_TOKENS}-r1-qmax-newquerypromptv${PROMPT_VERSION}-sweverified-eval-t${MAX_STEPS_ABSOLUTE}-kso-r${RETAIN_MOST_TURNS}"

export AWS_REGION_NAME="$AWS_REGION_NAME"

export LLM_BASE_URL="http://${ADDRESS:-localhost}:8000/v1"
export MEMORY_LLM_BASE_URL="http://localhost:9001/v1"

# ==== download model ==== #
# To switch to SFT model, uncomment below and comment out the GRPO model:
# MEM_MODEL="checkpoints/summary-sft-qwen3-4b"
# HF_MODEL="YWZBrandon/summary-sft-qwen3-4b"
MEM_MODEL="checkpoints/verl_agent_swebench_sum_reward_v2_grpo_qwen3_4b_qmax_pv5_2048_sft_v2_grp16_max40_s35"
HF_MODEL="YWZBrandon/verl_agent_swebench_sum_reward_v2_grpo_qwen3_4b_qmax_pv5_2048_sft_v2_grp16_max40_s35"
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

PYTHON="/root/miniconda3/envs/r2e-gym/bin/python" DEVICES="4,5,6,7" MODEL=$MEM_MODEL MAX_CONTEXT_LEN=262144 nohup bash scripts/start_vllm_server_with_sync.sh > "${EXP_NAME}_vllm_mem_model.out" 2>&1 &
VLLM_MEM_PID=$!

# Set up trap to kill the server processes on script exit (normal or error)
trap "pkill -P $VLLM_MEM_PID 2>/dev/null; kill $VLLM_MEM_PID 2>/dev/null || true" EXIT

wait_for_servers "http://${ADDRESS:-localhost}:8000/v1/models" "http://localhost:9001/v1/models"

# ==== start servers ==== #

# Run evaluation
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
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
  --memory_model_name "$MEM_MODEL" \
  --memory_model_temperature 0.0 \
  --memory_model_max_gen_tokens ${MEM_MODEL_TOKENS} \
  --max_steps_absolute ${MAX_STEPS_ABSOLUTE} \
  --max_reward_calc_time 1200 \
  --use_user_prompt True \
  --prompt_version ${PROMPT_VERSION}
# To test without summary (nosum), add the following flag:
#   --nosummary True
