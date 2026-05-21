#!/bin/bash
# Test-only script for validating trained checkpoints

set -x

# Parse options
SKIP_S3_SYNC=false
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --checkpoint) CHECKPOINT_PATH="$2"; shift 2 ;;
    --skip-s3-sync) SKIP_S3_SYNC=true; shift ;;
    *) break ;;
  esac
done

ENGINE=${1:-vllm}
ADDRESS=${ADDRESS:-"0.0.0.0"}
PORT=${PORT:-"8000"}
export VLLM_ATTENTION_BACKEND=XFORMERS
DEVICES=${DEVICES:-"0,1,2,3,4,5,6,7"}
N_GPUS=$(echo "$DEVICES" | awk -F',' '{print NF}')

##### Add these for a fix on greenland #####
# Force the current conda env's Python
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please 'conda activate verl-agent' before running this script." >&2
  exit 1
fi

# Put the env's bin FIRST so it wins over base
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

wandb login cde3bf4dce4d89d49519e73eabf0196c798f8ee8
##### Add these for a fix on greenland #####

train_data_size=16
val_data_size=128
group_size=8
RUN_NAME="test_grpo_sum_no_iter_qwen3_4b_temp1_pv5_2048_sft_v2"
EXP_NAME="verl_agent_swebench_sum_v1"

##### download existing checkpoints #####
nohup bash scripts/sync_checkpoints.sh --verbose >"sync_s3.out" 2>&1 | tee sync_s3.out &
SYNC_PID=$!

# Set up trap to kill the sync process on script exit (normal or error)
trap "echo 'Killing sync process (PID: $SYNC_PID)...'; kill $SYNC_PID 2>/dev/null || true" EXIT

if [[ "$SKIP_S3_SYNC" == "false" ]]; then
  # Checkpoint sync if EXP_NAME/RUN_NAME present
  # Only sync the latest checkpoint folder based on VERL's marker file (a single number).
  # Sync ONLY latest checkpoint using latest_checkpointed_iteration.txt (numeric)
  if command -v aws >/dev/null 2>&1 && [[ -n "${EXP_NAME:-}" && -n "${RUN_NAME:-}" ]]; then
    CHECKPOINT_BASE_S3="${CHECKPOINT_BASE_S3:-s3://shopqa-users/yuwzhan/verl-agent-latest/checkpoints}"
    LOCAL_CHECKPOINT_DIR="checkpoints/${EXP_NAME}/${RUN_NAME}"
    S3_CHECKPOINT_PREFIX="${CHECKPOINT_BASE_S3}/${EXP_NAME}/${RUN_NAME}"
    MARKER_FILE="latest_checkpointed_iteration.txt"
    mkdir -p "$LOCAL_CHECKPOINT_DIR"
    
    # ---- new: check if the prefix exists / has any objects ----
    SHOULD_SYNC=true
    if ! LS_OUT="$(aws s3 ls "${S3_CHECKPOINT_PREFIX}/" 2>&1)"; then
      echo "[bootstrap] Can't access ${S3_CHECKPOINT_PREFIX}/ (aws error below); skipping sync"
      echo "[bootstrap] ${LS_OUT}"
      SHOULD_SYNC=false
    elif [[ -z "${LS_OUT//[[:space:]]/}" ]]; then
      echo "[bootstrap] No objects found under ${S3_CHECKPOINT_PREFIX}/ yet; skipping sync"
      SHOULD_SYNC=false
    fi
    # -----------------------------------------------------------
    
    if [[ "$SHOULD_SYNC" == "true" ]]; then
      STEP="$(aws s3 cp "${S3_CHECKPOINT_PREFIX}/${MARKER_FILE}" - 2>/dev/null | head -n1 | tr -d '\r\n[:space:]')"
      echo "[bootstrap] Syncing global_step_${STEP}/ from ${S3_CHECKPOINT_PREFIX} -> ${LOCAL_CHECKPOINT_DIR}"
      aws s3 sync "${S3_CHECKPOINT_PREFIX}/global_step_${STEP}/" "${LOCAL_CHECKPOINT_DIR}/global_step_${STEP}/" \
        || echo "[bootstrap] checkpoint sync failed"
      aws s3 cp "${S3_CHECKPOINT_PREFIX}/${MARKER_FILE}" "${LOCAL_CHECKPOINT_DIR}/${MARKER_FILE}" \
        || echo "[bootstrap] marker file sync failed"
    fi
  fi
else
  echo "[bootstrap] Skipping S3 checkpoint sync (--skip-s3-sync flag set)"
fi
##### download existing checkpoints #####

# download initial model checkpoints
if command -v aws >/dev/null 2>&1; then
    INITIAL_MODEL_S3="s3://shopqa-users/yuwzhan/verl-agent-latest/checkpoints/summary-sft/summary-sft-qwen3-4b/global_step_28"
    LOCAL_INITIAL_MODEL="checkpoints/summary-sft/summary-sft-qwen3-4b/global_step_28"

    # Check if the S3 path exists and has objects
    SHOULD_SYNC_INITIAL=true
    if ! LS_OUT="$(aws s3 ls "${INITIAL_MODEL_S3}/" 2>&1)"; then
        echo "[bootstrap] Can't access ${INITIAL_MODEL_S3}/ (aws error below); skipping initial model sync"
        echo "[bootstrap] ${LS_OUT}"
        SHOULD_SYNC_INITIAL=false
    elif [[ -z "${LS_OUT//[[:space:]]/}" ]]; then
        echo "[bootstrap] No objects found under ${INITIAL_MODEL_S3}/ yet; skipping initial model sync"
        SHOULD_SYNC_INITIAL=false
    fi

    if [[ "$SHOULD_SYNC_INITIAL" == "true" ]]; then
        echo "[bootstrap] Syncing initial model from ${INITIAL_MODEL_S3} -> ${LOCAL_INITIAL_MODEL}"
        mkdir -p "$LOCAL_INITIAL_MODEL"
        aws s3 sync "${INITIAL_MODEL_S3}/" "${LOCAL_INITIAL_MODEL}/" \
        || echo "[bootstrap] initial model sync failed"
    fi
fi

# Create output directory for test results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="outputs/test_results/${EXP_NAME}/${RUN_NAME}/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$DEVICES python3 scripts/generate_env_test.py \
    --config verl/trainer/config/ppo_multi_agent_trainer.yaml \
    --output-dir "$OUTPUT_DIR" \
    --overrides \
    algorithm.adv_estimator=grpo \
    data.train_files=outputs/rollouts/r2egym-training-trajectories-deepswe_train.parquet \
    data.val_files=outputs/rollouts/r2egym-training-trajectories-deepswe_test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=69632 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.prompt_key='trajectory_steps' \
    actor_rollout_ref.model.path=$LOCAL_INITIAL_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=73728 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    algorithm.use_kl_in_reward=False \
    env.rollout.n=$group_size \
    env.traj_collection_type="sumapiresprecentlitellm" \
    env.response_agent.address=["http://$ADDRESS:$PORT/v1"] \
    env.response_agent.model_name="hosted_vllm/agentica-org/DeepSWE-Preview" \
    env.response_agent.max_concurrency=128 \
    env.response_agent.max_executor_threads=128 \
    env.response_agent.per_endpoint_concurrency=64 \
    env.response_agent.timeout_s=1200 \
    env.response_agent.temperature=0.0 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    $@ | tee "$OUTPUT_DIR/test.log"

echo ""
echo "=========================================="
echo "Test completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
