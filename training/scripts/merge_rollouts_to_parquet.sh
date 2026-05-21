set -x

# Parse options
SKIP_S3_SYNC=false
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --skip-s3-sync) SKIP_S3_SYNC=true; shift ;;
    *) break ;;
  esac
done

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

##### Sync rollout data from S3 #####
if [[ "$SKIP_S3_SYNC" == "false" ]]; then
  if command -v aws >/dev/null 2>&1; then
    ROLLOUT_BASE_S3="${ROLLOUT_BASE_S3:-s3://shopqa-users/yuwzhan/verl-agent-latest/outputs/rollouts}"

    # Define rollout directories to sync
    declare -a ROLLOUT_DIRS=(
      # "qwen3-8b/alfworld_30steps_32bs"
      # "qwen3-8b/webshop_small_20steps_32bs"
      # "qwen3-32b/alfworld_29steps_16bs"
      # "qwen3-32b/webshop_small_20steps_16bs"
      # "qwen3-32b/sciworld_25steps_32bs"

      # "grpo_full_memory_qwen3_8b_15steps_nopenalty_l1024_global_step_60/webshop_small_20steps_32bs"
      # "grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_60/alfworld_25steps_32bs"
      "grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_60/sciworld_25steps_32bs"

      # "grpo_main_no_iter_qwen3_4b_20steps_temp0_pv5_1536_sft_base_qwen3_8b_global_step_60/alfworld_25steps_32bs"

      # "grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_135/alfworld_20steps_32bs"
    )

    for ROLLOUT_DIR in "${ROLLOUT_DIRS[@]}"; do
      LOCAL_ROLLOUT_DIR="outputs/rollouts/${ROLLOUT_DIR}"
      S3_ROLLOUT_PREFIX="${ROLLOUT_BASE_S3}/${ROLLOUT_DIR}"

      echo "[merge_rollouts] Checking S3 prefix: ${S3_ROLLOUT_PREFIX}/"

      # Check if the prefix exists / has any objects
      SHOULD_SYNC=true
      if ! LS_OUT="$(aws s3 ls "${S3_ROLLOUT_PREFIX}/" 2>&1)"; then
        echo "[merge_rollouts] Can't access ${S3_ROLLOUT_PREFIX}/ (aws error below); skipping sync"
        echo "[merge_rollouts] ${LS_OUT}"
        SHOULD_SYNC=false
      elif [[ -z "${LS_OUT//[[:space:]]/}" ]]; then
        echo "[merge_rollouts] No objects found under ${S3_ROLLOUT_PREFIX}/ yet; skipping sync"
        SHOULD_SYNC=false
      fi

      if [[ "$SHOULD_SYNC" == "true" ]]; then
        mkdir -p "$LOCAL_ROLLOUT_DIR"
        echo "[merge_rollouts] Syncing ${S3_ROLLOUT_PREFIX}/ -> ${LOCAL_ROLLOUT_DIR}/"
        aws s3 sync "${S3_ROLLOUT_PREFIX}/" "${LOCAL_ROLLOUT_DIR}/" \
          --exclude "*.parquet" \
          || echo "[merge_rollouts] rollout sync failed for ${ROLLOUT_DIR}"
      fi
    done
  else
    echo "[merge_rollouts] AWS CLI not available; skipping S3 sync"
  fi
else
  echo "[merge_rollouts] Skipping S3 rollout sync (--skip-s3-sync flag set)"
fi
##### Sync rollout data from S3 #####

python -m pip install scikit-learn

# $PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/qwen3-8b/alfworld_30steps_32bs --output-file outputs/rollouts/qwen3-8b/alfworld_30steps_32bs/train.parquet --eval-output-file outputs/rollouts/qwen3-8b/alfworld_30steps_32bs/test.parquet --filter-active --filter-valid --max-steps 20 --train-records 3200 --eval-records 512 --data-source "qwen3-8b_alfworld_30steps_32bs" --end-iter 40
# $PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/qwen3-8b/webshop_small_20steps_32bs --output-file outputs/rollouts/qwen3-8b/webshop_small_20steps_32bs/train.parquet --eval-output-file outputs/rollouts/qwen3-8b/webshop_small_20steps_32bs/test.parquet --filter-active --filter-valid --max-steps 15 --train-records 3200 --eval-records 512 --data-source "qwen3-8b_webshop_small_20steps_32bs" --end-iter 40
# $PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/qwen3-8b/alfworld_30steps_32bs --output-file outputs/rollouts/qwen3-8b/alfworld_30steps_32bs/train_d100.parquet --eval-output-file outputs/rollouts/qwen3-8b/alfworld_30steps_32bs/test_d100.parquet --filter-active --filter-valid --max-steps 20 --train-records 4000 --eval-records 512 --data-source "qwen3-8b_alfworld_30steps_32bs"
# $PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/qwen3-8b/webshop_small_20steps_32bs --output-file outputs/rollouts/qwen3-8b/webshop_small_20steps_32bs/train_d100.parquet --eval-output-file outputs/rollouts/qwen3-8b/webshop_small_20steps_32bs/test_d100.parquet --filter-active --filter-valid --max-steps 15 --train-records 4000 --eval-records 512 --data-source "qwen3-8b_webshop_small_20steps_32bs"
# $PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/qwen3-32b/alfworld_29steps_16bs --output-file outputs/rollouts/qwen3-32b/alfworld_29steps_16bs/train.parquet --eval-output-file outputs/rollouts/qwen3-32b/alfworld_29steps_16bs/test.parquet --filter-active --filter-valid --max-steps 20 --train-records 1600 --eval-records 128 --data-source "qwen3-32b_alfworld_29steps_16bs" --end-iter 10
# $PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/qwen3-32b/webshop_small_20steps_16bs --output-file outputs/rollouts/qwen3-32b/webshop_small_20steps_16bs/train.parquet --eval-output-file outputs/rollouts/qwen3-32b/webshop_small_20steps_16bs/test.parquet --filter-active --filter-valid --max-steps 15 --train-records 1600 --eval-records 128 --data-source "qwen3-32b_webshop_small_20steps_16bs" --end-iter 10
# $PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/qwen3-32b/sciworld_25steps_32bs --output-file outputs/rollouts/qwen3-32b/sciworld_25steps_32bs/train.parquet --eval-output-file outputs/rollouts/qwen3-32b/sciworld_25steps_32bs/test.parquet --filter-active --filter-valid --max-steps 18 --train-records 1600 --eval-records 128 --data-source "qwen3-32b_sciworld_25steps_32bs" --end-iter 10

# $PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/grpo_full_memory_qwen3_8b_15steps_nopenalty_l1024_global_step_60/webshop_small_20steps_32bs --output-file outputs/rollouts/grpo_full_memory_qwen3_8b_15steps_nopenalty_l1024_global_step_60/webshop_small_20steps_32bs/train.parquet --eval-output-file outputs/rollouts/grpo_full_memory_qwen3_8b_15steps_nopenalty_l1024_global_step_60/webshop_small_20steps_32bs/test.parquet --filter-active --filter-valid --max-steps 15 --train-records 3200 --eval-records 512 --data-source "grpo_full_memory_qwen3_8b_15steps_nopenalty_l1024_global_step_60_webshop_small_20steps_32bs" --end-iter 50
# $PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_60/alfworld_25steps_32bs --output-file outputs/rollouts/grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_60/alfworld_25steps_32bs/train.parquet --eval-output-file outputs/rollouts/grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_60/alfworld_25steps_32bs/test.parquet --filter-active --filter-valid --max-steps 20 --train-records 3200 --eval-records 512 --data-source "grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_60_alfworld_25steps_32bs" --end-iter 50
$PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_60/sciworld_25steps_32bs --output-file outputs/rollouts/grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_60/sciworld_25steps_32bs/train.parquet --eval-output-file outputs/rollouts/grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_60/sciworld_25steps_32bs/test.parquet --filter-active --filter-valid --max-steps 18 --train-records 3200 --eval-records 512 --data-source "grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_60_sciworld_25steps_32bs" --end-iter 50

# $PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/grpo_main_no_iter_qwen3_4b_20steps_temp0_pv5_1536_sft_base_qwen3_8b_global_step_60/alfworld_25steps_32bs --output-file outputs/rollouts/grpo_main_no_iter_qwen3_4b_20steps_temp0_pv5_1536_sft_base_qwen3_8b_global_step_60/alfworld_25steps_32bs/train.parquet --eval-output-file outputs/rollouts/grpo_main_no_iter_qwen3_4b_20steps_temp0_pv5_1536_sft_base_qwen3_8b_global_step_60/alfworld_25steps_32bs/test.parquet --filter-active --filter-valid --max-steps 20 --train-records 640 --eval-records 128 --data-source "grpo_main_no_iter_qwen3_4b_20steps_temp0_pv5_1536_sft_base_qwen3_8b_global_step_60_alfworld_25steps_32bs" --end-iter 30

# $PYTHON scripts/merge_rollouts_to_parquet.py --input-dir outputs/rollouts/grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_135/alfworld_20steps_32bs --output-file outputs/rollouts/grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_135/alfworld_20steps_32bs/train.parquet --eval-output-file outputs/rollouts/grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_135/alfworld_20steps_32bs/test.parquet --filter-active --filter-valid --max-steps 20 --train-records 3200 --eval-records 512 --data-source "grpo_full_memory_qwen3_8b_20steps_nopenalty_global_step_135_alfworld_20steps_32bs" --end-iter 50 --min-steps 3