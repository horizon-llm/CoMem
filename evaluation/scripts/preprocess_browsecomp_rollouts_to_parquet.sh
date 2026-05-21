set -x

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

python -m pip install scikit-learn

# ---------- Configuration ----------
INPUT_DIR="${1:?Usage: bash scripts/preprocess_browsecomp_rollouts_to_parquet.sh <input_dir>}"
OUTPUT_NAME="${2:-browsecomp}"

MIN_STEPS="${MIN_STEPS:-2}"
MAX_STEPS="${MAX_STEPS:-30}"
TRAIN_RECORDS="${TRAIN_RECORDS:-3200}"
EVAL_RECORDS="${EVAL_RECORDS:-512}"

mkdir -p outputs/rollouts

$PYTHON scripts/preprocess_browsecomp_rollouts_to_parquet.py \
  --input-path "${INPUT_DIR}" \
  --output-file "outputs/rollouts/${OUTPUT_NAME}_train_max${MAX_STEPS}.parquet" \
  --eval-output-file "outputs/rollouts/${OUTPUT_NAME}_test_max${MAX_STEPS}.parquet" \
  --train-records ${TRAIN_RECORDS} --eval-records ${EVAL_RECORDS} \
  --min-steps ${MIN_STEPS} --max-steps ${MAX_STEPS}
