set -x

INPUT_FILE=${1:?"Usage: $0 <input-file-name> (e.g. r2egym-training-trajectories-deepswe-t100-w16)"}

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

mkdir -p traj

python -m pip install scikit-learn

hf download YWZBrandon/r2e-gym-train-trajectories ${INPUT_FILE}.jsonl --repo-type dataset --local-dir ./traj

$PYTHON scripts/preprocess_swebench_rollouts_to_parquet.py \
  --input-jsonl "traj/${INPUT_FILE}.jsonl" \
  --output-file "outputs/rollouts/${INPUT_FILE}_train_max40.parquet" \
  --eval-output-file "outputs/rollouts/${INPUT_FILE}_test_max40.parquet" \
  --train-records 3200 --eval-records 512 \
  --data-source "${INPUT_FILE}" \
  --min-steps 2 --max-steps 40
