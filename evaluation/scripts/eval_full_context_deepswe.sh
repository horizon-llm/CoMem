AWS_REGION_NAME="us-east-2"
MODEL="hosted_vllm/agentica-org/DeepSWE-Preview"

SKIP_EXISTING=${SKIP_EXISTING:-"False"}
EXP_NAME="full-context-deepswe-eval-t100-w64"

export AWS_REGION_NAME="$AWS_REGION_NAME"
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

export LLM_BASE_URL="http://localhost:8000/v1"

echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

# To test with openhands scaffold, change --scaffold "r2egym" to "openhands"
python src/r2egym/agenthub/run/edit.py runagent_multiple \
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
  --use_existing "$SKIP_EXISTING" \
  --skip_existing False \
  --max_steps_absolute 100 \
  --max_reward_calc_time 1200 \
  --condense_history False
