AWS_REGION_NAME="us-east-2"
MODEL="hosted_vllm/agentica-org/DeepSWE-Preview"

export AWS_REGION_NAME="$AWS_REGION_NAME"
export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

export LLM_BASE_URL="http://localhost:8000/v1"

echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

EXP_NAME="r2egym-training-trajectories-deepswe-t100-w16"

python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "./traj" \
  --max_workers 16 \
  --start_idx 0 \
  --k 2000 \
  --dataset "R2E-Gym/R2E-Gym-Lite" \
  --split "train" \
  --llm_name "$MODEL" \
  --use_fn_calling False \
  --backend "kubernetes" \
  --scaffold "r2egym" \
  --exp_name $EXP_NAME \
  --temperature 1.0 \
  --use_existing True \
  --skip_existing False \
  --max_steps 40 \
  --max_steps_absolute 100 \
  --max_reward_calc_time 1200
