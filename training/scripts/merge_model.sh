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
##### Add these for a fix on greenland #####
LOCAL_DIR=${LOCAL_DIR:-"checkpoints/verl_agent_alfworld_v3/grpo_trunc1_qwen3_8b_20steps_nopenalty/global_step_150/actor"}
python scripts/model_merger.py merge --backend fsdp --local_dir ${LOCAL_DIR} --target_dir ${LOCAL_DIR}/huggingface