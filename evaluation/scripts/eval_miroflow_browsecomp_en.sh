#!/bin/bash
# =============================================================================
# BrowseComp-EN evaluation using MiroFlowEnv + MiroFlowAgent
#
# Required env vars (export or put in .env):
#   SERPER_API_KEY    – Serper.dev search API key
#   OPENAI_API_KEY    – For the agent LLM and/or judge model
#
# Usage:
#   bash scripts/eval_miroflow_browsecomp_en.sh
# =============================================================================
set -euo pipefail

export SERPER_API_KEY="${SERPER_API_KEY:?Error: SERPER_API_KEY must be set}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Error: OPENAI_API_KEY must be set}"

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

# ---------- Configuration (edit as needed) ----------
DATA_PATH="${DATA_PATH:-data/browsecomp-128/standardized_data.jsonl}"
LLM_NAME="${LLM_NAME:-hosted_vllm/zai-org/GLM-4.7}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/browsecomp_en_$(date +%Y%m%d_%H%M%S)}"

MAX_TASKS="${MAX_TASKS:-""}"          # set to "" for all tasks
MAX_STEPS="${MAX_STEPS:-30}"
MAX_STEPS_ABS="${MAX_STEPS_ABS:-50}"
MAX_EXEC_TIME="${MAX_EXEC_TIME:-300}" # seconds per tool call
MAX_TOTAL_TIME="${MAX_TOTAL_TIME:-7200}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_CONTEXT="${MAX_CONTEXT:-128000}"

# ---------- Build command ----------
CMD="python -m r2egym.agenthub.run.edit_miroflow_browsecomp \
    --data_path ${DATA_PATH} \
    --llm_name ${LLM_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --max_steps ${MAX_STEPS} \
    --max_steps_absolute ${MAX_STEPS_ABS} \
    --max_exec_time ${MAX_EXEC_TIME} \
    --max_total_time ${MAX_TOTAL_TIME} \
    --max_concurrent ${MAX_CONCURRENT} \
    --temperature ${TEMPERATURE} \
    --judge_model ${JUDGE_MODEL} \
    --max_context_tokens ${MAX_CONTEXT}"

# Optional args
[ -n "${MAX_TASKS}" ]    && CMD="${CMD} --max_tasks ${MAX_TASKS}"
[ -n "${LLM_BASE_URL}" ] && CMD="${CMD} --llm_base_url ${LLM_BASE_URL}"

echo "============================================="
echo "BrowseComp-EN Evaluation"
echo "============================================="
echo "  LLM       : ${LLM_NAME}"
echo "  Judge     : ${JUDGE_MODEL}"
echo "  Data      : ${DATA_PATH}"
echo "  Output    : ${OUTPUT_DIR}"
echo "  Max tasks : ${MAX_TASKS:-all}"
echo "============================================="

eval ${CMD}
