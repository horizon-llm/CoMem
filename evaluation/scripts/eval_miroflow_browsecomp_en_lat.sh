#!/bin/bash
# =============================================================================
# BrowseComp-EN latency benchmark with full-context MiroFlowAgent
#
# Measures throughput and per-task latency of the full-context agent pipeline.
#
# Required env vars:
#   SERPER_API_KEY    - Serper.dev search API key
#   OPENAI_API_KEY    - For the judge model (gpt-4.1)
#
# Usage:
#   bash scripts/eval_miroflow_browsecomp_en_lat.sh
# =============================================================================
set -euo pipefail

export SERPER_API_KEY="${SERPER_API_KEY:?Error: SERPER_API_KEY must be set}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Error: OPENAI_API_KEY must be set}"

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

# ---------- Configuration ----------
DATA_PATH="${DATA_PATH:-data/browsecomp-128/standardized_data.jsonl}"
LLM_NAME="${LLM_NAME:-hosted_vllm/zai-org/GLM-4.7}"
LLM_BASE_URL="${LLM_BASE_URL:-http://${ADDRESS:-localhost}:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/browsecomp_en_lat_$(date +%Y%m%d_%H%M%S)}"

# Limits — higher concurrency for latency benchmarking
MAX_TASKS="${MAX_TASKS:-""}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_STEPS_ABS="${MAX_STEPS_ABS:-50}"
MAX_EXEC_TIME="${MAX_EXEC_TIME:-300}"
MAX_TOTAL_TIME="${MAX_TOTAL_TIME:-7200}"
MAX_CONCURRENT="${MAX_CONCURRENT:-128}"
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
    --max_context_tokens ${MAX_CONTEXT} \
    --latency_test True"

# Optional args
[ -n "${MAX_TASKS}" ]    && CMD="${CMD} --max_tasks ${MAX_TASKS}"
[ -n "${LLM_BASE_URL}" ] && CMD="${CMD} --llm_base_url ${LLM_BASE_URL}"

echo "============================================="
echo "BrowseComp-EN Full-Context Latency Benchmark"
echo "============================================="
echo "  LLM         : ${LLM_NAME}"
echo "  LLM URL     : ${LLM_BASE_URL}"
echo "  Concurrency : ${MAX_CONCURRENT}"
echo "  Judge       : ${JUDGE_MODEL}"
echo "  Data        : ${DATA_PATH}"
echo "  Output      : ${OUTPUT_DIR}"
echo "  Max tasks   : ${MAX_TASKS:-all}"
echo "============================================="

# Timed execution
BENCH_START=$(date +%s)
eval ${CMD}
BENCH_END=$(date +%s)
BENCH_ELAPSED=$((BENCH_END - BENCH_START))

# Write latency summary
LATENCY_FILE="${OUTPUT_DIR}/latency.txt"
TASK_COUNT=$(wc -l < "${OUTPUT_DIR}/results.jsonl" 2>/dev/null || echo 0)
TASK_COUNT=$(echo "$TASK_COUNT" | tr -d ' ')

echo "Total wall-clock time: ${BENCH_ELAPSED}s" | tee "$LATENCY_FILE"
if [ "$TASK_COUNT" -gt 0 ] 2>/dev/null; then
    AVG=$((BENCH_ELAPSED / TASK_COUNT))
    echo "Tasks completed: ${TASK_COUNT}" | tee -a "$LATENCY_FILE"
    echo "Average time per task: ${AVG}s" | tee -a "$LATENCY_FILE"
    echo "Throughput: $(echo "scale=2; $TASK_COUNT / $BENCH_ELAPSED * 3600" | bc) tasks/hour" | tee -a "$LATENCY_FILE"
fi
echo "Concurrency: ${MAX_CONCURRENT}" | tee -a "$LATENCY_FILE"
echo "Results: ${OUTPUT_DIR}/results.jsonl" | tee -a "$LATENCY_FILE"
