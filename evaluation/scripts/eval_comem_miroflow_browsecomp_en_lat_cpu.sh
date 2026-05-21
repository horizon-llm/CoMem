#!/bin/bash
# =============================================================================
# BrowseComp-EN latency benchmark with CoMeM (compressed memory)
#
# Mirrors eval_comem_kso_opensource_t100_lat_tp4_cpu_glm.sh but for BrowseComp.
# Measures throughput and per-task latency of the CoMeM agent pipeline.
#
# Features:
#   - Starts memory model vLLM server (with CPU KV offloading)
#   - Waits for all servers to be ready before evaluation
#   - Records wall-clock time and writes latency summary
#
# Required env vars:
#   SERPER_API_KEY    - Serper.dev search API key
#   OPENAI_API_KEY    - For the judge model (gpt-4.1)
#
# Usage:
#   bash scripts/eval_comem_miroflow_browsecomp_en_lat.sh
# =============================================================================
set -euo pipefail

export SERPER_API_KEY="${SERPER_API_KEY:?Error: SERPER_API_KEY must be set}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Error: OPENAI_API_KEY must be set}"

export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"
echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

# ---------- Configuration ----------
DATA_PATH="${DATA_PATH:-data/browsecomp-128/standardized_data.jsonl}"

# Agent LLM (assumed already running at ADDRESS:8000)
LLM_NAME="${LLM_NAME:-hosted_vllm/zai-org/GLM-4.7}"
LLM_BASE_URL="${LLM_BASE_URL:-http://${ADDRESS:-localhost}:8000/v1}"

# Memory model (summarizer) — should match the model used during training
MEM_MODEL="${MEM_MODEL:-checkpoints/verl_agent_browsecomp_sum_reward_v2/grpo_sum_qwen3_4b_browsecomp_glm_pv5_2048_v2_grp16_max30/global_step_35/actor}"
MEMORY_MODEL_NAME="${MEMORY_MODEL_NAME:-}"
MEMORY_MODEL_ADDRESS="${MEMORY_MODEL_ADDRESS:-http://localhost:9001/v1}"
MEMORY_MODEL_TEMPERATURE="${MEMORY_MODEL_TEMPERATURE:-0.0}"
MEMORY_MODEL_MAX_GEN_TOKENS="${MEMORY_MODEL_MAX_GEN_TOKENS:-2048}"
RETAIN_MOST_TURNS="${RETAIN_MOST_TURNS:-1}"

# Memory model server settings
MEM_DEVICES="${MEM_DEVICES:-4,5,6,7}"
MEM_PORT="${MEM_PORT:-9001}"
MEM_KV_OFFLOADING_SIZE="${MEM_KV_OFFLOADING_SIZE:-500}"

# Judge
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-logs/browsecomp_comem_lat_r${RETAIN_MOST_TURNS}}"

# Limits — higher concurrency for latency benchmarking
MAX_TASKS="${MAX_TASKS:-""}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_STEPS_ABS="${MAX_STEPS_ABS:-50}"
MAX_EXEC_TIME="${MAX_EXEC_TIME:-300}"
MAX_TOTAL_TIME="${MAX_TOTAL_TIME:-7200}"
MAX_CONCURRENT="${MAX_CONCURRENT:-128}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_CONTEXT="${MAX_CONTEXT:-128000}"

# ==== wait_for_servers helper ====
wait_for_servers() {
    local urls=("$@")
    local max_wait=${MAX_WAIT:-3000}
    local check_interval=5

    echo "Waiting for servers to be ready..."
    echo "URLs: ${urls[@]}"

    for url in "${urls[@]}"; do
        local elapsed=0
        echo -n "Checking $url... "

        while [ $elapsed -lt $max_wait ]; do
            if curl -s -f "$url" > /dev/null 2>&1; then
                echo "Ready (${elapsed}s)"
                break
            fi

            sleep $check_interval
            elapsed=$((elapsed + check_interval))

            if [ $elapsed -ge $max_wait ]; then
                echo "Timeout after ${max_wait}s"
                return 1
            fi
        done
    done

    echo "All servers are ready!"
    return 0
}

# ==== Start memory model vLLM server ====
echo "[server] Starting memory model vLLM server on port ${MEM_PORT}..."
PYTHON="$CONDA_PREFIX/bin/python" \
    DEVICES="$MEM_DEVICES" \
    MODEL="$MEM_MODEL" \
    PORT="$MEM_PORT" \
    KV_OFFLOADING_SIZE="$MEM_KV_OFFLOADING_SIZE" \
    nohup bash scripts/start_vllm_server_with_sync_cpu.sh \
    > "browsecomp_vllm_mem_model_lat.out" 2>&1 &
VLLM_MEM_PID=$!
echo "[server] Memory model server PID: $VLLM_MEM_PID"

# Trap to clean up server process on exit
trap "echo 'Cleaning up (PIDs: $VLLM_MEM_PID)...'; pkill -P $VLLM_MEM_PID 2>/dev/null; kill $VLLM_MEM_PID 2>/dev/null || true" EXIT

# ==== Wait for servers ====
MEMORY_MODEL_HOST="${MEMORY_MODEL_ADDRESS%/v1}"
wait_for_servers \
    "${LLM_BASE_URL%/v1}/v1/models" \
    "${MEMORY_MODEL_HOST}/v1/models"

# Align request model name with what the memory server actually serves.
MEMORY_MODELS_JSON="$(curl -s "${MEMORY_MODEL_HOST}/v1/models" || true)"
SERVED_MEMORY_MODEL_NAME="$(printf '%s' "${MEMORY_MODELS_JSON}" | "$PYTHON" -c 'import json,sys
try:
    payload = json.load(sys.stdin)
    data = payload.get("data", [])
    print(data[0].get("id", "") if data else "")
except Exception:
    print("")
' 2>/dev/null || true)"

if [[ -z "${MEMORY_MODEL_NAME}" ]]; then
    if [[ -n "${SERVED_MEMORY_MODEL_NAME}" ]]; then
        MEMORY_MODEL_NAME="${SERVED_MEMORY_MODEL_NAME}"
        echo "[memory] Auto-detected memory model name: ${MEMORY_MODEL_NAME}"
    else
        MEMORY_MODEL_NAME="${MEM_MODEL}"
        echo "[memory] Could not query /v1/models; defaulting memory model name to MEM_MODEL: ${MEMORY_MODEL_NAME}"
    fi
elif [[ -n "${SERVED_MEMORY_MODEL_NAME}" ]] && [[ "${MEMORY_MODEL_NAME}" != "${SERVED_MEMORY_MODEL_NAME}" ]]; then
    echo "[memory][warn] MEMORY_MODEL_NAME (${MEMORY_MODEL_NAME}) differs from served model (${SERVED_MEMORY_MODEL_NAME}); using served model to avoid 404."
    MEMORY_MODEL_NAME="${SERVED_MEMORY_MODEL_NAME}"
fi

# If the model path was an actor checkpoint, point MEMORY_MODEL_NAME to the merged path
if [[ "$MEM_MODEL" == *"/huggingface" ]] || [[ "$MEM_MODEL" == *"/actor/huggingface" ]]; then
    MEMORY_MODEL_NAME="${MEM_MODEL}"
fi

# ---------- Build command ----------
CMD="python -m r2egym.agenthub.run.edit_comem_miroflow_browsecomp \
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
    --memory_model_name ${MEMORY_MODEL_NAME} \
    --memory_model_temperature ${MEMORY_MODEL_TEMPERATURE} \
    --memory_model_max_gen_tokens ${MEMORY_MODEL_MAX_GEN_TOKENS} \
    --retain_most_turns ${RETAIN_MOST_TURNS} \
    --latency_test True"

# Optional args
[ -n "${MAX_TASKS}" ]            && CMD="${CMD} --max_tasks ${MAX_TASKS}"
[ -n "${LLM_BASE_URL}" ]        && CMD="${CMD} --llm_base_url ${LLM_BASE_URL}"
[ -n "${MEMORY_MODEL_ADDRESS}" ] && CMD="${CMD} --memory_model_address ${MEMORY_MODEL_ADDRESS}"

echo "============================================="
echo "BrowseComp-EN CoMeM Latency Benchmark"
echo "============================================="
echo "  Agent LLM    : ${LLM_NAME}"
echo "  Agent URL    : ${LLM_BASE_URL}"
echo "  Memory Model : ${MEMORY_MODEL_NAME}"
echo "  Memory Path  : ${MEM_MODEL}"
echo "  Memory URL   : ${MEMORY_MODEL_ADDRESS}"
echo "  Retain Turns : ${RETAIN_MOST_TURNS}"
echo "  Concurrency  : ${MAX_CONCURRENT}"
echo "  Judge        : ${JUDGE_MODEL}"
echo "  Data         : ${DATA_PATH}"
echo "  Output       : ${OUTPUT_DIR}"
echo "  Max tasks    : ${MAX_TASKS:-all}"
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
