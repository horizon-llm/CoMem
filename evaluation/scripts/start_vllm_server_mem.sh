#!/bin/bash

echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DEVICES=${DEVICES:-"0,1,2,3,4,5,6,7"}  # Override like DEVICES="0,1,2,3"
PORT=${PORT:-9001}  # Override like PORT=9001
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-131072}  # Override like MAX_CONTEXT_LEN=131072

# Split GPU list into an array
IFS=',' read -r -a GPUS <<< "$DEVICES"
N=${#GPUS[@]}

if (( N == 0 )); then
  echo "No GPUs specified in DEVICES: $DEVICES" >&2
  exit 1
fi

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
  echo "Invalid port (not a number): $PORT" >&2
  exit 1
fi
if ! (( PORT >= 1 && PORT <= 65535 )); then
  echo "Invalid port (out of range 1-65535): $PORT" >&2
  exit 1
fi

join_by() { local IFS="$1"; shift; echo "$*"; }
gpu_devices=$(join_by , "${GPUS[@]}")

echo "GPUs:  $gpu_devices"
echo "Port: $PORT"

TP_SIZE=${TP_SIZE:-$N}

CUDA_VISIBLE_DEVICES="$gpu_devices" "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --host 0.0.0.0 --port "$PORT" \
  --max_model_len "$MAX_CONTEXT_LEN"
