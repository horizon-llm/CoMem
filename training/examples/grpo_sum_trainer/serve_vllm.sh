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

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DEVICES=${DEVICES:-"0,1,2,3,4,5,6,7"}  # <<< NEW: override like DEVICES="0,1,2,3,4,5,6,7"
PORTS=${PORTS:-"9001,9002"}  # <<< NEW: override like PORTS="9001,9002"

mkdir -p examples/grpo_main_trainer/logs/vllm

# If S3_MODEL_BUCKET is set, check if MODEL exists as a subfolder in that S3 bucket
# and download it to a local cache directory
S3_MODEL_BUCKET="${S3_MODEL_BUCKET:-s3://shopqa-users/yuwzhan/verl-agent-latest/}"
if [[ -n "${S3_MODEL_BUCKET}" ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    echo "S3_MODEL_BUCKET is set but 'aws' CLI is not found in PATH. Please install AWS CLI." >&2
    exit 1
  fi

  echo "S3_MODEL_BUCKET is set to: $S3_MODEL_BUCKET"

  # Construct the full S3 path by appending MODEL to S3_MODEL_BUCKET
  # Normalize: ensure S3_MODEL_BUCKET ends with / and doesn't have duplicate slashes
  S3_BASE="${S3_MODEL_BUCKET%/}/"
  echo "S3 base path: $S3_BASE"
  S3_FULL_PATH="${S3_BASE}${MODEL}"
  S3_FULL_PATH_NO_SLASH="${S3_FULL_PATH%/}"

  echo "Checking if model exists in S3: $S3_FULL_PATH"
  
  # Convert s3://bucket/key/... -> bucket/key/... for local path
  LOCAL_DIR="${MODEL%/}"
  
  # Check if model already exists locally
  if [[ -d "$LOCAL_DIR" ]]; then
    echo "Model already exists locally at: $LOCAL_DIR"
    echo "Skipping S3 download."
  elif aws s3 ls "$S3_FULL_PATH_NO_SLASH/" 2>/dev/null | grep -q . ; then
    echo "Found model in S3 at: $S3_FULL_PATH"
    echo "Syncing model from S3 to local cache: $S3_FULL_PATH -> $LOCAL_DIR"
    mkdir -p "$LOCAL_DIR"
    
    # Sync the S3 path to local directory
    if ! aws s3 sync "$S3_FULL_PATH" "$LOCAL_DIR/" --no-progress ; then
      echo "Failed to sync $S3_FULL_PATH to $LOCAL_DIR" >&2
      exit 1
    fi
    
    # Point MODEL to the local directory for vLLM
    echo "Using downloaded and merged model path: $MODEL"
  else
    echo "Model not found in S3 at: $S3_FULL_PATH"
    echo "Will attempt to use MODEL as-is (may be HuggingFace Hub path or local path): $MODEL"
  fi
  # Final sanity check
  if [[ "$LOCAL_DIR" == *"/actor" ]]; then
    LOCAL_DIR=${LOCAL_DIR} bash scripts/merge_model.sh
    MODEL="${LOCAL_DIR}/huggingface"
  fi
fi

# Split GPU list into an array
IFS=',' read -r -a GPUS <<< "$DEVICES"
N=${#GPUS[@]}

# Require an even number so we can split in half
if (( N % 2 != 0 )); then
  echo "Need an even number of GPUs in DEVICES (got $N: $DEVICES)" >&2
  exit 1
fi

# Split/validate ports
IFS=',' read -r -a PORT_ARR <<< "$PORTS"
if (( ${#PORT_ARR[@]} != 2 )); then
  echo "PORTS must contain exactly two comma-separated ports (got: $PORTS)" >&2
  exit 1
fi
for p in "${PORT_ARR[@]}"; do
  if ! [[ "$p" =~ ^[0-9]+$ ]]; then
    echo "Invalid port (not a number): $p" >&2
    exit 1
  fi
  if ! (( p >= 1 && p <= 65535 )); then
    echo "Invalid port (out of range 1-65535): $p" >&2
    exit 1
  fi
done
if [[ "${PORT_ARR[0]}" == "${PORT_ARR[1]}" ]]; then
  echo "Ports must be distinct: $PORTS" >&2
  exit 1
fi
PORT1="${PORT_ARR[0]}"
PORT2="${PORT_ARR[1]}"

HALF=$(( N / 2 ))

# First and second halves as arrays
FIRST_HALF=("${GPUS[@]:0:HALF}")
SECOND_HALF=("${GPUS[@]:HALF}")

# Join back to comma-separated strings
join_by() { local IFS="$1"; shift; echo "$*"; }
first_half_devices=$(join_by , "${FIRST_HALF[@]}")
second_half_devices=$(join_by , "${SECOND_HALF[@]}")

echo "First half GPUs:  $first_half_devices"
echo "Second half GPUs: $second_half_devices"
echo "Ports: $PORT1 (first server), $PORT2 (second server)"

# If each server should use exactly half the GPUs:
TP_PER_SERVER=${TP_PER_SERVER:-$HALF}

# Example: MODEL="Qwen/Qwen3-8B"
MODEL_SAFE="${MODEL//\//-}"   # -> "Qwen-Qwen3-8B"

CUDA_VISIBLE_DEVICES="$first_half_devices" "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size "$TP_PER_SERVER" \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --host 0.0.0.0 --port "$PORT1" \
  --max_model_len 32768 \
  > "examples/grpo_main_trainer/logs/vllm/${MODEL_SAFE}-${PORT1}.log" 2>&1 &

CUDA_VISIBLE_DEVICES="$second_half_devices" "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size "$TP_PER_SERVER" \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --host 0.0.0.0 --port "$PORT2" \
  --max_model_len 32768 \
  > "examples/grpo_main_trainer/logs/vllm/${MODEL_SAFE}-${PORT2}.log" 2>&1
