#!/usr/bin/env bash
# Sweep concurrency for a given memory model to find peak throughput.
#
# Usage:
#   MEM_MODEL=Qwen/Qwen3-8B bash scripts/vllm_bench_mem_model_all.sh
#   MEM_MODEL=Qwen/Qwen3-32B DEVICES=0,1,2,3,4,5,6,7 bash scripts/vllm_bench_mem_model_all.sh

MEM_MODEL="${MEM_MODEL:?ERROR: Set MEM_MODEL (e.g. Qwen/Qwen3-8B)}"

repeat=(1 2 3)
batch_sizes=(1 8 16 32 64 128)
input_lengths=(28206)
output_lengths=(1041)

MODEL_TAG=$(echo "$MEM_MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | \
    sed 's/NVIDIA //g' | tr '[:upper:]' '[:lower:]' | sed 's/ /-/g' | sed 's/gb/g/g')
GPU_NAME="${GPU_NAME:-unknown-gpu}"
OUTDIR="benchmarks-${MODEL_TAG}-${GPU_NAME}"

for r in "${repeat[@]}"; do
for bs in "${batch_sizes[@]}"; do
  for inp_len in "${input_lengths[@]}"; do
    for out_len in "${output_lengths[@]}"; do
      OUTFILE="${OUTDIR}/vllm_bench_batch${bs}_inp${inp_len}_outp${out_len}_repeat${r}.out"

      # Skip if already exists locally
      if [ -f "$OUTFILE" ]; then
        echo "Skipping (already exists): $OUTFILE"
        continue
      fi

      echo "=========================================="
      echo "Running: model=$MEM_MODEL batch=$bs inp=$inp_len out=$out_len repeat=$r"
      echo "=========================================="

      batch_size=$bs input_length=$inp_len output_length=$out_len repeat=$r \
        MEM_MODEL="$MEM_MODEL" bash scripts/vllm_bench_mem_model.sh

      echo "Completed: batch=$bs inp=$inp_len out=$out_len repeat=$r"
      echo ""
    done
  done
done
done

echo "All benchmarks completed for $MEM_MODEL!"
