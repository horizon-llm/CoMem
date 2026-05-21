#!/usr/bin/env bash

# Define arrays of values to iterate over
repeat=(1 2 3)
batch_sizes=(1 8 16 32 64 128)
input_lengths=(2048 4096 8192 16384 32768 60000)
output_lengths=(1024)

# Get and format GPU name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | \
    sed 's/NVIDIA //g' | \
    tr '[:upper:]' '[:lower:]' | \
    sed 's/ /-/g' | \
    sed 's/gb/g/g')

# Loop through all combinations
for r in ${repeat[@]}; do
for bs in ${batch_sizes[@]}; do
  for inp_len in ${input_lengths[@]}; do
    for out_len in ${output_lengths[@]}; do
      # Skip if results already exist locally
      OUTFILE="vllm_bench_batch${bs}_inp${inp_len}_outp${out_len}_repeat${r}.out"
      if [ -f "$OUTFILE" ]; then
        echo "Skipping (already exists): $OUTFILE"
        continue
      fi

      echo "=========================================="
      echo "Running benchmark with:"
      echo "  repeat=$r"
      echo "  batch_size=$bs"
      echo "  input_length=$inp_len"
      echo "  output_length=$out_len"
      echo "=========================================="

      # Run the benchmark script with the specific parameters
      batch_size=$bs input_length=$inp_len output_length=$out_len repeat=$r bash scripts/vllm_bench_batch_deepswe.sh

      echo "Completed: batch_size=$bs, input_length=$inp_len, output_length=$out_len, repeat=$r"
      echo ""
    done
  done
done
done

echo "All benchmarks completed!"
