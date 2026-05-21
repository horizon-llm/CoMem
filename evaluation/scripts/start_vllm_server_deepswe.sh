#!/bin/bash

echo "Using Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"
export MAX_CONTEXT_LEN=65536
export CUDA_VISIBLE_DEVICES=${DEVICES:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
"$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model agentica-org/DeepSWE-Preview \
    --tensor-parallel-size $NUM_GPUS \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max_num_batched_tokens 4096 \
    --port 8000