export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

export MAX_CONTEXT_LEN=32000
export CUDA_VISIBLE_DEVICES=${DEVICES:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
"$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Coder-480B-A35B-Instruct \
    --max-model-len $MAX_CONTEXT_LEN \
    --enable-expert-parallel \
    --tensor-parallel-size $NUM_GPUS \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --port 8000
