export PATH="$CONDA_PREFIX/bin:$PATH"
PYTHON="$CONDA_PREFIX/bin/python"

export MAX_CONTEXT_LEN=65536
export CUDA_VISIBLE_DEVICES=${DEVICES:-"0,1,2,3,4,5,6,7"}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
"$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model zai-org/GLM-4.7 \
    --max-model-len $MAX_CONTEXT_LEN \
    --enable-expert-parallel \
    --tensor-parallel-size $NUM_GPUS \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --port 8000