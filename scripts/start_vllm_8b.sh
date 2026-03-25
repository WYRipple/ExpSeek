#!/bin/bash

MODEL_PATH=xxx/Qwen3-8B
MODEL_NAME=Qwen3-8B

mkdir -p logs

CUDA_VISIBLE_DEVICES=0,1 nohup vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --tensor_parallel_size 2 \
    --gpu-memory-utilization 0.75 \
    --port 8012 > logs/vllm_qwen3_8b.log 2>&1 &
INSTANCE_PID=$!
echo "Qwen3-8B deployed on port 8012 using GPU 0,1 (PID: $INSTANCE_PID)"

echo "---------------------------------------"
echo "All deployed model instances:"
ps aux | grep "vllm serve" | grep -v grep
echo "---------------------------------------"

trap "kill $INSTANCE_PID" SIGTERM
wait $INSTANCE_PID
