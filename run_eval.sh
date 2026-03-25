#!/bin/bash
# run_eval.sh

# ======= 参数配置 =======
INPUT_DIR=outputs/Qwen3-8B-webwalker_train_demo
TOKENIZER_PATH=tokenizer
API_KEY=sk-xxx
API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
JUDGE_MODEL=qwen3-235b-a22b-instruct-2507
NUM_WORKERS=20
DEBUG=false

# ======= 执行 =======
DEBUG_FLAG=""
if [ "$DEBUG" = true ]; then
    DEBUG_FLAG="--debug"
fi

echo "===== Step 1: Evaluation ====="
python scripts/evaluate.py \
    --input_dir "$INPUT_DIR" \
    --api_key "$API_KEY" \
    --api_base "$API_BASE" \
    --judge_model "$JUDGE_MODEL" \
    --num_workers $NUM_WORKERS \
    $DEBUG_FLAG

echo "===== Step 2: Metrics ====="
python scripts/metric.py \
    --input_dir "$INPUT_DIR" \
    --tokenizer_path "$TOKENIZER_PATH"

echo "===== All Done ====="
