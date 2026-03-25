#!/bin/bash
# run_build_kb.sh

# ======= Path configuration =======
EVAL_DIR=outputs/Qwen3-8B-webwalker_train_demo/eval_results
EXP_KB_DIR=experience_base/demo/Qwen3-8B

# ======= LLM API configuration =======
API_KEY=sk-xxx
API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL=qwen3-235b-a22b-instruct-2507

# ======= Embedding API configuration =======
EMB_API_KEY=sk-xxx
EMB_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
EMB_MODEL=text-embedding-v4
EMB_NUM_WORKERS=16

# ======= Run configuration =======
BATCH_SIZE=20
NUM_WORKERS=20
RUN_EMBEDDING=false
RUN_THRESHOLD=true

# =====================================================

echo "===== Step 1: Aggregate rollouts & create pairs ====="
python offline/step1_aggregate.py \
    --eval_dir "$EVAL_DIR" \
    --exp_kb_dir "$EXP_KB_DIR"
[ $? -ne 0 ] && echo "❌ Step1 failed." && exit 1



echo "===== Step 2: Generate experience (LLM) ====="
python offline/step2_generate_exp.py \
    --exp_kb_dir "$EXP_KB_DIR" \
    --api_key "$API_KEY" \
    --api_base "$API_BASE" \
    --model "$MODEL" \
    --num_workers $NUM_WORKERS
[ $? -ne 0 ] && echo "❌ Step2 failed." && exit 1



echo "===== Step 3: Label topics (LLM) ====="
python offline/step3_label_topic.py \
    --exp_kb_dir "$EXP_KB_DIR" \
    --api_key "$API_KEY" \
    --api_base "$API_BASE" \
    --model "$MODEL" \
    --batch_size $BATCH_SIZE
[ $? -ne 0 ] && echo "❌ Step3 failed." && exit 1



echo "===== Step 4: Build knowledge base ====="
python offline/step4_build_kb.py \
    --exp_kb_dir "$EXP_KB_DIR"
[ $? -ne 0 ] && echo "❌ Step4 failed." && exit 1



if [ "$RUN_THRESHOLD" = true ]; then
    echo "===== Step 5: Entropy threshold analysis ====="
    python offline/step5_entropy_threshold.py \
        --eval_dir "$EVAL_DIR" \
        --exp_kb_dir "$EXP_KB_DIR"
    [ $? -ne 0 ] && echo "❌ Step5 failed." && exit 1
fi



if [ "$RUN_EMBEDDING" = true ]; then
    echo "===== Step 6: Build embedding index ====="
    python offline/step6_build_embedding.py \
        --exp_kb_dir "$EXP_KB_DIR" \
        --api_key "$EMB_API_KEY" \
        --api_base "$EMB_API_BASE" \
        --model "$EMB_MODEL" \
        --num_workers $EMB_NUM_WORKERS
    [ $? -ne 0 ] && echo "❌ Step6 failed." && exit 1
fi



echo "===== All Done ====="
echo "Knowledge base saved to: $EXP_KB_DIR"
