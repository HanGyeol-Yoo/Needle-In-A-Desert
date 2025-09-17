# ======================================================================================
# --- Configuration Loader ---
# ======================================================================================
CONFIG_FILE="configs.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "오류: 설정 파일($CONFIG_FILE)을 찾을 수 없습니다."
    exit 1
fi

get_config() {
    grep "^$1:" "$CONFIG_FILE" | sed -e "s/^$1: *//;s/['\"]//g;s/ *$//;s/ *#.*$//"
}

export MODEL_PATH=$(get_config "model_path")
export USE_CHAT_TEMPLATE=$(get_config "use_chat_template")
export OUTPUT_DIR=$(get_config "output_dir")
export NEEDLES_PATH=$(get_config "needles_path")
export EN_JSONL=$(get_config "en_jsonl")
export KR_JSONL=$(get_config "kr_jsonl")
export NUM_GPUS=$(get_config "num_gpus")

export LEN_START=$(get_config "len_start")
export LEN_END=$(get_config "len_end")
export LEN_STEP=$(get_config "len_step")
export REPEATS_PER_COMBO=$(get_config "repeats_per_combo")

export MAX_NEW_TOKENS=$(get_config "max_new_tokens")

# ======================================================================================
# --- Script Logic ---
# ======================================================================================
DATASET_DIR="$OUTPUT_DIR/evaluation_dataset"
INFERENCE_DIR="$OUTPUT_DIR/inference_results"
VISUALIZATION_DIR="$OUTPUT_DIR/visualizations"
MERGED_RESULTS_FILE="$INFERENCE_DIR/all_results.parquet"

MODEL_NAME=$(basename "$MODEL_PATH")

CHAT_FLAG=""
if [ "$USE_CHAT_TEMPLATE" = true ]; then
    CHAT_FLAG="--chat"
    MODEL_NAME="${MODEL_NAME}_chat"
fi

echo "============================================================"
echo "🚀 Starting Needle-in-a-Haystack Pipeline"
echo "============================================================"
echo "- Model: $MODEL_PATH"
echo "- Chat Template: $USE_CHAT_TEMPLATE"
echo "- Output Directory: $OUTPUT_DIR"
echo "- GPUs: $NUM_GPUS"
echo "- Dataset Config: Start=$LEN_START, End=$LEN_END, Step=$LEN_STEP, Repeats=$REPEATS_PER_COMBO"
echo "- Inference Config: Max New Tokens=$MAX_NEW_TOKENS"
echo "------------------------------------------------------------"


# --- Step 1: Build Dataset ---
echo "STEP 1: Building dataset..."
python build_dataset.py \
    --tokenizer_model "$MODEL_PATH" \
    --output_path "$DATASET_DIR" \
    --needles_path "$NEEDLES_PATH" \
    --en_jsonl "$EN_JSONL" \
    --kr_jsonl "$KR_JSONL" \
    --len_start "$LEN_START" \
    --len_end "$LEN_END" \
    --len_step "$LEN_STEP" \
    --repeats_per_combo "$REPEATS_PER_COMBO" \
    $CHAT_FLAG

echo "✅ Dataset created at $DATASET_DIR"
echo "------------------------------------------------------------"


# --- Step 2: Run Inference ---
echo "STEP 2: Running inference with $NUM_GPUS GPUs..."
torchrun --nproc_per_node=$NUM_GPUS run_inference.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_DIR" \
    --output_dir "$INFERENCE_DIR" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --dtype "bfloat16"

echo "✅ Inference complete. Merged results at $MERGED_RESULTS_FILE"
echo "------------------------------------------------------------"


# --- Step 3: Visualize Results ---
echo "STEP 3: Generating visualizations..."
python visualize.py \
    --results_path "$MERGED_RESULTS_FILE" \
    --output_dir "$VISUALIZATION_DIR" \
    --model_name "$MODEL_NAME"

echo "✅ Visualizations saved to $VISUALIZATION_DIR"
echo "============================================================"
echo "🎉 Pipeline finished successfully!"
echo "============================================================"