#!/bin/bash

# Single Video Fine-tuning Script for VideoLLaMA3
# Usage: ./train_single_video.sh [video_path] [description_text] [model_size]

set -e

# Parse arguments
VIDEO_PATH=${1:-"videollama3/videos/videos_data_2024_10_25_03_04_21_session_2_Town05_wk1_9_25.mp4"}
DESCRIPTION=${2:-"A detailed description of what's happening in the video for a blind person."}
MODEL_SIZE=${3:-"2b"}

# Configuration
TRAINING_DATA_DIR="./single_video_training_data"
OUTPUT_DIR="./single_video_checkpoint"
RUN_NAME="single_video_finetune_$(date +%Y%m%d_%H%M%S)"

echo "=== VideoLLaMA3 Single Video Fine-tuning ==="
echo "Video: $VIDEO_PATH"
echo "Description: $DESCRIPTION"
echo "Model size: $MODEL_SIZE"
echo "Output directory: $OUTPUT_DIR"
echo "============================================="

# Step 1: Prepare training data
echo "Step 1: Preparing training data..."
python prepare_single_video_data.py \
    --video_path "$VIDEO_PATH" \
    --description "$DESCRIPTION" \
    --output_dir "$TRAINING_DATA_DIR" \
    --copy_video

# Step 2: Set model configuration based on size
if [ "$MODEL_SIZE" == "2b" ]; then
    MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"
    VISION_ENCODER="DAMO-NLP-SG/SigLIP-NaViT"
    PROJECTOR_TYPE="mlp2x_gelu"
    MAX_LENGTH=8192
    MM_MAX_LENGTH=4096
    BATCH_SIZE=1
    GRAD_ACCUM=4
    LR=5e-5
    MM_PROJECTOR_LR=1e-4
    VISION_LR=5e-6
elif [ "$MODEL_SIZE" == "7b" ]; then
    MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
    VISION_ENCODER="DAMO-NLP-SG/SigLIP-NaViT"
    PROJECTOR_TYPE="mlp2x_gelu"
    MAX_LENGTH=8192
    MM_MAX_LENGTH=4096
    BATCH_SIZE=1
    GRAD_ACCUM=8
    LR=2e-5
    MM_PROJECTOR_LR=5e-5
    VISION_LR=2e-6
else
    echo "Unsupported model size: $MODEL_SIZE. Use '2b' or '7b'"
    exit 1
fi

# Step 3: Run training
echo "Step 2: Starting fine-tuning..."
python videollama3/train.py \
    --model_type videollama3_qwen2 \
    --model_path "$MODEL_PATH" \
    --vision_encoder "$VISION_ENCODER" \
    --mm_projector_type "$PROJECTOR_TYPE" \
    --data_path "${TRAINING_DATA_DIR}/annotations.jsonl" \
    --data_folder "$TRAINING_DATA_DIR" \
    --image_merge_size 1 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 64 \
    --model_max_length "$MAX_LENGTH" \
    --mm_max_length "$MM_MAX_LENGTH" \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate "$LR" \
    --mm_projector_lr "$MM_PROJECTOR_LR" \
    --vision_encoder_lr "$VISION_LR" \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to "none" \
    --run_name "$RUN_NAME" \
    --lora_enable True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1

echo "============================================="
echo "Fine-tuning completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "To test your model, use the inference script with:"
echo "  --model_path $OUTPUT_DIR"
echo "============================================="