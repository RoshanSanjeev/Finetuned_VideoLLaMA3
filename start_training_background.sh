#!/bin/bash
# Start VideoLLaMA3 training in background with full logging

echo "🚀 Starting VideoLLaMA3 Background Training"
echo "Training 188 videos for blind navigation"
echo "Output will be logged to: training_$(date +%Y%m%d_%H%M%S).log"

# Prevent system sleep
caffeinate -d -i -m -s &
CAFFEINATE_PID=$!
echo "Caffeinate PID: $CAFFEINATE_PID"

# Create log file
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

# Start training in background
nohup python videollama3/train.py \
--model_type videollama3_qwen2 \
--model_path "Qwen/Qwen2.5-1.5B-Instruct" \
--vision_encoder "DAMO-NLP-SG/SigLIP-NaViT" \
--mm_projector_type "mlp2x_gelu" \
--mm_attn_implementation "eager" \
--data_path "overnight_training_data.jsonl" \
--data_folder "data/videos" \
--image_merge_size 1 \
--video_merge_size 2 \
--fps 1 \
--max_frames 4 \
--model_max_length 256 \
--mm_max_length 512 \
--bf16 False \
--tf32 False \
--fp16 False \
--output_dir "./trained_models/blind_navigation_188_videos_final" \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--do_train True \
--save_strategy "steps" \
--save_steps 20 \
--save_total_limit 5 \
--learning_rate 5e-5 \
--mm_projector_lr 1e-4 \
--vision_encoder_lr 5e-6 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--gradient_checkpointing True \
--dataloader_num_workers 0 \
--report_to "none" \
--run_name "blind_navigation_background" \
--lora_enable True \
--lora_r 32 \
--lora_alpha 64 \
--lora_dropout 0.1 > "$LOG_FILE" 2>&1 &

TRAINING_PID=$!

echo "Training started with PID: $TRAINING_PID"
echo "Log file: $LOG_FILE"
echo "Monitor with: python monitor_training.py"
echo "View logs with: tail -f $LOG_FILE"

# Save process info
echo "TRAINING_PID=$TRAINING_PID" > training_process.info
echo "CAFFEINATE_PID=$CAFFEINATE_PID" >> training_process.info
echo "LOG_FILE=$LOG_FILE" >> training_process.info
echo "START_TIME=$(date)" >> training_process.info

echo ""
echo "✅ Training started in background!"
echo "💤 Safe to close terminal - training will continue"
echo "🔄 Check status: python monitor_training.py"