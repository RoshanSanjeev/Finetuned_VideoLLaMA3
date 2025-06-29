#!/bin/bash
# Start CPU-only training to avoid MPS dtype issues

echo "🖥️  Starting CPU-only VideoLLaMA3 Training"
echo "This will be slower but more stable on M3 Max"

# Disable MPS completely
export PYTORCH_ENABLE_MPS_FALLBACK=0
export CUDA_VISIBLE_DEVICES=""

nohup python videollama3/train.py \
--model_type videollama3_qwen2 \
--model_path "Qwen/Qwen2.5-1.5B-Instruct" \
--vision_encoder "DAMO-NLP-SG/SigLIP-NaViT" \
--mm_projector_type "mlp2x_gelu" \
--mm_attn_implementation "eager" \
--data_path "overnight_training_data.jsonl" \
--data_folder "data/videos" \
--image_merge_size 1 \
--video_merge_size 1 \
--fps 1 \
--max_frames 2 \
--model_max_length 128 \
--mm_max_length 256 \
--bf16 False \
--tf32 False \
--fp16 False \
--no_cuda True \
--output_dir "./trained_models/blind_navigation_cpu" \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--do_train True \
--save_strategy "steps" \
--save_steps 10 \
--save_total_limit 3 \
--learning_rate 1e-4 \
--mm_projector_lr 2e-4 \
--vision_encoder_lr 1e-5 \
--weight_decay 0.01 \
--warmup_ratio 0.05 \
--lr_scheduler_type "linear" \
--logging_steps 1 \
--gradient_checkpointing False \
--dataloader_num_workers 0 \
--report_to "none" \
--run_name "blind_navigation_cpu" \
--lora_enable True \
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.1 > cpu_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "CPU training started with PID: $!"
echo "This will take longer but should be stable"
echo "Monitor with: tail -f cpu_training_*.log"