#!/bin/bash
#SBATCH --job-name=videollama3_blind_nav
#SBATCH --account=your_account_name
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# UC Merced Cluster Training Script for VideoLLaMA3 Blind Navigation
echo "Starting VideoLLaMA3 training on UC Merced cluster..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load required modules
module load python/3.10
module load cuda/11.8
module load pytorch/2.0.0

# Activate virtual environment
source videollama3_env/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_CACHE=/tmp/hf_cache_$SLURM_JOB_ID

# Create output directories
mkdir -p logs
mkdir -p trained_models
mkdir -p $TRANSFORMERS_CACHE

# Training configuration
MODEL_SIZE=${1:-"2b"}
EPOCHS=${2:-"5"}
BATCH_SIZE=${3:-"2"}
DATA_SAMPLES=${4:-"1000"}

echo "Configuration:"
echo "  Model Size: $MODEL_SIZE"
echo "  Epochs: $EPOCHS" 
echo "  Batch Size: $BATCH_SIZE"
echo "  Data Samples: $DATA_SAMPLES"

# Run training
python videollama3/train.py \
    --model_type videollama3_qwen2 \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --mm_attn_implementation flash_attention_2 \
    --data_path ./data/annotations/blind_navigation.jsonl \
    --data_folder ./data \
    --image_merge_size 1 \
    --video_merge_size 2 \
    --fps 1 \
    --max_frames 32 \
    --model_max_length 2048 \
    --mm_max_length 1024 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ./trained_models/blind_navigation_${MODEL_SIZE}_${SLURM_JOB_ID} \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --mm_projector_lr 1e-4 \
    --vision_encoder_lr 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --run_name blind_navigation_${MODEL_SIZE}_${SLURM_JOB_ID} \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --remove_unused_columns False \
    --dataset_cache_dir $TRANSFORMERS_CACHE

# Check training results
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "Model saved to: ./trained_models/blind_navigation_${MODEL_SIZE}_${SLURM_JOB_ID}"
    
    # Test the trained model
    echo "🧪 Testing trained model..."
    python test_single_video_model.py \
        --model_path ./trained_models/blind_navigation_${MODEL_SIZE}_${SLURM_JOB_ID} \
        --video_path data/videos/test_video.mp4
        
    # Create deployment package
    echo "📦 Creating deployment package..."
    tar -czf trained_models/blind_navigation_${MODEL_SIZE}_${SLURM_JOB_ID}.tar.gz \
        trained_models/blind_navigation_${MODEL_SIZE}_${SLURM_JOB_ID}/
        
    echo "✅ Deployment package created: blind_navigation_${MODEL_SIZE}_${SLURM_JOB_ID}.tar.gz"
else
    echo "❌ Training failed! Check error logs."
    exit 1
fi

# Cleanup
rm -rf $TRANSFORMERS_CACHE

echo "Job completed at: $(date)"