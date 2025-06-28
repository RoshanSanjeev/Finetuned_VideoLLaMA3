# Single Video Fine-tuning for VideoLLaMA3

This guide will help you fine-tune VideoLLaMA3 on a single video with your own description/narration.

## Files Created

1. **`prepare_single_video_data.py`** - Prepares training data from video + description
2. **`train_single_video.sh`** - Main training script with optimized settings
3. **`test_single_video_model.py`** - Test the fine-tuned model
4. **`example_usage.py`** - Complete example workflow

## Quick Start

### Method 1: Using the Training Script

```bash
# Make script executable
chmod +x train_single_video.sh

# Run training with your video and description
./train_single_video.sh \
    "path/to/your/video.mp4" \
    "Your detailed description for a blind person" \
    "2b"
```

### Method 2: Step by Step

```bash
# Step 1: Prepare training data
python prepare_single_video_data.py \
    --video_path "your_video.mp4" \
    --description "Your detailed narration" \
    --output_dir "./training_data" \
    --copy_video

# Step 2: Run training
python videollama3/train.py \
    --model_type videollama3_qwen2 \
    --model_path "Qwen/Qwen2.5-1.5B-Instruct" \
    --vision_encoder "DAMO-NLP-SG/SigLIP-NaViT" \
    --data_path "./training_data/annotations.jsonl" \
    --data_folder "./training_data" \
    --output_dir "./checkpoint" \
    --lora_enable True \
    --num_train_epochs 3 \
    [... other parameters]
```

### Method 3: Using Example Script

```bash
# Edit example_usage.py with your video path and description
python example_usage.py
```

## Testing Your Model

```bash
python test_single_video_model.py \
    --model_path "./single_video_checkpoint" \
    --video_path "your_test_video.mp4"
```

## Configuration Options

### Model Sizes
- **2b**: Qwen2.5-1.5B-Instruct (faster, less memory)
- **7b**: Qwen2.5-7B-Instruct (better quality, more memory)

### Training Parameters
- **LoRA**: Enabled by default for efficient fine-tuning
- **Epochs**: 3 (adjustable)
- **Max frames**: 64 (adjustable)
- **FPS**: 1 (adjustable)

## Expected Results

After fine-tuning, your model should:
1. Generate descriptions similar to your training data style
2. Focus on details you emphasized in your narration
3. Adapt to your specific use case (e.g., navigation for blind users)

## Hardware Requirements

### Minimum (2B model):
- 8GB GPU memory
- 16GB RAM

### Recommended (7B model):
- 16GB+ GPU memory  
- 32GB+ RAM

## Troubleshooting

### Common Issues:

1. **Out of memory**: Reduce `per_device_train_batch_size` or use 2B model
2. **CUDA errors**: Ensure PyTorch CUDA version matches your driver
3. **Model not found**: Check internet connection for model downloads

### Memory Optimization:
```bash
# Use gradient checkpointing and smaller batch size
--gradient_checkpointing True \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8
```

## File Structure After Training

```
VideoLLaMA3/
├── single_video_training_data/
│   ├── annotations.jsonl
│   └── your_video.mp4
├── single_video_checkpoint/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── config.json
│   └── ...
└── [training scripts]
```

## Next Steps

1. Test on multiple videos to ensure generalization
2. Adjust training epochs based on performance
3. Fine-tune hyperparameters for your specific use case
4. Create evaluation metrics for your task

## Notes

- Training uses LoRA (Low-Rank Adaptation) for efficiency
- Original model weights are preserved
- You can merge LoRA weights later if needed
- The model learns your specific description style