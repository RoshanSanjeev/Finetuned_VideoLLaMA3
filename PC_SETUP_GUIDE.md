# 🖥️ VideoLLaMA3 PC Setup Guide

## Quick PC Deployment Instructions

### 1. Clone Repository on Your PC
```bash
git clone https://github.com/RoshanSanjeev/Finetuned_VideoLLaMA33.git
cd Finetuned_VideoLLaMA33
```

### 2. Install Dependencies
```bash
# Create virtual environment
python -m venv videollama3_env
source videollama3_env/bin/activate  # On Windows: videollama3_env\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Install additional dependencies
pip install transformers accelerate peft datasets
pip install opencv-python pillow imageio-ffmpeg
pip install tensorboard wandb
pip install psutil  # For monitoring
```

### 3. Quick Training Test
```bash
# Test if GPU training works
python monitor_training.py  # Check system status

# Start training on 188 videos (should work on PC with GPU)
python train_188_videos_overnight.py
```

### 4. Monitor Progress
```bash
# Check training status
python check_training_status.py

# Real-time monitoring  
python monitor_training.py

# View logs
tail -f logs/training_*.log
```

## 📊 Expected PC Performance

### With NVIDIA GPU:
- **Training Time**: 1-3 hours for 188 videos
- **Memory Usage**: 8-16GB GPU memory
- **No dtype issues**: CUDA handles mixed precision properly

### Key Differences from Mac:
- ✅ **CUDA Support**: No MPS compatibility issues
- ✅ **Mixed Precision**: BF16/FP16 will work properly
- ✅ **Faster Training**: GPU acceleration vs CPU-only on Mac
- ✅ **More Stable**: Fewer edge case errors

## 🎯 Files Ready for PC Training

### Core Training Scripts:
- `train_188_videos_overnight.py` - Main training (188 videos)
- `start_cpu_training.sh` - CPU fallback (if no GPU)
- `monitor_training.py` - Real-time status monitoring

### Training Data:
- `overnight_training_data.jsonl` - 188 formatted video annotations
- `data/videos/` - Video files (188 available, need to copy videos)

### Expected Success Rate on PC: **95%+**
The same dtype issues that blocked Mac training should not occur on PC with proper CUDA/GPU setup.

## 🔧 Troubleshooting for PC

### If GPU not detected:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should return True
```

### If CUDA issues:
```bash
# Check CUDA version
nvidia-smi
# Reinstall PyTorch with correct CUDA version
```

### If training fails:
```bash
# Try CPU training as fallback
bash start_cpu_training.sh
```

## 🎉 Next Steps After PC Training

1. **Validate Model**: `python test_trained_model.py`
2. **Deploy to Cluster**: Use UC Merced scripts  
3. **Test Navigation**: Run inference on new videos

Your PC should handle this training much better than the M3 Max!