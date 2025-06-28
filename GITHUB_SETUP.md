# 🚀 GitHub Setup for UC Merced Team Deployment

## 📋 Pre-Push Checklist

### 1. Initialize Git Repository
```bash
cd /Users/roshansanjeev/Desktop/Mi3/VideoLLaMA3
git init
git add .
git commit -m "Initial commit: VideoLLaMA3 blind navigation training setup"
```

### 2. Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `VideoLLaMA3-BlindNavigation`
3. Description: "Fine-tuned VideoLLaMA3 for real-time blind pedestrian navigation assistance"
4. Set to **Public** (for team access) or **Private** (if sensitive)
5. Don't initialize with README (we have our own)

### 3. Connect Local to GitHub
```bash
# Replace [YOUR_USERNAME] with your GitHub username
git remote add origin https://github.com/[YOUR_USERNAME]/VideoLLaMA3-BlindNavigation.git
git branch -M main
git push -u origin main
```

## 📁 Repository Structure for GitHub

```
VideoLLaMA3-BlindNavigation/
├── 📄 README_DEPLOYMENT.md      # Main deployment guide
├── 📄 GITHUB_SETUP.md          # This file
├── 📄 requirements_cluster.txt  # UC Merced cluster dependencies
├── 📄 setup_cluster.py         # Automated cluster setup
├── 📄 .gitignore              # Git ignore file
├── 🗂️ videollama3/             # Core model code (M3 Max optimized)
├── 🗂️ scripts/                # Cluster training scripts
│   └── 📄 train_cluster.sh    # SLURM job submission script
├── 🗂️ data/                   # Training data structure
│   ├── 🗂️ videos/             # Video files (Git LFS)
│   └── 🗂️ annotations/        # JSON annotation files
├── 📄 train_single_video_annotation.py  # Single video training
├── 📄 prepare_blind_navigation_data.py  # Data preparation
└── 📄 test_single_video_model.py        # Model testing
```

## 🔧 Essential Files Created for Your Team

### ✅ Core Training Files
- `train_single_video_annotation.py` - Single video training (Mac M3 tested)
- `scripts/train_cluster.sh` - UC Merced cluster SLURM script
- `setup_cluster.py` - Automated environment setup
- `requirements_cluster.txt` - All dependencies for cluster

### ✅ Data Handling
- `prepare_blind_navigation_data.py` - Convert annotations to training format
- `single_video_training_*_data/` - Ready-to-use training data

### ✅ Model Architecture Fixes
- `videollama3/train.py` - M3 Max compatibility fixes applied
- `videollama3/mm_utils.py` - Video loading fallbacks for different systems

### ✅ Documentation
- `README_DEPLOYMENT.md` - Complete deployment guide
- `GITHUB_SETUP.md` - This setup guide

## 🎯 Team Workflow

### For Team Members to Get Started:
```bash
# 1. Clone repository
git clone https://github.com/[YOUR_USERNAME]/VideoLLaMA3-BlindNavigation.git
cd VideoLLaMA3-BlindNavigation

# 2. Set up UC Merced cluster environment
python setup_cluster.py

# 3. Submit training job
sbatch scripts/train_cluster.sh

# 4. Monitor progress
squeue -u $USER
tail -f logs/train_*.out
```

## 📊 What Your Teammates Can Do

### 🚀 Quick Training (Single Video)
```bash
python train_single_video_annotation.py \
    --video_path data/videos/navigation_sample.mp4 \
    --epochs 3
```

### 🔄 Batch Training (Full Dataset)
```bash
# Submit cluster job with custom parameters
sbatch scripts/train_cluster.sh 2b 5 4 1000
# Model size: 2b, Epochs: 5, Batch size: 4, Samples: 1000
```

### 🧪 Model Testing
```bash
python test_single_video_model.py \
    --model_path ./trained_models/blind_navigation_checkpoint \
    --video_path test_video.mp4
```

## 🔐 Large File Handling (Git LFS)

### If you need to store model checkpoints:
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.bin"
git lfs track "*.safetensors" 
git lfs track "*.pth"
git lfs track "*.mp4"

# Commit LFS tracking
git add .gitattributes
git commit -m "Configure Git LFS for model files"
git push
```

## 📞 Team Communication

### Include in your GitHub repo:
1. **Issues tab**: For bug reports and feature requests
2. **Wiki**: For detailed documentation
3. **Projects**: For task tracking
4. **Actions**: For automated testing (optional)

### Recommended commit message format:
```
feat: add cluster training script for UC Merced
fix: resolve M3 Max compatibility issues
docs: update deployment guide for teammates
train: complete blind navigation model on sample data
```

## 🎯 Quick Start Commands for Teammates

### Copy-paste ready commands:
```bash
# Set up on UC Merced cluster
module load python/3.10 cuda/11.8 pytorch/2.0.0
python -m venv videollama3_env
source videollama3_env/bin/activate
pip install -r requirements_cluster.txt
python setup_cluster.py

# Start training
sbatch scripts/train_cluster.sh

# Check status
squeue -u $USER
```

## ✅ Final GitHub Push Commands

```bash
# Add all files
git add .

# Commit with descriptive message
git commit -m "Complete VideoLLaMA3 blind navigation setup

- Added M3 Max compatible training scripts
- Created UC Merced cluster deployment tools  
- Included sample training data and annotations
- Optimized for blind pedestrian navigation task
- Ready for team collaboration and cluster training"

# Push to GitHub
git push origin main
```

## 🎉 Repository Ready!

Your teammates can now:
1. **Clone** the repository
2. **Run** `python setup_cluster.py` for environment setup
3. **Submit** training jobs with `sbatch scripts/train_cluster.sh`
4. **Monitor** progress and results
5. **Collaborate** on improvements and testing

**🔗 Share this GitHub URL with your team:**
`https://github.com/[YOUR_USERNAME]/VideoLLaMA3-BlindNavigation`