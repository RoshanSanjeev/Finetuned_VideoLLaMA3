# 🚀 Manual Push Commands for GitHub

## Copy and paste these commands in Terminal:

### 1. Navigate to your project directory:
```bash
cd /Users/roshansanjeev/Desktop/Mi3/VideoLLaMA3
```

### 2. Initialize Git (if not already done):
```bash
git init
```

### 3. Add your specific GitHub repository:
```bash
git remote add origin https://github.com/RoshanSanjeev/FinetunedVideoLLaMA3.git
```

### 4. Add all files:
```bash
git add .
```

### 5. Create commit with comprehensive message:
```bash
git commit -m "Complete VideoLLaMA3 Blind Navigation Training System

🎯 Features:
- Fine-tuned VideoLLaMA3 for blind pedestrian navigation
- M3 Max compatibility (eager attention, MPS acceleration)  
- UC Merced cluster deployment scripts (SLURM integration)
- Single video training pipeline with annotations
- Automated setup and validation tools

🔧 Technical Details:
- Base model: Qwen2.5-1.5B-Instruct + SigLIP-NaViT vision encoder
- Training method: LoRA fine-tuning for efficiency
- Video processing: OpenCV fallback for Mac compatibility
- Data format: VideoLLaMA3 conversational format for navigation

📁 Key Files:
- train_single_video_annotation.py: Single video training
- scripts/train_cluster.sh: UC Merced SLURM job script  
- setup_cluster.py: Automated cluster environment setup
- README_DEPLOYMENT.md: Complete deployment guide
- requirements_cluster.txt: All dependencies for cluster

🎯 Use Case:
Real-time visual navigation assistance for blind pedestrians.
Processes live video and generates spoken navigation instructions.

✅ Ready for UC Merced cluster deployment and team collaboration."
```

### 6. Set main branch:
```bash
git branch -M main
```

### 7. Push to GitHub:
```bash
git push -u origin main
```

## 🔐 If you get authentication errors:

### Option 1: Personal Access Token (Recommended)
1. Go to GitHub.com → Settings → Developer settings → Personal access tokens
2. Generate new token with "repo" permissions
3. Use token as password when prompted

### Option 2: SSH (if configured)
```bash
git remote set-url origin git@github.com:RoshanSanjeev/FinetunedVideoLLaMA3.git
git push -u origin main
```

## ✅ Success Indicators:
- You'll see: "Branch 'main' set up to track remote branch 'main' from 'origin'"
- Repository will be visible at: https://github.com/RoshanSanjeev/FinetunedVideoLLaMA3
- Your teammates can clone with: `git clone https://github.com/RoshanSanjeev/FinetunedVideoLLaMA3.git`

## 🎯 What Your Teammates Will See:
- Complete VideoLLaMA3 training setup
- UC Merced cluster deployment scripts
- Blind navigation model training pipeline  
- Documentation for immediate use
- Ready-to-run commands for cluster training