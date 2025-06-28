#!/bin/bash
# Push VideoLLaMA3 Blind Navigation to GitHub
# Repository: https://github.com/RoshanSanjeev/FinetunedVideoLLaMA3.git

echo "🚀 Pushing VideoLLaMA3 Blind Navigation to GitHub..."
echo "Repository: https://github.com/RoshanSanjeev/FinetunedVideoLLaMA3.git"
echo "=" * 60

# Navigate to project directory
cd /Users/roshansanjeev/Desktop/Mi3/VideoLLaMA3

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add remote repository (your specific GitHub repo)
echo "🔗 Setting up remote repository..."
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/RoshanSanjeev/FinetunedVideoLLaMA3.git
echo "✅ Remote repository set to: https://github.com/RoshanSanjeev/FinetunedVideoLLaMA3.git"

# Check git status
echo "📋 Checking repository status..."
git status

# Add all files
echo "📦 Adding files to Git..."
git add .

# Show what will be committed
echo "📝 Files to be committed:"
git status --short

# Create comprehensive commit message
echo "💬 Creating commit..."
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

# Set the main branch
echo "🌟 Setting main branch..."
git branch -M main

# Push to GitHub
echo "🚀 Pushing to GitHub..."
echo "You may be prompted for your GitHub credentials..."
git push -u origin main

# Check if push was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Repository pushed to GitHub!"
    echo "🔗 Your repository is now available at:"
    echo "   https://github.com/RoshanSanjeev/FinetunedVideoLLaMA3"
    echo ""
    echo "📋 Next steps for your teammates:"
    echo "1. Clone: git clone https://github.com/RoshanSanjeev/FinetunedVideoLLaMA3.git"
    echo "2. Setup: python setup_cluster.py"
    echo "3. Train: sbatch scripts/train_cluster.sh"
    echo ""
    echo "✅ Your VideoLLaMA3 blind navigation model is ready for UC Merced cluster!"
else
    echo ""
    echo "❌ Push failed. Common solutions:"
    echo "1. Check your GitHub credentials"
    echo "2. Ensure repository exists at: https://github.com/RoshanSanjeev/FinetunedVideoLLaMA3"
    echo "3. Verify you have push permissions"
    echo ""
    echo "Try running manually:"
    echo "git push -u origin main"
fi