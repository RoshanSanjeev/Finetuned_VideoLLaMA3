#!/bin/bash
# Fix GitHub commit to remove Claude attribution and update message

echo "🔧 Fixing GitHub commit attribution and message..."
echo "=" * 50

# Navigate to project directory
cd /Users/roshansanjeev/Desktop/Mi3/VideoLLaMA3

# Amend the last commit with new message and author
echo "📝 Updating commit message and author..."
git commit --amend --author="RoshanSanjeev <your-email@example.com>" -m "VideoLLaMA3 Blind Navigation Training System

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

# Force push the amended commit
echo "🚀 Force pushing updated commit to GitHub..."
git push --force-with-lease origin main

# Check if push was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Commit updated on GitHub!"
    echo "🔗 Repository: https://github.com/RoshanSanjeev/FinetunedVideoLLaMA3"
    echo ""
    echo "✅ Changes made:"
    echo "- Removed Claude attribution"
    echo "- Updated commit author to RoshanSanjeev"
    echo "- Cleaned up commit message"
    echo ""
    echo "📋 Your teammates will now see:"
    echo "- Author: RoshanSanjeev"
    echo "- Clean, professional commit message"
    echo "- No AI attribution"
else
    echo ""
    echo "❌ Push failed. You may need to run manually:"
    echo "git push --force-with-lease origin main"
fi