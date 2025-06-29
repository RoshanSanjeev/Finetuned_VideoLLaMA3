# 📁 Data Transfer Guide - Mac to PC

## Video Files for PC Training

### What You Need to Copy:

1. **The 188 Training Videos** (located in `data/videos/`)
2. **The Annotations File** (already in GitHub: `overnight_training_data.jsonl`)

### Option 1: Copy Specific Training Videos Only

```bash
# On Mac - Create archive of just the 188 training videos
python create_training_archive.py
# This will create: training_videos_188.zip
```

### Option 2: Copy All Videos 

```bash
# On Mac - ZIP the entire video directory
cd data/videos
zip -r all_videos.zip *.mp4
```

### Option 3: Cloud Transfer (Recommended)

```bash
# Upload to Google Drive/Dropbox/OneDrive
# Or use rsync/scp if both machines are on same network
```

## 🚀 Quick PC Setup Commands

Once you have the videos on your PC:

```bash
# 1. Clone repository
git clone https://github.com/RoshanSanjeev/Finetuned_VideoLLaMA3.git
cd Finetuned_VideoLLaMA3

# 2. Create data directory and extract videos
mkdir -p data/videos
# Extract your video archive here
unzip training_videos_188.zip -d data/videos/

# 3. Install and run
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install transformers accelerate peft datasets opencv-python

# 4. Start training
python train_188_videos_overnight.py
```

## 📊 Expected Results on PC:

- ✅ **GPU Acceleration**: 10-20x faster than Mac CPU training
- ✅ **Stable Training**: No MPS dtype issues
- ✅ **Mixed Precision**: BF16/FP16 support
- ✅ **Progress Tracking**: Real-time video count monitoring
- ✅ **Overnight Training**: 2-4 hours for 188 videos

Your PC should successfully complete what the M3 Max couldn't!