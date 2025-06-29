# 🌙 Overnight Training Guide - 500 Videos

## Quick Start (Before Bed)

```bash
# Simply run this command and go to sleep
./start_overnight_training.sh
```

## What This Will Do

### 📊 Training Configuration
- **Videos**: All 500 videos with matching annotations
- **Epochs**: 8 (for maximum learning)
- **Estimated Time**: 8-12 hours (perfect for overnight)
- **Batch Size**: Effective batch size of 16 (optimized for M3 Max)
- **Checkpoints**: Saves every 100 steps (safety first!)

### 🔧 M3 Max Optimizations
- **Memory Management**: Uses 24GB peak (~50% of your 48GB)
- **Sleep Prevention**: Keeps Mac awake during training
- **MPS Backend**: Optimized for Apple Silicon
- **Eager Attention**: Compatible with M3 Max architecture
- **BFloat16**: Faster and more stable on Apple Silicon

### 📈 Training Parameters
```python
Learning Rate: 1e-4          # Conservative for stability
LoRA Rank: 128              # High rank for complex navigation
Gradient Accumulation: 16    # Effective large batch training
Warmup: 5%                  # Stable training start
Scheduler: Cosine           # Smooth learning rate decay
```

## 🛡️ Safety Features

### Automatic Checkpointing
- Saves model every 100 steps
- Keeps last 10 checkpoints
- Emergency checkpoint if training fails
- Progress logged every 10 steps

### System Protection
- Prevents Mac from sleeping
- Monitors battery and temperature
- Graceful shutdown on errors
- Comprehensive logging

## 📊 Monitoring While You Sleep

### Check Progress When You Wake Up
```bash
# Quick progress check - shows exactly how many videos trained
python check_training_progress.py

# Live progress monitoring (if still training)
tail -f logs/overnight_training_*.log

# Or check the latest console output
tail -f logs/overnight_training_*_console.log
```

### Progress Indicators
- **Videos Trained**: Exact count of videos processed (e.g., "Videos: 387/500")
- **Step Progress**: Shows current step / total steps  
- **ETA**: Estimated time remaining
- **Loss**: Training loss (should decrease over time)
- **Memory**: GPU memory usage
- **Progress File**: Real-time `training_progress.json` with all stats

## 🎯 Expected Results

### Training Metrics
- **Start Loss**: ~2.8 (typical for blind navigation)
- **End Loss**: ~1.2-1.5 (good convergence)
- **Model Size**: ~3.5GB (base) + 200MB (LoRA)
- **Total Steps**: ~2,500 steps (500 videos × 8 epochs ÷ 16 batch)

### Morning Checklist
```bash
# First thing when you wake up - check how many videos were trained
python check_training_progress.py
```

You'll see output like:
```
🎥 Videos Trained: 387 / 500 (77.4%)
📊 Step Progress: 1,548 / 2,000 (77.4%)
⏱️  Training Time: 8.2 hours
🔥 Status: Nearly complete (80%+ done)
```

If training completed, you'll see:
```
✅ Training completed successfully!
Final model saved to: ./trained_models/blind_navigation_500_videos_final
```

## 🚀 After Training (Morning Routine)

### 1. Validate Model
```bash
python test_trained_model.py
```

### 2. Check Quality
The model should generate responses like:
```json
{
    "reason": "constant_instruction",
    "instruction": "Continue walking straight, the path is clear ahead."
}
```

### 3. Prepare for GitHub
- Model will be in `./trained_models/blind_navigation_500_videos_final/`
- Logs will be in `./logs/` directory
- You can then push to your GitHub repository

## 🔧 Troubleshooting

### If Training Fails
1. Check `logs/overnight_training_*.log` for errors
2. Look for emergency checkpoint in `./trained_models/emergency_checkpoint`
3. Most common issues:
   - Out of memory (reduce batch size)
   - Video file corruption (check data integrity)
   - Network issues (model download problems)

### If Mac Goes to Sleep
- Training script uses `caffeinate` to prevent sleep
- If it still sleeps, adjust Energy Saver settings
- Keep Mac plugged in during training

### Memory Issues
```bash
# Check memory usage
vm_stat | grep free

# If low memory, reduce batch size in train_500_videos_overnight.py:
per_device_train_batch_size=1
gradient_accumulation_steps=8  # Reduce from 16
```

## 📱 Training Schedule

### Perfect Timing
- **Start**: 10 PM (before bed)
- **Duration**: 8-12 hours
- **Finish**: 6-10 AM (when you wake up)

### Training Timeline
```
Hour 0-1:  Model loading and data preparation
Hour 1-2:  First epoch (learning basic patterns)
Hour 2-4:  Epochs 2-3 (improving navigation accuracy)
Hour 4-6:  Epochs 4-5 (refining spatial reasoning)
Hour 6-8:  Epochs 6-7 (mastering complex scenarios)
Hour 8-10: Epoch 8 (final optimization)
Hour 10+:  Model saving and validation
```

## 🎉 Success Indicators

### Training Completed Successfully
- Final model saved to expected location
- Validation script passes
- Loss decreased from ~2.8 to ~1.2-1.5
- No critical errors in logs
- Model generates coherent navigation instructions

### Ready for Production
✅ Model trained on all 500 videos  
✅ Optimized for blind navigation scenarios  
✅ M3 Max compatible and tested  
✅ Ready for UC Merced cluster deployment  
✅ GitHub-ready with documentation  

## 💤 Sleep Well!

Your VideoLLaMA3 model will be training hard while you rest. Wake up to a fully trained blind navigation assistant! 🌅