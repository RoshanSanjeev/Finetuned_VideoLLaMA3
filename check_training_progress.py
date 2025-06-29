#!/usr/bin/env python3
"""
Check Training Progress - Morning Status Check
Quick script to see how many videos have been trained overnight
"""

import json
import os
from datetime import datetime
from pathlib import Path

def check_training_progress():
    """Check current training progress and video count"""
    print("📊 VideoLLaMA3 Training Progress Check")
    print("=" * 50)
    
    # Check if progress file exists
    progress_file = Path("training_progress.json")
    if not progress_file.exists():
        print("❌ No training progress file found")
        print("   Training may not have started or progress tracking failed")
        return False
    
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        # Display current status
        print(f"🕐 Last Update: {progress['timestamp']}")
        print(f"📈 Training Step: {progress['current_step']:,} / {progress['total_steps']:,}")
        print(f"🎯 Current Epoch: {progress['current_epoch']}")
        print(f"🎥 Videos Trained: {progress['videos_trained']:,} / {progress['total_videos']:,}")
        print(f"📊 Step Progress: {progress['progress_percentage']:.1f}%")
        print(f"🎬 Video Progress: {progress['videos_percentage']:.1f}%")
        print(f"⏱️  Training Time: {progress['elapsed_hours']:.1f} hours")
        
        # Calculate remaining videos
        remaining_videos = progress['total_videos'] - progress['videos_trained']
        
        print(f"\n🔢 Summary:")
        print(f"   ✅ Completed: {progress['videos_trained']:,} videos")
        print(f"   ⏳ Remaining: {remaining_videos:,} videos")
        
        # Training status assessment
        if progress['videos_trained'] >= progress['total_videos']:
            print(f"   🎉 Status: TRAINING COMPLETE!")
        elif progress['videos_trained'] >= progress['total_videos'] * 0.8:
            print(f"   🔥 Status: Nearly complete (80%+ done)")
        elif progress['videos_trained'] >= progress['total_videos'] * 0.5:
            print(f"   💪 Status: Good progress (50%+ done)")
        elif progress['videos_trained'] >= progress['total_videos'] * 0.2:
            print(f"   🚀 Status: Getting started (20%+ done)")
        else:
            print(f"   🌱 Status: Early training phase")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading progress file: {e}")
        return False

def check_model_checkpoints():
    """Check available model checkpoints"""
    print("\n💾 Available Model Checkpoints:")
    
    checkpoint_dirs = [
        "trained_models/blind_navigation_500_videos",
        "trained_models/blind_navigation_500_videos_final",
        "trained_models/emergency_checkpoint"
    ]
    
    found_checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            # Get checkpoint info
            config_file = os.path.join(checkpoint_dir, "config.json")
            if os.path.exists(config_file):
                mod_time = os.path.getmtime(checkpoint_dir)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                found_checkpoints.append((checkpoint_dir, mod_time_str))
    
    if found_checkpoints:
        for checkpoint, mod_time in found_checkpoints:
            print(f"   ✅ {checkpoint} (Updated: {mod_time})")
    else:
        print("   ❌ No model checkpoints found yet")
    
    return len(found_checkpoints) > 0

def check_training_logs():
    """Check recent training logs"""
    print("\n📄 Recent Training Logs:")
    
    log_dir = Path("logs")
    if not log_dir.exists():
        print("   ❌ No logs directory found")
        return
    
    # Find recent log files
    log_files = list(log_dir.glob("overnight_training_*.log"))
    if not log_files:
        print("   ❌ No training logs found")
        return
    
    # Get latest log
    latest_log = max(log_files, key=os.path.getctime)
    print(f"   📄 Latest log: {latest_log.name}")
    
    # Show last few lines
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            print("   📋 Last 5 log entries:")
            for line in lines[-5:]:
                if line.strip():
                    print(f"      {line.strip()}")
    except Exception as e:
        print(f"   ❌ Error reading log: {e}")

def give_recommendations():
    """Provide recommendations based on current progress"""
    print("\n💡 Recommendations:")
    
    # Check if training is still running
    import psutil
    training_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'train_500_videos_overnight.py' in ' '.join(proc.info['cmdline'] or []):
                training_processes.append(proc.info['pid'])
        except:
            pass
    
    if training_processes:
        print("   🔄 Training is still running (PIDs: {})".format(', '.join(map(str, training_processes))))
        print("   ⏳ You can safely let it continue or stop it with Ctrl+C")
        print("   📊 Current progress will be saved automatically")
    else:
        print("   ⏹️  Training process not detected")
        print("   📝 Check logs to see if training completed or stopped")
    
    # Check progress file
    progress_file = Path("training_progress.json")
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            
            if progress['videos_trained'] >= 400:  # 80% of 500
                print("   ✅ Great progress! You can proceed with testing and GitHub push")
                print("   🧪 Run: python test_trained_model.py")
            elif progress['videos_trained'] >= 250:  # 50% of 500  
                print("   👍 Decent progress! Consider running a bit longer or proceed with current model")
            else:
                print("   ⏳ Limited progress so far, consider letting training continue")
                
        except:
            pass

def main():
    """Main progress checking function"""
    success = check_training_progress()
    
    if success:
        check_model_checkpoints()
        check_training_logs()
        give_recommendations()
    else:
        print("\n🔍 Alternative checks:")
        check_model_checkpoints()
        check_training_logs()
    
    print("\n📋 Next Steps:")
    print("1. If satisfied with progress: python test_trained_model.py")
    print("2. Stop training: Ctrl+C in training terminal")
    print("3. Continue training: Let it run longer")
    print("4. Push to GitHub: After validation")

if __name__ == "__main__":
    main()