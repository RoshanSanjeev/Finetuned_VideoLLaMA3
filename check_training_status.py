#!/usr/bin/env python3
"""
Check Training Status - Quick Morning Check
Shows exactly how many videos were trained and current status
"""

import json
import os
from datetime import datetime
from pathlib import Path

def check_completion_status():
    """Check if training completed successfully"""
    print("🔍 Checking Training Status")
    print("=" * 40)
    
    # Check completion file
    if os.path.exists('training_completion.json'):
        with open('training_completion.json', 'r') as f:
            info = json.load(f)
        
        print("🎉 TRAINING COMPLETED!")
        print(f"✅ Videos Trained: {info['videos_trained']}")
        print(f"⏱️  Total Time: {info['total_time_hours']:.1f} hours")
        print(f"📁 Model Path: {info['model_path']}")
        print(f"🕐 Completed: {info['completion_time']}")
        return True, info['videos_trained']
    
    # Check interruption file
    elif os.path.exists('training_interruption.json'):
        with open('training_interruption.json', 'r') as f:
            info = json.load(f)
        
        print("⏸️  TRAINING INTERRUPTED")
        print(f"📊 Videos Started: {info['videos_started']}")
        print(f"⏱️  Training Time: {info['training_time_hours']:.1f} hours")
        print(f"📁 Checkpoint Dir: {info['checkpoint_dir']}")
        print(f"🕐 Interrupted: {info['interruption_time']}")
        
        # Check for actual checkpoints
        checkpoint_dir = Path(info['checkpoint_dir'])
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                print(f"💾 Latest Checkpoint: {latest_checkpoint}")
            else:
                print("❌ No checkpoints found")
        
        return False, info['videos_started']
    
    else:
        print("❓ No completion status found")
        print("🔍 Checking for running processes...")
        
        # Check for running training process
        import psutil
        training_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'train_188_videos_overnight.py' in cmdline or 'videollama3/train.py' in cmdline:
                    training_processes.append(proc.info['pid'])
            except:
                pass
        
        if training_processes:
            print(f"🔄 Training still running (PIDs: {', '.join(map(str, training_processes))})")
            print("⏳ Check back later or stop with Ctrl+C")
            return None, 0
        else:
            print("❌ No training process detected")
            return False, 0

def check_model_files():
    """Check what model files are available"""
    print("\n💾 Available Model Files:")
    
    model_dirs = [
        "trained_models/blind_navigation_188_videos",
        "trained_models/blind_navigation_500_videos",
        "single_video_checkpoint"
    ]
    
    found_models = []
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            # Check for model files
            config_file = os.path.join(model_dir, "config.json")
            model_file = os.path.join(model_dir, "pytorch_model.bin") 
            adapter_file = os.path.join(model_dir, "adapter_model.bin")
            
            if os.path.exists(config_file):
                mod_time = os.path.getmtime(model_dir)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                
                model_info = f"✅ {model_dir} (Updated: {mod_time_str})"
                if os.path.exists(adapter_file):
                    model_info += " [LoRA]"
                if os.path.exists(model_file):
                    model_info += " [Full Model]"
                
                print(f"   {model_info}")
                found_models.append(model_dir)
    
    if not found_models:
        print("   ❌ No trained models found")
    
    return found_models

def check_logs():
    """Check recent training logs"""
    print("\n📄 Recent Training Logs:")
    
    log_dir = Path("logs")
    if not log_dir.exists():
        print("   ❌ No logs directory")
        return
    
    # Find recent training logs
    log_files = list(log_dir.glob("training_output_*.log")) + list(log_dir.glob("overnight_training_*.log"))
    
    if not log_files:
        print("   ❌ No training logs found")
        return
    
    # Get most recent log
    latest_log = max(log_files, key=os.path.getctime)
    print(f"   📄 Latest: {latest_log.name}")
    
    # Show last few lines
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        
        print("   📋 Last few lines:")
        for line in lines[-5:]:
            if line.strip():
                print(f"      {line.strip()}")
    except Exception as e:
        print(f"   ❌ Error reading log: {e}")

def give_next_steps(completed, videos_trained):
    """Provide next step recommendations"""
    print("\n💡 Next Steps:")
    
    if completed:
        print("   ✅ Training completed successfully!")
        print("   🧪 Test model: python test_trained_model.py")
        print("   📤 Push to GitHub when ready")
        print("   🚀 Deploy to UC Merced cluster")
    elif videos_trained > 0:
        print("   ⏸️  Training was interrupted but made progress")
        print(f"   📊 {videos_trained} videos were processed")
        print("   🔄 Can resume from checkpoint or use current model")
        print("   🧪 Test partial model: python test_trained_model.py")
    else:
        print("   ❌ Training didn't start or failed early")
        print("   🔍 Check logs for error details")
        print("   🔄 Try starting training again")

def main():
    """Main status checking function"""
    print("🌅 Good Morning! Checking overnight training results...")
    print()
    
    # Check training completion status
    completed, videos_trained = check_completion_status()
    
    # Check available models
    found_models = check_model_files()
    
    # Check logs
    check_logs()
    
    # Provide recommendations
    give_next_steps(completed, videos_trained)
    
    print("\n🎯 Summary:")
    if completed:
        print(f"   🎉 SUCCESS: {videos_trained} videos trained successfully")
    elif videos_trained > 0:
        print(f"   ⚠️  PARTIAL: {videos_trained} videos processed before interruption")
    else:
        print("   ❌ FAILED: Training did not complete")
    
    if found_models:
        print(f"   💾 Models available: {len(found_models)} directories")
    else:
        print("   💾 No trained models found")

if __name__ == "__main__":
    main()