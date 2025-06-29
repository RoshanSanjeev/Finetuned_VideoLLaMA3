#!/usr/bin/env python3
"""
Real-time Training Monitor
Check training progress and show current status
"""

import os
import psutil
import time
from datetime import datetime

def monitor_training():
    """Monitor the current training process"""
    print("🔍 VideoLLaMA3 Training Monitor")
    print("=" * 40)
    
    # Check for training processes
    training_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'videollama3/train.py' in cmdline:
                training_processes.append({
                    'pid': proc.info['pid'],
                    'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                    'runtime': time.time() - proc.info['create_time']
                })
        except:
            pass
    
    if training_processes:
        for proc in training_processes:
            runtime_hours = proc['runtime'] / 3600
            print(f"🔄 Training Process Found:")
            print(f"   PID: {proc['pid']}")
            print(f"   Memory: {proc['memory_mb']:.1f} MB")
            print(f"   Runtime: {runtime_hours:.2f} hours")
    else:
        print("❌ No training process detected")
    
    # Check output directories
    print(f"\n📁 Output Directories:")
    output_dirs = [
        "./trained_models/blind_navigation_188_videos",
        "./trained_models/blind_navigation_188_videos_final"
    ]
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            # Count checkpoints
            checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint-')]
            print(f"   ✅ {output_dir}: {len(checkpoints)} checkpoints")
            
            # Show latest files
            try:
                files = os.listdir(output_dir)
                if files:
                    latest_file = max([os.path.join(output_dir, f) for f in files], key=os.path.getctime)
                    mod_time = datetime.fromtimestamp(os.path.getctime(latest_file))
                    print(f"      Latest update: {mod_time.strftime('%H:%M:%S')}")
            except:
                pass
        else:
            print(f"   ❌ {output_dir}: Not created yet")
    
    # Check for log files
    print(f"\n📄 Recent Activity:")
    if os.path.exists("logs"):
        log_files = [f for f in os.listdir("logs") if f.endswith('.log')]
        if log_files:
            latest_log = max([os.path.join("logs", f) for f in log_files], key=os.path.getctime)
            mod_time = datetime.fromtimestamp(os.path.getctime(latest_log))
            print(f"   📄 Latest log: {os.path.basename(latest_log)}")
            print(f"   🕐 Last update: {mod_time.strftime('%H:%M:%S')}")
        else:
            print("   ❌ No log files found")
    
    # System status
    print(f"\n💻 System Status:")
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"   CPU Usage: {cpu_percent:.1f}%")
    print(f"   Memory: {memory.percent:.1f}% used ({memory.used/1e9:.1f}GB / {memory.total/1e9:.1f}GB)")
    
    return len(training_processes) > 0

if __name__ == "__main__":
    is_training = monitor_training()
    
    if is_training:
        print(f"\n✅ Training is actively running!")
        print(f"💤 You can safely let it continue overnight")
        print(f"🔄 Run this script again to check progress")
    else:
        print(f"\n⏹️  No active training detected")
        print(f"🔍 Check logs or start training manually")