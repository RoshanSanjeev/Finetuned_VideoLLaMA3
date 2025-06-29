#!/usr/bin/env python3
"""
Simplified Overnight Training for Available Videos
Uses the existing videollama3/train.py with proper arguments
"""

import os
import json
import time
import logging
import subprocess
from datetime import datetime
from pathlib import Path

def setup_logging():
    """Setup comprehensive logging for overnight training"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"overnight_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def prepare_training_data():
    """Prepare training data from available videos"""
    logger = logging.getLogger(__name__)
    
    # Load annotations
    with open('data/annotations/split_train.json', 'r') as f:
        annotations = json.load(f)
    
    logger.info(f"Loaded {len(annotations)} annotations")
    
    # Find matching videos
    valid_samples = []
    for item in annotations:
        video_path = item.get('video', '')
        if video_path:
            video_filename = os.path.basename(video_path)
            actual_video_path = os.path.join('data/videos', video_filename)
            
            if os.path.exists(actual_video_path):
                # Update the item with correct path and keep only necessary fields
                # Format video as list with relative path (as expected by train.py)
                relative_video_path = os.path.relpath(actual_video_path, "data/videos")
                training_item = {
                    "id": item["id"],
                    "video": [relative_video_path],  # Convert to list format
                    "conversations": item["conversations"]
                }
                valid_samples.append(training_item)
    
    logger.info(f"Found {len(valid_samples)} videos with matching files")
    
    # Save as JSONL for training
    training_file = "overnight_training_data.jsonl"
    with open(training_file, 'w') as f:
        for item in valid_samples:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Saved training data to {training_file}")
    return training_file, len(valid_samples)

def create_progress_monitor(log_file):
    """Create a progress monitoring function"""
    def monitor_progress():
        """Monitor training progress by reading log file"""
        if not os.path.exists(log_file):
            return
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for step information in the logs
        step_info = {}
        for line in reversed(lines[-50:]):  # Check last 50 lines
            if "{'train_runtime':" in line or "train_loss" in line:
                try:
                    # Extract training metrics
                    if "step" in line.lower():
                        step_info["last_log"] = line.strip()
                        break
                except:
                    pass
        
        return step_info
    
    return monitor_progress

def run_overnight_training():
    """Run the overnight training using the existing train.py"""
    logger = setup_logging()
    logger.info("🌙 Starting VideoLLaMA3 Overnight Training (Simplified)")
    logger.info("=" * 60)
    
    # Prepare training data
    training_file, num_videos = prepare_training_data()
    
    if num_videos < 50:
        logger.error(f"Only {num_videos} videos found. Need at least 50.")
        return False
    
    # Estimate training time
    estimated_hours = (num_videos * 8) / 16 / 100  # rough estimate
    logger.info(f"⏱️  Training {num_videos} videos for 8 epochs")
    logger.info(f"⏱️  Estimated time: {estimated_hours:.1f} hours")
    
    # Set up training arguments (based on successful single video training)
    training_args = [
        "python", "videollama3/train.py",
        "--model_type", "videollama3_qwen2",
        "--model_path", "Qwen/Qwen2.5-1.5B-Instruct",
        "--vision_encoder", "DAMO-NLP-SG/SigLIP-NaViT",
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_attn_implementation", "eager",
        "--data_path", training_file,
        "--data_folder", "data/videos",
        "--image_merge_size", "1",
        "--video_merge_size", "2", 
        "--fps", "1",
        "--max_frames", "8",   # Further reduced for CPU training
        "--model_max_length", "512",   # Further reduced for CPU training
        "--mm_max_length", "1024",
        "--bf16", "False",  # Disabled for MPS compatibility
        "--tf32", "False",  # Disabled for M3 Max
        "--fp16", "False",  # Disabled for MPS compatibility
        "--dataloader_num_workers", "0",  # Disable multiprocessing for debugging
        "--output_dir", "./trained_models/blind_navigation_188_videos",
        "--num_train_epochs", "8",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--evaluation_strategy", "no",
        "--do_train", "True",
        "--save_strategy", "steps",
        "--save_steps", "25",  # Save frequently 
        "--save_total_limit", "10",
        "--learning_rate", "1e-4",
        "--mm_projector_lr", "2e-4",
        "--vision_encoder_lr", "1e-5",
        "--weight_decay", "0.01",
        "--warmup_ratio", "0.05",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "5",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "4",
        "--report_to", "tensorboard",
        "--run_name", f"blind_navigation_overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "--lora_enable", "True",
        "--lora_r", "128",  # Higher rank for complex reasoning
        "--lora_alpha", "256",
        "--lora_dropout", "0.05"
    ]
    
    # Create progress tracking
    training_log = f"logs/training_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger.info("🚀 Starting training process...")
    logger.info("💤 Training will continue overnight - you can safely sleep!")
    
    try:
        # Start training with output capture
        with open(training_log, 'w') as log_file:
            process = subprocess.Popen(
                training_args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor progress (non-blocking)
            start_time = time.time()
            while process.poll() is None:
                time.sleep(60)  # Check every minute
                elapsed = time.time() - start_time
                logger.info(f"⏱️  Training running for {elapsed/3600:.1f} hours...")
            
            # Training completed
            return_code = process.wait()
            
            if return_code == 0:
                logger.info("🎉 Training completed successfully!")
                logger.info(f"📁 Model saved to: ./trained_models/blind_navigation_188_videos")
                
                # Save completion status
                completion_info = {
                    "status": "completed",
                    "videos_trained": num_videos,
                    "total_time_hours": (time.time() - start_time) / 3600,
                    "completion_time": datetime.now().isoformat(),
                    "model_path": "./trained_models/blind_navigation_188_videos"
                }
                
                with open('training_completion.json', 'w') as f:
                    json.dump(completion_info, f, indent=2)
                
                return True
            else:
                logger.error(f"Training failed with return code: {return_code}")
                logger.info(f"Check training log: {training_log}")
                return False
                
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        logger.info("💾 Checking for saved checkpoints...")
        
        # Save interruption status
        interruption_info = {
            "status": "interrupted",
            "videos_started": num_videos,
            "training_time_hours": (time.time() - start_time) / 3600,
            "interruption_time": datetime.now().isoformat(),
            "checkpoint_dir": "./trained_models/blind_navigation_188_videos"
        }
        
        with open('training_interruption.json', 'w') as f:
            json.dump(interruption_info, f, indent=2)
        
        return False
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return False

if __name__ == "__main__":
    success = run_overnight_training()
    if success:
        print("\n🎉 Overnight training completed successfully!")
        print("🧪 Run: python test_trained_model.py")
        print("📤 Ready for GitHub push!")
    else:
        print("\n⚠️  Training stopped. Check logs for details.")
        print("💾 Checkpoints may be available for recovery.")