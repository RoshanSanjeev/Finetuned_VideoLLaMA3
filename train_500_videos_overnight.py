#!/usr/bin/env python3
"""
VideoLLaMA3 Overnight Training - 500 Videos Batch Training
Optimized for M3 Max with 48GB RAM - Sleep Training Configuration
"""

import os
import json
import time
import torch
import logging
from datetime import datetime
from pathlib import Path
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import sys
sys.path.append('.')
from videollama3.train import LazySupervisedDataset, DataArguments
from videollama3.videollama3_trainer import VideoLLaMA3Trainer
from videollama3.model.processor import Videollama3Processor

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

def estimate_training_time(num_samples, epochs, batch_size=1):
    """Estimate training time for overnight planning"""
    # Based on M3 Max performance: ~3.2 seconds per step
    steps_per_epoch = num_samples // batch_size
    total_steps = steps_per_epoch * epochs
    estimated_seconds = total_steps * 3.2
    
    hours = estimated_seconds // 3600
    minutes = (estimated_seconds % 3600) // 60
    
    return hours, minutes, total_steps

def prepare_500_video_dataset():
    """Prepare all 500 videos for training"""
    logger = logging.getLogger(__name__)
    
    # Load annotations
    with open('data/annotations/split_train.json', 'r') as f:
        annotations = json.load(f)
    
    logger.info(f"Loaded {len(annotations)} annotations")
    
    # Verify video files exist and fix paths
    valid_samples = []
    missing_videos = []
    
    for item in annotations:
        video_path = item.get('video', '')
        if video_path:
            # Fix the video path - replace the annotation path with actual path
            video_filename = os.path.basename(video_path)
            actual_video_path = os.path.join('data/videos', video_filename)
            
            if os.path.exists(actual_video_path):
                # Update the item with correct path
                item_copy = item.copy()
                item_copy['video'] = actual_video_path
                valid_samples.append(item_copy)
            else:
                missing_videos.append(video_path)
    
    logger.info(f"Valid samples: {len(valid_samples)}")
    if missing_videos:
        logger.warning(f"Missing {len(missing_videos)} video files")
        for missing in missing_videos[:5]:  # Show first 5
            logger.warning(f"Missing: {missing}")
    
    return valid_samples

def create_overnight_training_config():
    """Optimized training configuration for overnight M3 Max training"""
    return TrainingArguments(
        # Output and logging
        output_dir="./trained_models/blind_navigation_500_videos",
        logging_dir="./logs/tensorboard",
        logging_steps=10,
        logging_strategy="steps",
        
        # Saving strategy - frequent saves for overnight training
        save_strategy="steps",
        save_steps=100,  # Save every 100 steps
        save_total_limit=10,  # Keep last 10 checkpoints
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=200,
        
        # Training parameters
        num_train_epochs=8,  # More epochs for better learning
        per_device_train_batch_size=1,  # Memory constraint
        gradient_accumulation_steps=16,  # Effective batch size = 16
        
        # Learning rate and optimization
        learning_rate=1e-4,  # Slightly lower for stability
        weight_decay=0.01,
        warmup_ratio=0.05,  # 5% warmup for stability
        lr_scheduler_type="cosine",
        
        # Memory optimization
        gradient_checkpointing=True,
        dataloader_num_workers=6,  # M3 Max has good I/O
        remove_unused_columns=False,
        group_by_length=True,
        
        # Model parameters
        model_max_length=1024,
        bf16=True,  # Use bfloat16 for M3 Max
        tf32=False,  # Disable tf32 for M3 Max
        
        # Monitoring
        report_to="tensorboard",
        run_name=f"videollama3_500videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        
        # Stability
        max_grad_norm=1.0,
        seed=42,
        data_seed=42,
        
        # Performance
        dataloader_pin_memory=True,
        ignore_data_skip=True,
        
        # Safety for overnight training
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

def setup_lora_config():
    """LoRA configuration optimized for blind navigation"""
    return LoraConfig(
        r=128,  # Higher rank for complex navigation reasoning
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )

def create_progress_monitor():
    """Create progress monitoring system with video count tracking"""
    class ProgressCallback:
        def __init__(self, logger, total_steps, total_videos, batch_size=16):
            self.logger = logger
            self.total_steps = total_steps
            self.total_videos = total_videos
            self.batch_size = batch_size
            self.start_time = time.time()
            self.videos_per_step = batch_size  # Effective batch size
            
        def calculate_videos_trained(self, current_step):
            """Calculate how many videos have been trained on"""
            videos_trained = min(current_step * self.videos_per_step, self.total_videos)
            return videos_trained
            
        def save_progress_status(self, current_step, epoch):
            """Save current progress to a status file"""
            videos_trained = self.calculate_videos_trained(current_step)
            progress_info = {
                "timestamp": datetime.now().isoformat(),
                "current_step": current_step,
                "total_steps": self.total_steps,
                "current_epoch": epoch,
                "videos_trained": videos_trained,
                "total_videos": self.total_videos,
                "progress_percentage": (current_step / self.total_steps) * 100,
                "videos_percentage": (videos_trained / self.total_videos) * 100,
                "elapsed_hours": (time.time() - self.start_time) / 3600
            }
            
            with open('training_progress.json', 'w') as f:
                json.dump(progress_info, f, indent=2)
            
        def on_step_end(self, args, state, control, **kwargs):
            videos_trained = self.calculate_videos_trained(state.global_step)
            
            # Save progress every step
            self.save_progress_status(state.global_step, state.epoch)
            
            if state.global_step % 50 == 0:  # Every 50 steps
                elapsed = time.time() - self.start_time
                progress = state.global_step / self.total_steps
                eta = (elapsed / progress) - elapsed if progress > 0 else 0
                
                self.logger.info(f"Step {state.global_step}/{self.total_steps} "
                               f"({progress:.1%}) - Videos: {videos_trained}/{self.total_videos} "
                               f"({videos_trained/self.total_videos:.1%}) - ETA: {eta/3600:.1f}h - "
                               f"Loss: {state.log_history[-1].get('train_loss', 'N/A') if state.log_history else 'N/A'}")
                
        def on_epoch_end(self, args, state, control, **kwargs):
            videos_trained = self.calculate_videos_trained(state.global_step)
            self.logger.info(f"Epoch {state.epoch} completed - "
                           f"Step {state.global_step} - Videos trained: {videos_trained}/{self.total_videos}")
            self.save_progress_status(state.global_step, state.epoch)
    
    return ProgressCallback

def main():
    logger = setup_logging()
    logger.info("🚀 Starting VideoLLaMA3 500-Video Overnight Training")
    logger.info("=" * 60)
    
    # System info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Prepare dataset
    logger.info("📊 Preparing 500-video dataset...")
    valid_samples = prepare_500_video_dataset()
    
    if len(valid_samples) < 50:
        logger.error(f"Only {len(valid_samples)} valid samples found. Need at least 50.")
        return False
    
    logger.info(f"✅ Proceeding with {len(valid_samples)} videos for training")
    
    # Training time estimation
    hours, minutes, total_steps = estimate_training_time(
        len(valid_samples), epochs=8, batch_size=16  # effective batch size
    )
    logger.info(f"⏱️  Estimated training time: {hours}h {minutes}m ({total_steps} steps)")
    
    if hours > 12:
        logger.warning("Training may take longer than 12 hours!")
    
    # Setup model path (use the same approach as successful single video training)
    logger.info("🤖 Setting up VideoLLaMA3 model...")
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    vision_encoder = "DAMO-NLP-SG/SigLIP-NaViT"
    
    # Use the videollama3/train.py approach instead of direct loading
    logger.info(f"Base model: {base_model}")
    logger.info(f"Vision encoder: {vision_encoder}")
    logger.info("✅ Model configuration set successfully")
    
    # Setup LoRA
    logger.info("🔧 Setting up LoRA configuration...")
    lora_config = setup_lora_config()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Save valid samples to JSON file for dataset loading
    logger.info("📁 Preparing training data...")
    training_data_file = "training_data_500_videos.json"
    with open(training_data_file, 'w') as f:
        json.dump(valid_samples, f, indent=2)
    
    # Create data arguments
    data_args = DataArguments()
    data_args.data_path = [training_data_file]
    data_args.data_folder = "data/videos"  
    data_args.image_aspect_ratio = "pad"
    data_args.fps = 1
    data_args.max_frames = 16
    data_args.is_multimodal = True
    data_args.mm_max_length = 1024
    
    # Create processor
    logger.info("🔧 Creating video processor...")
    processor = Videollama3Processor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Create dataset
    logger.info("📊 Creating training dataset...")
    dataset = LazySupervisedDataset(
        data_path=[training_data_file],
        vlprocessor=processor,
        data_args=data_args
    )
    
    # Training configuration
    training_args = create_overnight_training_config()
    
    # Create trainer with progress monitoring
    progress_callback = create_progress_monitor()(logger, total_steps, len(valid_samples), batch_size=16)
    
    trainer = VideoLLaMA3Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        callbacks=[progress_callback]
    )
    
    # Start training
    logger.info("🎯 Starting overnight training...")
    logger.info(f"Training on {len(valid_samples)} samples for 8 epochs")
    logger.info("Sleep well! Training will continue overnight 😴")
    
    try:
        train_result = trainer.train()
        
        # Save final model
        final_model_path = "./trained_models/blind_navigation_500_videos_final"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Training metrics: {train_result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("Attempting to save current checkpoint...")
        try:
            trainer.save_model("./trained_models/emergency_checkpoint")
            logger.info("Emergency checkpoint saved")
        except:
            logger.error("Failed to save emergency checkpoint")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Overnight training completed! Ready for GitHub push.")
    else:
        print("\n❌ Training failed. Check logs for details.")