#!/usr/bin/env python3
"""
Train VideoLLaMA3 on a single video with its corresponding annotation.
Perfect for quick testing and validation.
"""

import json
import os
import sys
import argparse
import subprocess
from pathlib import Path

def find_annotation_for_video(video_filename, annotation_json_path):
    """Find the annotation that matches the given video filename."""
    
    print(f"Searching for annotation matching: {video_filename}")
    
    with open(annotation_json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Search for matching annotation
    matching_annotation = None
    for annotation in annotations:
        video_path = annotation.get('video', '')
        if video_filename in video_path:
            matching_annotation = annotation
            break
    
    if matching_annotation:
        print(f"✅ Found matching annotation with ID: {matching_annotation['id']}")
        
        # Extract instruction
        instruction = "None"
        for conv in matching_annotation['conversations']:
            if conv['from'] == 'gpt':
                if 'value_if_spoke' in conv:
                    instruction = conv['value_if_spoke']
                else:
                    try:
                        gpt_response = json.loads(conv['value'])
                        instruction = gpt_response.get('instruction', 'None')
                    except:
                        instruction = conv['value']
                break
        
        print(f"📝 Instruction: {instruction}")
        return matching_annotation, instruction
    else:
        print(f"❌ No annotation found for {video_filename}")
        return None, None

def create_single_video_training_data(video_path, annotation, output_dir):
    """Create training data for a single video."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract instruction
    instruction = "Walk straight ahead."  # default
    for conv in annotation['conversations']:
        if conv['from'] == 'gpt':
            if 'value_if_spoke' in conv:
                instruction = conv['value_if_spoke']
            else:
                try:
                    gpt_response = json.loads(conv['value'])
                    instruction = gpt_response.get('instruction', 'Walk straight ahead.')
                except:
                    instruction = conv['value']
            break
    
    # Create VideoLLaMA3 training format
    training_data = {
        "id": f"single_video_{annotation['id']}",
        "video": [Path(video_path).name],
        "conversations": [
            {
                "from": "human", 
                "value": "<video>You are guiding a blind person. Describe what you see and provide navigation instructions. Focus on immediate obstacles, path directions, and safety information."
            },
            {
                "from": "gpt",
                "value": instruction
            }
        ]
    }
    
    # Save annotation
    jsonl_path = os.path.join(output_dir, "annotations.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(training_data, ensure_ascii=False) + '\n')
    
    # Copy video to training directory
    import shutil
    video_dest = os.path.join(output_dir, Path(video_path).name)
    if not os.path.exists(video_dest):
        shutil.copy2(video_path, video_dest)
    
    print(f"✅ Training data created in: {output_dir}")
    print(f"📁 Video: {video_dest}")
    print(f"📄 Annotation: {jsonl_path}")
    
    return jsonl_path

def train_single_video(video_path, annotation_json_path, model_size="2b", epochs=3):
    """Train on a single video with its annotation."""
    
    video_filename = Path(video_path).name
    output_dir = f"./single_video_training_{Path(video_path).stem}"
    
    print("🎯 Single Video Training Setup")
    print("=" * 50)
    print(f"Video: {video_filename}")
    print(f"Model size: {model_size}")
    print(f"Epochs: {epochs}")
    print(f"Output: {output_dir}")
    print("=" * 50)
    
    # Find matching annotation
    annotation, instruction = find_annotation_for_video(video_filename, annotation_json_path)
    if not annotation:
        print("❌ Cannot proceed without matching annotation")
        return False
    
    # Create training data
    training_dir = f"{output_dir}_data"
    jsonl_path = create_single_video_training_data(video_path, annotation, training_dir)
    
    # Set model configuration - start from base Qwen model like stage 1
    if model_size == "2b":
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"
        batch_size = 1
        grad_accum = 2
        lr = 5e-5
    else:
        model_path = "Qwen/Qwen2.5-7B-Instruct"
        batch_size = 1
        grad_accum = 4
        lr = 2e-5
    
    print(f"\n🚀 Starting training...")
    print(f"Expected time: 10-30 minutes for {epochs} epochs")
    
    # Training command - following stage 1 pattern
    cmd = [
        "python", "videollama3/train.py",
        "--model_type", "videollama3_qwen2",
        "--model_path", model_path,
        "--vision_encoder", "DAMO-NLP-SG/SigLIP-NaViT",
        "--mm_projector_type", "mlp2x_gelu",
        "--mm_attn_implementation", "eager",
        "--data_path", jsonl_path,
        "--data_folder", training_dir,
        "--image_merge_size", "1",
        "--video_merge_size", "2", 
        "--fps", "1",
        "--max_frames", "16",
        "--model_max_length", "1024",
        "--mm_max_length", "512",
        "--bf16", "True",
        "--tf32", "False",
        "--fp16", "False",
        "--output_dir", output_dir,
        "--num_train_epochs", str(epochs),
        "--per_device_train_batch_size", str(batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--eval_strategy", "no",
        "--save_strategy", "epoch",
        "--save_total_limit", "2",
        "--learning_rate", str(lr),
        "--mm_projector_lr", str(lr * 2),
        "--vision_encoder_lr", str(lr * 0.1),
        "--weight_decay", "0.01",
        "--warmup_ratio", "0.1",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "1",
        "--report_to", "none",
        "--run_name", f"single_video_{Path(video_path).stem}",
        "--lora_enable", "True",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--lora_dropout", "0.1",
        "--remove_unused_columns", "False"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("🎉 Training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
        
        # Test the model
        print(f"\n🧪 Testing the trained model...")
        test_cmd = [
            "python", "test_single_video_model.py",
            "--model_path", output_dir,
            "--video_path", video_path
        ]
        
        test_result = subprocess.run(test_cmd, capture_output=True, text=True)
        if test_result.returncode == 0:
            print("✅ Model test successful!")
            print("Generated description:")
            print("-" * 30)
            print(test_result.stdout)
        else:
            print("⚠️ Model test had issues:")
            print(test_result.stderr)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train VideoLLaMA3 on a single video with its annotation")
    parser.add_argument("--video_path", required=True, help="Path to video file")
    parser.add_argument("--annotation_json", 
                       default="data/annotations/split_train.json",
                       help="Path to split_train.json")
    parser.add_argument("--model_size", choices=["2b", "7b"], default="2b",
                       help="Model size to use")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        return
    
    if not os.path.exists(args.annotation_json):
        print(f"❌ Annotation file not found: {args.annotation_json}")
        return
    
    # Run training
    success = train_single_video(
        args.video_path,
        args.annotation_json, 
        args.model_size,
        args.epochs
    )
    
    if success:
        print("\n🎯 Quick Start Guide:")
        print("1. Training completed successfully")
        print("2. Model saved and tested")
        print("3. Ready for blind navigation guidance!")
    else:
        print("\n⚠️ Training encountered issues. Check logs above.")

if __name__ == "__main__":
    main()