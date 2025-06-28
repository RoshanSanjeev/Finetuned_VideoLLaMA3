#!/usr/bin/env python3
"""
Script to prepare the blind navigation dataset for VideoLLaMA3 training.
Converts split_train.json format to VideoLLaMA3 training format.
"""

import json
import os
import argparse
from pathlib import Path
import random

def convert_annotation_format(input_json_path, video_base_dir, output_dir, num_samples=None):
    """
    Convert blind navigation annotations to VideoLLaMA3 format.
    
    Args:
        input_json_path (str): Path to split_train.json
        video_base_dir (str): Base directory containing video files
        output_dir (str): Output directory for training data
        num_samples (int): Number of samples to use (None for all)
    """
    
    # Load the annotation data
    print(f"Loading annotations from: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"Found {len(annotations)} annotations")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter and sample if needed
    if num_samples and num_samples < len(annotations):
        annotations = random.sample(annotations, num_samples)
        print(f"Randomly sampled {num_samples} annotations")
    
    # Convert to VideoLLaMA3 format
    converted_data = []
    missing_videos = []
    
    for i, item in enumerate(annotations):
        # Extract video path
        video_relative_path = item['video']
        
        # Check multiple possible video locations
        possible_video_paths = [
            os.path.join(video_base_dir, os.path.basename(video_relative_path)),
            os.path.join(video_base_dir, video_relative_path),
            os.path.join("videollama3/videos", os.path.basename(video_relative_path)),
            video_relative_path
        ]
        
        video_exists = False
        final_video_path = None
        
        for video_path in possible_video_paths:
            if os.path.exists(video_path):
                video_exists = True
                final_video_path = os.path.basename(video_path)  # Use just filename for training
                break
        
        if not video_exists:
            missing_videos.append(video_relative_path)
            continue
        
        # Extract instruction from conversations
        instruction = None
        for conv in item['conversations']:
            if conv['from'] == 'gpt':
                # Use value_if_spoke if available, otherwise parse JSON
                if 'value_if_spoke' in conv:
                    instruction = conv['value_if_spoke']
                else:
                    try:
                        gpt_response = json.loads(conv['value'])
                        instruction = gpt_response.get('instruction', conv['value'])
                    except:
                        instruction = conv['value']
                break
        
        if not instruction:
            continue
        
        # Create VideoLLaMA3 format
        converted_item = {
            "id": f"blind_nav_{item['id']}",
            "video": [final_video_path],
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
        
        converted_data.append(converted_item)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(annotations)} annotations")
    
    # Save converted data
    output_jsonl = os.path.join(output_dir, "annotations.jsonl")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n=== Conversion Results ===")
    print(f"Total annotations: {len(annotations)}")
    print(f"Successfully converted: {len(converted_data)}")
    print(f"Missing videos: {len(missing_videos)}")
    print(f"Output file: {output_jsonl}")
    
    if missing_videos:
        print(f"\nFirst 10 missing videos:")
        for video in missing_videos[:10]:
            print(f"  - {video}")
    
    return output_jsonl, len(converted_data), len(missing_videos)

def create_video_symlinks(video_source_dir, training_data_dir):
    """Create symlinks for videos in the training data directory."""
    
    video_dir = os.path.join(training_data_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    if not os.path.exists(video_source_dir):
        print(f"Warning: Video source directory not found: {video_source_dir}")
        return
    
    # Find all MP4 files
    video_files = list(Path(video_source_dir).glob("*.mp4"))
    
    for video_file in video_files:
        link_path = os.path.join(video_dir, video_file.name)
        if not os.path.exists(link_path):
            os.symlink(video_file.absolute(), link_path)
    
    print(f"Created symlinks for {len(video_files)} videos in {video_dir}")

def main():
    parser = argparse.ArgumentParser(description="Prepare blind navigation dataset for VideoLLaMA3")
    parser.add_argument("--annotation_json", 
                       default="data/annotations/split_train.json",
                       help="Path to split_train.json file")
    parser.add_argument("--video_dir", 
                       default="videollama3/videos",
                       help="Directory containing video files")
    parser.add_argument("--output_dir", 
                       default="./blind_navigation_training_data",
                       help="Output directory for training data")
    parser.add_argument("--num_samples", type=int, 
                       help="Number of samples to use (default: all)")
    parser.add_argument("--create_symlinks", action="store_true",
                       help="Create symlinks for videos in training directory")
    
    args = parser.parse_args()
    
    # Convert annotations
    output_jsonl, converted_count, missing_count = convert_annotation_format(
        args.annotation_json,
        args.video_dir,
        args.output_dir,
        args.num_samples
    )
    
    # Create video symlinks if requested
    if args.create_symlinks:
        create_video_symlinks(args.video_dir, args.output_dir)
    
    print(f"\n=== Training Setup ===")
    print(f"Training data ready in: {args.output_dir}")
    print(f"Use --data_path {output_jsonl}")
    print(f"Use --data_folder {args.output_dir}")
    
    # Provide recommendations based on dataset size
    if converted_count > 10000:
        print(f"\n=== Training Recommendations ===")
        print(f"Large dataset ({converted_count} samples) - consider:")
        print(f"- Start with 1000-5000 samples for initial testing")
        print(f"- Use cloud training (Google Colab Pro, Vast.ai, etc.)")
        print(f"- Enable gradient checkpointing and LoRA")
    elif converted_count > 1000:
        print(f"\n=== Training Recommendations ===")
        print(f"Medium dataset ({converted_count} samples) - good for:")
        print(f"- Local training with M3 Max (overnight)")
        print(f"- Cloud training for faster results")
    else:
        print(f"\n=== Training Recommendations ===")
        print(f"Small dataset ({converted_count} samples) - perfect for:")
        print(f"- Local testing and development")
        print(f"- Quick training iterations")

if __name__ == "__main__":
    main()