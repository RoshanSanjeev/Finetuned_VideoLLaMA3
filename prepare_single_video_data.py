#!/usr/bin/env python3
"""
Script to prepare training data for fine-tuning VideoLLaMA3 on a single video.
This script creates the proper JSON format for training data.
"""

import json
import os
import argparse
from pathlib import Path

def create_training_data(video_path, description, output_dir="./training_data"):
    """
    Create training data in the format expected by VideoLLaMA3.
    
    Args:
        video_path (str): Path to the video file
        description (str): Description/narration for the video
        output_dir (str): Directory to save the training data
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video filename without extension for ID
    video_name = Path(video_path).stem
    
    # Create the training data format
    training_data = {
        "id": video_name,
        "video": [Path(video_path).name],  # Just the filename, not full path
        "conversations": [
            {
                "from": "human",
                "value": "<video>Describe this video as if you are narrating for a blind person. Include all visual details, movements, objects, and spatial relationships."
            },
            {
                "from": "gpt", 
                "value": description
            }
        ]
    }
    
    # Write to JSONL format (each line is a JSON object)
    jsonl_path = os.path.join(output_dir, "annotations.jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(training_data, ensure_ascii=False) + '\n')
    
    print(f"Training data saved to: {jsonl_path}")
    print(f"Data folder should contain: {output_dir}")
    print(f"Video file should be copied to: {os.path.join(output_dir, Path(video_path).name)}")
    
    return jsonl_path, output_dir

def main():
    parser = argparse.ArgumentParser(description="Prepare single video training data for VideoLLaMA3")
    parser.add_argument("--video_path", required=True, help="Path to the video file")
    parser.add_argument("--description", required=True, help="Video description/narration")
    parser.add_argument("--output_dir", default="./training_data", help="Output directory for training data")
    parser.add_argument("--copy_video", action="store_true", help="Copy video to output directory")
    
    args = parser.parse_args()
    
    # Create training data
    jsonl_path, output_dir = create_training_data(
        args.video_path, 
        args.description, 
        args.output_dir
    )
    
    # Optionally copy video to training directory
    if args.copy_video:
        import shutil
        video_name = Path(args.video_path).name
        dest_video_path = os.path.join(output_dir, video_name)
        shutil.copy2(args.video_path, dest_video_path)
        print(f"Video copied to: {dest_video_path}")
    
    print("\nNext steps:")
    print(f"1. Ensure your video is in: {output_dir}")
    print(f"2. Run the training script with --data_path {jsonl_path} --data_folder {output_dir}")

if __name__ == "__main__":
    main()