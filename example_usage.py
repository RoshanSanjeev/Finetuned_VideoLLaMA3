#!/usr/bin/env python3
"""
Example usage of the single video fine-tuning scripts for VideoLLaMA3.
This demonstrates how to fine-tune on one video with a description.
"""

import os
import subprocess

def main():
    # Example configuration
    video_path = "videollama3/videos/videos_data_2024_10_25_03_04_21_session_2_Town05_wk1_9_25.mp4"
    
    # Example description for blind person narration
    description = """
    The video shows a street scene from a pedestrian's perspective. 
    There are concrete sidewalks on both sides of a paved road. 
    Several buildings line the street, including what appears to be 
    residential and commercial structures. Trees and vegetation are 
    visible along the sidewalks. The camera appears to be moving 
    forward along the street, providing a first-person view of 
    walking through this urban environment. The lighting suggests 
    it's daytime with natural sunlight illuminating the scene.
    """
    
    model_size = "2b"  # or "7b" for larger model
    
    print("=" * 60)
    print("VideoLLaMA3 Single Video Fine-tuning Example")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Model size: {model_size}")
    print(f"Description length: {len(description)} characters")
    print("=" * 60)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")
        return
    
    # Run the training script
    print("Starting fine-tuning process...")
    try:
        result = subprocess.run([
            "./train_single_video.sh",
            video_path,
            description,
            model_size
        ], check=True, capture_output=True, text=True)
        
        print("Training completed successfully!")
        print("Output:", result.stdout)
        
        # Test the model
        print("\n" + "=" * 60)
        print("Testing the fine-tuned model...")
        print("=" * 60)
        
        test_result = subprocess.run([
            "python", "test_single_video_model.py",
            "--model_path", "./single_video_checkpoint",
            "--video_path", video_path
        ], check=True, capture_output=True, text=True)
        
        print("Test completed!")
        print("Output:", test_result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print("ERROR: Training script not found. Make sure train_single_video.sh is executable.")
        print("Run: chmod +x train_single_video.sh")

if __name__ == "__main__":
    main()