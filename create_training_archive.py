#!/usr/bin/env python3
"""
Create Training Video Archive
Creates a ZIP file with only the 188 videos needed for training
"""

import json
import os
import zipfile
from pathlib import Path

def create_training_archive():
    """Create ZIP archive with only the videos used in training"""
    
    print("📦 Creating training video archive...")
    
    # Load training data to get video list
    with open('overnight_training_data.jsonl', 'r') as f:
        training_videos = []
        for line in f:
            item = json.loads(line)
            video_path = item['video'][0]  # Get video filename from list
            training_videos.append(video_path)
    
    print(f"Found {len(training_videos)} training videos")
    
    # Create ZIP archive
    archive_name = f"training_videos_{len(training_videos)}.zip"
    
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        videos_found = 0
        videos_missing = 0
        
        for video_file in training_videos:
            video_path = os.path.join('data/videos', video_file)
            
            if os.path.exists(video_path):
                # Add to archive with original structure
                zipf.write(video_path, video_file)
                videos_found += 1
                if videos_found % 20 == 0:
                    print(f"  Archived {videos_found} videos...")
            else:
                videos_missing += 1
                print(f"  Missing: {video_file}")
    
    # Show summary
    file_size = os.path.getsize(archive_name) / (1024**3)  # GB
    
    print(f"\n✅ Archive created: {archive_name}")
    print(f"📊 Videos archived: {videos_found}")
    print(f"❌ Videos missing: {videos_missing}")  
    print(f"💾 Archive size: {file_size:.2f} GB")
    
    print(f"\n📋 Next steps:")
    print(f"1. Transfer {archive_name} to your PC")
    print(f"2. Extract in the data/videos/ directory")
    print(f"3. Run training: python train_188_videos_overnight.py")
    
    return archive_name

if __name__ == "__main__":
    create_training_archive()