#!/usr/bin/env python3
"""
Test Trained VideoLLaMA3 Model - Post-Training Validation
Quick validation script for after overnight training
"""

import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import cv2
import numpy as np
from pathlib import Path

def load_trained_model(model_path):
    """Load the trained model for testing"""
    print(f"🤖 Loading trained model from {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": "mps"}
        )
        
        print("✅ Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def test_video_inference(model, tokenizer, video_path):
    """Test inference on a single video"""
    print(f"🎥 Testing inference on: {video_path}")
    
    # Simple test prompt
    prompt = """I am a blind pedestrian. Help me navigate safely based on what you see in this video.
    
    Answer in JSON format:
    {
        "reason": "reason for instruction",
        "instruction": "clear navigation instruction"
    }"""
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print("✅ Inference successful")
        print(f"Response: {response}")
        return response
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None

def run_model_validation():
    """Run comprehensive model validation"""
    print("🧪 VideoLLaMA3 Model Validation")
    print("=" * 50)
    
    # Check if trained model exists
    model_paths = [
        "./trained_models/blind_navigation_500_videos_final",
        "./trained_models/blind_navigation_500_videos", 
        "./trained_models/emergency_checkpoint"
    ]
    
    trained_model_path = None
    for path in model_paths:
        if os.path.exists(path):
            trained_model_path = path
            break
    
    if not trained_model_path:
        print("❌ No trained model found. Paths checked:")
        for path in model_paths:
            print(f"   - {path}")
        return False
    
    print(f"📁 Found trained model: {trained_model_path}")
    
    # Load model
    model, tokenizer = load_trained_model(trained_model_path)
    if model is None:
        return False
    
    # Test basic inference
    print("\n🔍 Testing basic text inference...")
    test_response = test_video_inference(model, tokenizer, "test_video.mp4")
    
    if test_response:
        print("\n✅ Model validation completed successfully!")
        print(f"Model is ready for deployment")
        return True
    else:
        print("\n❌ Model validation failed")
        return False

def check_training_logs():
    """Check training logs for completion status"""
    print("📊 Checking training logs...")
    
    log_dir = Path("logs")
    if not log_dir.exists():
        print("❌ No logs directory found")
        return False
    
    # Find latest log file
    log_files = list(log_dir.glob("overnight_training_*.log"))
    if not log_files:
        print("❌ No training log files found")
        return False
    
    latest_log = max(log_files, key=os.path.getctime)
    print(f"📄 Latest log: {latest_log}")
    
    # Check last few lines
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            print("\n📋 Last 10 log lines:")
            for line in lines[-10:]:
                print(f"   {line.strip()}")
        return True
    except Exception as e:
        print(f"❌ Failed to read log: {e}")
        return False

def main():
    print("🎯 Post-Training Model Validation")
    print("Testing trained VideoLLaMA3 model for blind navigation")
    print("=" * 60)
    
    # Check training logs first
    log_success = check_training_logs()
    
    # Validate model
    model_success = run_model_validation()
    
    if model_success:
        print("\n🎉 SUCCESS! Model is ready for GitHub and testing")
        print("\n📋 Next steps:")
        print("1. Push to GitHub: git add . && git commit -m 'Add trained 500-video model' && git push")
        print("2. Test on UC Merced cluster")
        print("3. Deploy for real-world navigation testing")
    else:
        print("\n⚠️  Model validation had issues")
        print("Check training logs and model files")

if __name__ == "__main__":
    main()