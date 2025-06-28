#!/usr/bin/env python3
"""
Script to test a fine-tuned VideoLLaMA3 model on video input.
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoProcessor
sys.path.append('./')

from videollama3.model import *
from videollama3.mm_utils import load_video
from videollama3.model.processor import Videollama3Processor

def load_model(model_path, model_type="videollama3_qwen2"):
    """Load the fine-tuned model and processor."""
    
    print(f"Loading model from: {model_path}")
    
    # Load config and model
    config = VLLMConfigs[model_type].from_pretrained(model_path)
    model = VLLMs[model_type].from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    # Load processor
    processor = Videollama3Processor(
        model.get_vision_encoder().image_processor,
        tokenizer
    )
    
    model.eval()
    return model, processor, tokenizer

def generate_description(model, processor, tokenizer, video_path, prompt=None, max_frames=64, fps=1):
    """Generate description for a video."""
    
    if prompt is None:
        prompt = "Describe this video as if you are narrating for a blind person. Include all visual details, movements, objects, and spatial relationships."
    
    # Load video
    print(f"Loading video: {video_path}")
    frames, timestamps = load_video(video_path, fps=fps, max_frames=max_frames)
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "num_frames": len(frames), "timestamps": timestamps},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process input
    inputs = processor(
        images=[frames],
        text=messages,
        merge_size=2,
        return_tensors="pt"
    )
    
    # Move to device
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(model.device)
    
    # Generate
    print("Generating description...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned VideoLLaMA3 model")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--video_path", required=True, help="Path to test video")
    parser.add_argument("--prompt", help="Custom prompt (optional)")
    parser.add_argument("--max_frames", type=int, default=64, help="Maximum frames to process")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second")
    parser.add_argument("--model_type", default="videollama3_qwen2", help="Model type")
    
    args = parser.parse_args()
    
    # Load model
    model, processor, tokenizer = load_model(args.model_path, args.model_type)
    
    # Generate description
    description = generate_description(
        model, processor, tokenizer, 
        args.video_path, args.prompt, 
        args.max_frames, args.fps
    )
    
    print("=" * 50)
    print("GENERATED DESCRIPTION:")
    print("=" * 50)
    print(description)
    print("=" * 50)

if __name__ == "__main__":
    main()