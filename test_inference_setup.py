#!/usr/bin/env python3
"""
Test VideoLLaMA3 inference setup to verify everything works before training
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import sys

def test_videollama3_setup():
    """Test if VideoLLaMA3 can be loaded and run inference"""
    print("🔧 Testing VideoLLaMA3 Setup")
    print("=" * 50)
    
    # Check if we can download and load the model
    try:
        print("📥 Loading VideoLLaMA3-2B model...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model_path = "DAMO-NLP-SG/VideoLLaMA3-2B"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": device},
            torch_dtype=torch.float16,
            attn_implementation="eager",  # M3 Max compatibility
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Test text-only conversation
        print("\n🧪 Testing text conversation...")
        conversation = [
            {"role": "system", "content": "You are a helpful navigation assistant for blind pedestrians."},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Hello, can you help me navigate?"},
                ]
            },
        ]
        
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(f"✅ Response: {response}")
        
        print("\n🎯 VideoLLaMA3 setup is working correctly!")
        print("Ready for training and video analysis.")
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_videollama3_setup()
    if success:
        print("\n✅ Your M3 Max can run VideoLLaMA3!")
        print("Proceed with training or video analysis.")
        sys.exit(0)
    else:
        print("\n❌ Setup needs attention.")
        sys.exit(1)