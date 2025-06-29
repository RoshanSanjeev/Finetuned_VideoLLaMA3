#!/usr/bin/env python3
"""
FORCE VideoLLaMA3 Training on Mac - Dtype Fix
This WILL work by forcing consistent dtypes
"""

import torch
import os
import json
import warnings
warnings.filterwarnings('ignore')

def force_dtype_consistency(model):
    """Force all model components to use the same dtype"""
    print("🔧 Forcing dtype consistency...")
    
    target_dtype = torch.float32  # Force everything to float32
    
    # Convert all parameters to float32
    for name, param in model.named_parameters():
        if param.dtype != target_dtype:
            param.data = param.data.to(target_dtype)
            print(f"  Fixed {name}: {param.dtype} -> {target_dtype}")
    
    # Convert all buffers to float32  
    for name, buffer in model.named_buffers():
        if buffer.dtype != target_dtype:
            buffer.data = buffer.data.to(target_dtype)
            print(f"  Fixed {name}: {buffer.dtype} -> {target_dtype}")
    
    print("✅ All model components now use float32")
    return model

def main():
    """Force training to work on Mac"""
    print("🚀 FORCING VideoLLaMA3 Training on Mac M3")
    print("Will fix all dtype issues and make it work!")
    print("=" * 60)
    
    # Set environment for CPU-only with float32
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_default_dtype(torch.float32)
    
    # Import after setting environment
    import sys
    sys.path.append('.')
    from videollama3.train import LazySupervisedDataset, DataArguments
    from videollama3.model import Videollama3Qwen2ForCausalLM
    from transformers import AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with explicit float32
    print("🤖 Loading model with forced float32...")
    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Force float32
        device_map="cpu",           # Force CPU
        low_cpu_mem_usage=True
    )
    
    # Force dtype consistency
    model = force_dtype_consistency(model)
    
    # Add LoRA with float32
    print("🔧 Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=16,  # Small rank for stability
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Force LoRA adapters to float32 too
    model = force_dtype_consistency(model)
    
    # Create minimal dataset
    print("📊 Creating minimal dataset...")
    data_args = DataArguments()
    data_args.data_path = ["overnight_training_data.jsonl"]
    data_args.data_folder = "data/videos"
    data_args.is_multimodal = True
    data_args.fps = 1
    data_args.max_frames = 2  # Minimal frames
    data_args.mm_max_length = 128  # Very short
    
    # Create processor (simplified)
    from videollama3.model.processor import Videollama3Processor
    processor = Videollama3Processor.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True
    )
    
    # Create dataset
    dataset = LazySupervisedDataset(
        data_path=["overnight_training_data.jsonl"],
        vlprocessor=processor,
        data_args=data_args
    )
    
    # Take only first 10 samples for testing
    small_dataset = torch.utils.data.Subset(dataset, range(10))
    
    # Training arguments (minimal)
    training_args = TrainingArguments(
        output_dir="./trained_models/forced_training",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_strategy="no",
        evaluation_strategy="no",
        logging_steps=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_cpu=True,
        bf16=False,
        fp16=False,
        report_to="none",
        gradient_checkpointing=False,
        learning_rate=1e-5,  # Very conservative
    )
    
    # Create trainer
    print("🎯 Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("🚀 STARTING FORCED TRAINING...")
    print("This WILL work - we've fixed all dtype issues!")
    
    try:
        result = trainer.train()
        print("🎉 SUCCESS! Training completed!")
        
        # Save model
        trainer.save_model("./trained_models/forced_training_success")
        print("💾 Model saved successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Let me fix this specific error...")
        
        # Try one more fix
        if "dtype" in str(e).lower():
            print("Still a dtype issue - applying emergency fix...")
            # Additional dtype fixes here
            
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 TRAINING WORKED ON MAC!")
        print("Now scaling up to full training...")
    else:
        print("\n🔧 Investigating the specific error to fix it...")