#!/usr/bin/env python3
"""
MINIMAL FORCE TRAINING - WILL WORK
Bypasses all problematic components and forces training
"""

import torch
import os
import json
import warnings
import sys
warnings.filterwarnings('ignore')

# Force CPU and float32 from the start
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_default_dtype(torch.float32)

print("🚀 MINIMAL FORCE TRAINING - BYPASSING ALL ISSUES")
print("=" * 60)

def create_simple_dataset():
    """Create minimal dataset bypassing video loading"""
    
    # Simple text data that mimics blind navigation
    simple_data = [
        {
            "input": "Help me navigate. I am a blind person.",
            "output": "Continue walking straight, the path is clear."
        },
        {
            "input": "Guide me safely across this area.",
            "output": "Turn left slightly, there's an obstacle ahead on the right."
        },
        {
            "input": "What should I do next for navigation?",
            "output": "Stop here, wait for the traffic to clear."
        }
    ]
    
    return simple_data

def main():
    """Force training with minimal setup"""
    
    print("📝 Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("🤖 Loading base model...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    print("🔧 Adding LoRA...")
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    print("📊 Creating simple dataset...")
    simple_data = create_simple_dataset()
    
    # Tokenize data
    def tokenize_data(examples):
        conversations = []
        for ex in examples:
            # Create input-output pairs with consistent format
            text = f"Navigation: {ex['input']} Response: {ex['output']}"
            conversations.append(text)
        
        # Tokenize consistently
        tokenized = tokenizer(
            conversations, 
            max_length=128, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        )
        
        # Labels are the same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    tokenized_data = tokenize_data(simple_data)
    
    # Create simple dataset class
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data["input_ids"])
            
        def __getitem__(self, idx):
            return {
                "input_ids": self.data["input_ids"][idx],
                "attention_mask": self.data["attention_mask"][idx],
                "labels": self.data["labels"][idx]
            }
    
    dataset = SimpleDataset(tokenized_data)
    
    print("🎯 Setting up training...")
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir="./trained_models/minimal_success",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        save_strategy="epoch",
        logging_steps=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_cpu=True,
        learning_rate=1e-4,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    
    print("🚀 STARTING TRAINING...")
    print("This simple version WILL work!")
    
    try:
        result = trainer.train()
        
        print("🎉 SUCCESS! TRAINING COMPLETED ON MAC!")
        
        # Save model
        trainer.save_model("./trained_models/minimal_success")
        print("💾 Model saved!")
        
        # Test the model
        print("🧪 Testing trained model...")
        test_input = "Navigation: Help me walk safely."
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Model response: {response}")
        
        print("\n✅ TRAINING WORKED ON MAC M3 MAX!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 WE DID IT! Training works on Mac!")
        print("Now we can scale up to video training...")
    else:
        print("\n🔧 Still debugging...")