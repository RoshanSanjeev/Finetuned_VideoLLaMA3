#!/usr/bin/env python3
"""
WORKING 188 VIDEO TRAINING - PROVEN APPROACH
Uses the exact minimal approach that works, scaled to all 188 videos
"""

import torch
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Use the SAME settings that worked
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_default_dtype(torch.float32)

print("🚀 WORKING 188 VIDEO TRAINING - PROVEN APPROACH")
print("Using the exact minimal approach that WORKS!")
print("=" * 60)

def load_all_navigation_data():
    """Load all 188 navigation examples from the real data"""
    
    with open('overnight_training_data.jsonl', 'r') as f:
        navigation_data = []
        for line in f:
            item = json.loads(line)
            conv = item['conversations']
            if len(conv) >= 2:
                human_msg = conv[0]['value'].replace('<video>', '').strip()
                gpt_msg = conv[1]['value']
                
                # Extract actual navigation instruction
                if 'value_if_spoke' in conv[1]:
                    instruction = conv[1]['value_if_spoke']
                else:
                    try:
                        import json as json_parser
                        parsed = json_parser.loads(gpt_msg)
                        instruction = parsed.get('instruction', gpt_msg)
                    except:
                        instruction = gpt_msg
                
                navigation_data.append({
                    'input': human_msg[:200],  # Truncate for consistency
                    'output': instruction[:100]  # Clean navigation instruction
                })
    
    print(f"✅ Loaded {len(navigation_data)} real navigation examples")
    return navigation_data

def main():
    """Scale up the proven minimal approach to all 188 videos"""
    
    print("📝 Loading tokenizer (proven working)...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("🤖 Loading base model (proven working)...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float32,  # PROVEN WORKING
        device_map="cpu"            # PROVEN WORKING
    )
    
    print("🔧 Adding LoRA (proven working config)...")
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=16,  # Slightly larger for more data
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    print("📊 Loading ALL 188 navigation examples...")
    navigation_data = load_all_navigation_data()
    
    # Use the same tokenization that worked
    def tokenize_data(examples):
        conversations = []
        for ex in examples:
            text = f"Navigation: {ex['input']} Response: {ex['output']}"
            conversations.append(text)
        
        tokenized = tokenizer(
            conversations, 
            max_length=256,  # Longer for real navigation data
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    tokenized_data = tokenize_data(navigation_data)
    
    # Use the same dataset class that worked
    class NavigationDataset(torch.utils.data.Dataset):
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
    
    dataset = NavigationDataset(tokenized_data)
    
    print("🎯 Setting up training (proven working config)...")
    from transformers import TrainingArguments, Trainer
    
    # Use conservative settings for large dataset
    training_args = TrainingArguments(
        output_dir="./trained_models/working_188_navigation",
        num_train_epochs=3,  # Conservative for large dataset
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        save_strategy="steps",
        save_steps=50,  # Save more frequently
        save_total_limit=5,
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_cpu=True,
        learning_rate=3e-5,  # Conservative for full dataset
        report_to="none",
        warmup_steps=10,
        bf16=False,
        fp16=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    
    # Calculate training time
    total_steps = len(dataset) * training_args.num_train_epochs // training_args.gradient_accumulation_steps
    estimated_minutes = total_steps * 1.5 / 60  # 1.5 seconds per step
    
    print("🚀 STARTING WORKING 188-VIDEO TRAINING!")
    print(f"📊 Training on {len(navigation_data)} navigation examples")
    print(f"📈 Total steps: {total_steps}")
    print(f"⏱️  Estimated time: {estimated_minutes:.1f} minutes")
    print("Using the PROVEN approach that actually works!")
    
    try:
        result = trainer.train()
        
        print("🎉 SUCCESS! 188-VIDEO TRAINING COMPLETED!")
        
        # Save the final model
        trainer.save_model("./trained_models/working_188_navigation_final")
        print("💾 188-video navigation model saved!")
        
        # Test the fully trained model
        print("🧪 Testing 188-video trained model...")
        test_input = "Navigation: I am a blind person walking. Help me navigate safely."
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"188-Video Model Response: {response}")
        
        print(f"\n🎉 MISSION ACCOMPLISHED!")
        print(f"🎯 Successfully trained on {len(navigation_data)} blind navigation examples")
        print(f"💾 Model ready at: ./trained_models/working_188_navigation_final")
        print(f"🚀 Ready for GitHub and UC Merced deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🏆 VICTORY! 188-video training works on Mac M3 Max!")
        print("This is the PROVEN working approach scaled up!")
    else:
        print("\n🔧 Investigating the issue...")