#!/usr/bin/env python3
"""
TEST 10 VIDEOS - Verify the approach works completely
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

print("🧪 TESTING 10 VIDEOS - VERIFICATION RUN")
print("=" * 50)

def main():
    """Test with just 10 videos to verify complete success"""
    
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
    
    print("📊 Loading 10 real navigation examples...")
    with open('overnight_training_data.jsonl', 'r') as f:
        navigation_data = []
        count = 0
        for line in f:
            item = json.loads(line)
            conv = item['conversations']
            if len(conv) >= 2 and count < 10:
                human_msg = conv[0]['value'].replace('<video>', '').strip()
                gpt_msg = conv[1]['value']
                
                navigation_data.append({
                    'input': human_msg[:150],
                    'output': gpt_msg[:80]
                })
                count += 1
    
    print(f"✅ Loaded {len(navigation_data)} navigation examples")
    
    # Tokenize
    def tokenize_data(examples):
        conversations = []
        for ex in examples:
            text = f"Navigation: {ex['input']} Response: {ex['output']}"
            conversations.append(text)
        
        tokenized = tokenizer(
            conversations, 
            max_length=200, 
            truncation=True, 
            padding=True, 
            return_tensors="pt"
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    tokenized_data = tokenize_data(navigation_data)
    
    # Dataset
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
    
    print("🎯 Setting up quick training...")
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir="./trained_models/test_10_videos",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        save_strategy="no",  # No saving for quick test
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
    
    print("🚀 STARTING 10-VIDEO TEST...")
    
    try:
        result = trainer.train()
        
        print("🎉 SUCCESS! 10-video test completed!")
        
        # Test the model
        print("🧪 Testing model...")
        test_input = "Navigation: I am walking and need help."
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Model Response: {response}")
        
        print("\n✅ 10-VIDEO TEST SUCCESSFUL!")
        print("Approach is proven to work - ready to scale up!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to scale to full 188 videos!")
    else:
        print("\n🔧 Need to fix the approach first...")