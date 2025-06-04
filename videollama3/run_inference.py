import os
import sys
import types
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Optional: quiet tokenizer fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Patch to skip decord for now
sys.modules['decord'] = types.ModuleType("decord")
sys.modules['decord'].VideoReader = lambda x: None
sys.modules['decord'].cpu = lambda: None

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "DAMO-NLP-SG/VideoLLaMA3-2B"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float32,
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

video_path = "./videos/sample.mp4"  # ← ensure this exists
question = "You are a helpful assistant analyzing videos involving visually impaired individuals. The video may be in first-person (you are helping the person) or third-person (you are watching the person navigating or being assisted)."

conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant analyzing videos involving visually impaired individuals. The video may be in first-person (you are helping the person) or third-person (you are watching the person navigating or being assisted)."
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 180}},
            {"type": "text", "text": question}
        ]
    }
]

inputs = processor(
    conversation=conversation,
    add_system_prompt=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

# Fix tensor types for CPU
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].float()

print("Tensor dtype before generate:", inputs.get("pixel_values", torch.tensor([])).dtype)

output_ids = model.generate(**inputs, max_new_tokens=256)
response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

print("\nResponse:")
print(response)
