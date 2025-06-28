import os
import sys
import json
import types
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Give this run a name (the folder will be created under runs/)
run_name = "run_single_video_test"

# If you want to process exactly one video, put its filename (relative to ./videos/)
# Otherwise, set single_video_filename = None to process all .mp4 files in ./videos/
single_video_filename = "videos_data_2024_10_25_03_04_21_session_2_Town05_wk1_9_25.mp4"

# Your system prompt (edit before each run)
system_prompt = (
    "Answer in English. You are assisting a visually impaired person navigate through a complex urban intersection. "
    "The pedestrian is in open terrain"
    "Please provide clear, spoken instructions in second-person format, taking into account any relevant "
    "environmental cues or obstacles that may impact the pedestrian's ability to cross the street. "
    "Be sure to provide specific guidance on how to navigate around any potential hazards or traffic signals. "
    "Always respond in English."
)

question = (
    "Answer in English. You are assisting a visually impaired person navigate through a complex urban intersection. "
    "The pedestrian is currently approaching the intersection and needs guidance on how to safely cross. "
    "Please provide clear, spoken instructions in second-person format, taking into account any relevant "
    "environmental cues or obstacles that may impact the pedestrian's ability to cross the street. "
    "Be sure to provide specific guidance on how to navigate around any potential hazards or traffic signals. "
    "Answer in English."
)


# ──────────────────────────────────────────────────────────────────────────────
# END CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

# Figure out where we are
base_dir = os.getcwd()
runs_dir = os.path.join(base_dir, "runs")
os.makedirs(runs_dir, exist_ok=True)

current_run_dir = os.path.join(runs_dir, run_name)
if os.path.exists(current_run_dir):
    print(f"Warning: run folder already exists ({current_run_dir}), outputs may be overwritten.")
else:
    os.makedirs(current_run_dir)

# Save the prompt (system + user) into a simple text file
prompt_file = os.path.join(current_run_dir, "prompt.txt")
with open(prompt_file, "w") as f:
    f.write("--- System Prompt ---\n")
    f.write(system_prompt + "\n\n")
    f.write("--- User Question ---\n")
    f.write(question + "\n")

print(f"Created run folder: {current_run_dir}")
print(f"Saved prompt to: {prompt_file}\n")

# ──────────────────────────────────────────────────────────────────────────────
# MODEL + PROCESSOR SETUP
# ──────────────────────────────────────────────────────────────────────────────

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# If decord is not installed (common on M1/M2/M3 Macs), patch it out
sys.modules['decord'] = types.ModuleType("decord")
sys.modules['decord'].VideoReader = lambda x: None
sys.modules['decord'].cpu = lambda: None

# Pick MPS if available, else fall back to CPU
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

model_path = "DAMO-NLP-SG/VideoLLaMA3-2B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto" if device in ("cuda", "mps") else None,
    torch_dtype=torch.float32,
)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# ──────────────────────────────────────────────────────────────────────────────
# GATHER VIDEO(S)
# ──────────────────────────────────────────────────────────────────────────────

video_dir = os.path.join(base_dir, "videos")
if not os.path.isdir(video_dir):
    raise FileNotFoundError(f"Video directory not found: {video_dir}")

if single_video_filename:
    video_files = [single_video_filename]
else:
    video_files = [f for f in os.listdir(video_dir) if f.lower().endsWith(".mp4")]
    video_files.sort()

if not video_files:
    raise FileNotFoundError("No .mp4 files found under videos/")

# ──────────────────────────────────────────────────────────────────────────────
# RUN INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

all_predictions = []

for idx, filename in enumerate(video_files):
    video_path = os.path.join(video_dir, filename)
    if not os.path.exists(video_path):
        print(f"Skipping (not found): {video_path}")
        continue

    print(f"Processing ({idx+1}/{len(video_files)}): {video_path}")

    conversation = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                # {"type": "video", "video": {"video_path": video_path, "fps": 0.5, "max_frames": 1}},
                {"type": "text", "text": question},
            ]
        }
    ]


    # Prepare inputs (the MPS or CPU tensors come back here)
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Move all tensors onto device (CPU/MPS/CUDA)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    if "pixel_values" in inputs:
        # ensure float32 on MPS
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

    # Generate one answer
    output_ids = model.generate(**inputs, max_new_tokens=256)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # Append to our JSON‐output list
    entry = {
        "id": idx,
        "video_id": os.path.splitext(filename)[0],
        "video": video_path,
        "prediction": response,
        "spoken_reason": "None"
    }
    all_predictions.append(entry)

# Write out all predictions in one JSON file
predictions_file = os.path.join(current_run_dir, "predictions.json")
with open(predictions_file, "w") as f:
    json.dump(all_predictions, f, indent=4)

print(f"\nFinished run. Saved {len(all_predictions)} prediction(s) to:")
print(predictions_file)
