
# Updated MiniGPT-4 Evaluation Script with CaM (Cache Merging) Logic

import sys
sys.path.append('/data/dishant.zaveri/MiniGPT-4')

import os
import json
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import torch

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat

# === Load MiniGPT-4 config === #
config_path = "/data/dishant.zaveri/MiniGPT-4/eval_configs/minigpt4_eval.yaml"
cfg = Config(config_path)

# Override paths manually
cfg.model_path = "/data/dishant.zaveri/MiniGPT-4/checkpoints/minigpt4_13b.pth"
cfg.llama_model = "/data/dishant.zaveri/MiniGPT-4/checkpoints/llama-13b"
cfg.model = "minigpt4"

# === Initialize model === #
model_cls = registry.get_model_class(cfg.model)
model = model_cls.from_config(cfg).cuda().eval()
chat = Chat(model, vis_processor=cfg.vis_processor.eval)

# === Load dataset (sample) === #
dataset = load_dataset("FreedomIntelligence/MileBench", split="OCR_VQA_test")
results = []
save_path = "minigpt4_cam_milebench_results.jsonl"

# Placeholder: integrate CaM logic here
def apply_cam(image_tensor, model, prompt):
    # This is a placeholder for Cache Merging (CaM) logic
    # In practice, modify attention layers / cache usage here
    chat_state = None
    response, chat_state = chat.answer(image_tensor, prompt, chat_state)
    return response

for item in tqdm(dataset.select(range(5))):  # use more for full test
    image_path = item.get("image_path") or item.get("image")
    if not image_path or not os.path.exists(image_path):
        continue

    question = item.get("question", "Describe the image.")
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = chat.vis_processor(image).unsqueeze(0).cuda()

        # Use modified inference with CaM logic
        response = apply_cam(image_tensor, model, question)

        results.append({
            "image": image_path,
            "question": question,
            "response": response
        })

    except Exception as e:
        results.append({
            "image": image_path,
            "question": question,
            "response": f"[ERROR] {str(e)}"
        })

# Save to JSONL
with open(save_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"✅ Results saved to {save_path}")
