
import sys
sys.path.append('/data/dishant.zaveri/MiniGPT-4')

import os
import json
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import torch
from types import SimpleNamespace

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat

args = SimpleNamespace(cfg_path="/data/dishant.zaveri/MiniGPT-4/eval_configs/minigpt4_eval.yaml", options=[])
cfg = Config(args)

cfg.model_path = "/data/dishant.zaveri/MiniGPT-4/checkpoints/minigpt4_13b.pth"
cfg.llama_model = "/data/dishant.zaveri/MiniGPT-4/checkpoints/llama-13b"
cfg.model = "minigpt4"

model_cls = registry.get_model_class(cfg.model)
model = model_cls.from_config(cfg).cuda().eval()
chat = Chat(model, vis_processor=cfg.vis_processor.eval)

dataset = load_dataset("FreedomIntelligence/MileBench", split="OCR_VQA_test")
results = []
save_path = "minigpt4_milebench_results_cam.jsonl"

for item in tqdm(dataset.select(range(5))):
    image_path = item.get("image_path") or item.get("image")
    if not image_path or not os.path.exists(image_path):
        continue
    question = item.get("question", "Describe the image.")
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = chat.vis_processor(image).unsqueeze(0).cuda()
        prompt = question
        chat_state = None
        with torch.no_grad():
            output_text, chat_state = chat.answer(image_tensor, prompt, chat_state)
        kv_cache_full = model.llama_model.model.layers[-1].self_attn.kv_cache if hasattr(model.llama_model.model.layers[-1].self_attn, 'kv_cache') else None
        cam_applied = False
        if kv_cache_full is not None:
            for layer in model.llama_model.model.layers:
                if hasattr(layer.self_attn, 'k_cache') and layer.self_attn.k_cache is not None:
                    layer.self_attn.k_cache = layer.self_attn.k_cache[:, :, -8:, :]
                    layer.self_attn.v_cache = layer.self_attn.v_cache[:, :, -8:, :]
                    cam_applied = True
        cam_response = output_text
        results.append({
            "image": image_path,
            "question": question,
            "response": cam_response,
            "cam_used": cam_applied
        })
    except Exception as e:
        results.append({
            "image": image_path,
            "question": question,
            "response": f"[ERROR] {str(e)}",
            "cam_used": False
        })

with open(save_path, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"Saved to {save_path}")
