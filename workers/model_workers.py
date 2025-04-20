from workers.baseworker import BaseWorker
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoConfig
)
from utils_lm_eval.modify_llama_cam import convert_kvcache_llama_cam
from PIL import Image
import copy
import torch

class LLaVA(BaseWorker):
    def init_components(self, config):
        model_id = config.model_dir
        llava_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        llama_cfg = llava_cfg.text_config
        if getattr(config, "enable_cam", False):
            llama_cfg.start_ratio  = config.start_ratio
            llama_cfg.recent_ratio = config.recent_ratio

        # processor
        self.processor = LlavaNextProcessor.from_pretrained(model_id, trust_remote_code=True)

        # 1) load model on CPU
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            config=llava_cfg,
            torch_dtype=torch.float16
        ).to("cpu")

        # 2) wrap Llama backbone if requested
        if getattr(config, "enable_cam", False):
            ckpt = copy.deepcopy(self.model.language_model.state_dict())
            self.model.language_model = convert_kvcache_llama_cam(
                self.model.language_model, llama_cfg
            )
            self.model.language_model.load_state_dict(ckpt)

        # 3) then move the full model to GPU once
        self.model = self.model.to(config.device)
        self.model.eval()

        self.tokenizer = self.processor.tokenizer
        self.device    = self.model.device

    def forward(self, questions, image_paths, device, gen_kwargs):
        answers = []
        for q, imgs in zip(questions, image_paths):
            # build chat prompt
            parts = q.split("<ImageHere>")
            content = []
            for i, txt in enumerate(parts):
                txt = txt.strip()
                if txt:
                    content.append({"type":"text","text":txt})
                if i < len(imgs):
                    content.append({"type":"image"})
            conv   = [{"role":"user","content":content}]
            prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)

            # load images
            pil_imgs = [Image.open(p).convert("RGB") for p in imgs]
            inputs   = self.processor(images=pil_imgs, text=prompt, return_tensors="pt")

            # move to device, preserving integer dtypes for input_ids/attention_mask
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)

            # cast only float inputs (pixel_values) to half
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            with torch.no_grad():
                out_ids = self.model.generate(**inputs, **gen_kwargs)

            txt = self.processor.tokenizer.decode(
                out_ids[0][2:], skip_special_tokens=True
            ).split("assistant:")[-1].strip()
            answers.append(txt)
            print("→", txt)

        return answers
