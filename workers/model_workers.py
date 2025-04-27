from workers.baseworker import BaseWorker
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoConfig,
    SinkCache
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
        

        # processor
        self.processor = LlavaNextProcessor.from_pretrained(model_id, trust_remote_code=True)

        # store config for streaming
        self.config = config

        # loading model on CPU
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            config=llava_cfg,
            torch_dtype=torch.float16
        ).to("cpu")

        # wrapping Llama backbone if requested (CAM)
        if getattr(config, "enable_streaming", False):
            # no CaM wrapper here Streaming LLM is used in forward()
            pass
        elif getattr(config, "enable_cam", False):
             ckpt = copy.deepcopy(self.model.language_model.state_dict())
             self.model.language_model = convert_kvcache_llama_cam(
                 self.model.language_model, llama_cfg
             )
             self.model.language_model.load_state_dict(ckpt)

        # moving the full model to target device
        self.model = self.model.to(config.device)
        self.model.eval()

        self.tokenizer = self.processor.tokenizer
        self.device    = self.model.device

    def forward(self, questions, image_paths, device, gen_kwargs):
        answers = []
        for q, imgs in zip(questions, image_paths):
            # building chat prompt
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

            # load and preprocess images
            pil_imgs = [Image.open(p).convert("RGB") for p in imgs]
            inputs   = self.processor(images=pil_imgs, text=prompt, return_tensors="pt").to(self.device)

            # cast floats to half
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()

            # streaming or regular generate
            with torch.no_grad():
                if getattr(self.config, "enable_streaming", False):
                    # Streaming LLM for each example
                    cache = SinkCache(
                        window_length=self.config.stream_window_length,
                        num_sink_tokens=self.config.num_sink_tokens
                    )
                    out_ids = self.model.generate(
                        **inputs,
                        use_cache=True,
                        past_key_values=cache,
                        **gen_kwargs
                    )
                else:
                    out_ids = self.model.generate(
                        **inputs,
                        **gen_kwargs
                    )

            # decode skipping special tokens and assistant label
            seq = out_ids[0]
            # remove BOS token
            seq = seq[2:]
            txt = self.tokenizer.decode(seq, skip_special_tokens=True)
            txt = txt.split("assistant:")[-1].strip()
            answers.append(txt)
            print("→", txt)

            # cleanup
            del out_ids, inputs, prompt, pil_imgs
            torch.cuda.empty_cache()

        return answers