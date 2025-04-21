from workers.baseworker import *
import sys
from PIL import Image
import torch
from haystack.components.generators import HuggingFaceLocalGenerator
import gc
import numpy as np
from workers.modified_llava_cam import convert_kvcache_qwen_cam, local_cam_mask, Qwen2AttentionCam
from transformers import SinkCache, OffloadedCache
######################## Multi-image application ########################


class LLaVA(BaseWorker):

    def init_components(self, config):
        sys.path.insert(0, '/path/to/LLaVA/packages/')
        from llava.model.builder import load_pretrained_model
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

        self.tokenizer, self.model, self.processor, context_len = load_pretrained_model(
            model_path=config.model_dir,
            model_base=None,
            model_name=config.model_dir,
            device_map='cuda',
        )

        if getattr(self.model.config, 'mm_use_im_start_end', False):
            self.single_img_tokens = DEFAULT_IM_START_TOKEN + \
                DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            self.single_img_tokens = DEFAULT_IMAGE_TOKEN

        self.conv_temp = conv_templates["llava_llama_2"]
        stop_str = self.conv_temp.sep if self.conv_temp.sep_style != SeparatorStyle.TWO else self.conv_temp.sep2
        self.keywords = [stop_str]

        self.model.eval()

    def forward(self, questions, image_paths, device, gen_kwargs):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria

        answers = []
        for question, images_path in zip(questions, image_paths):
            conv = self.conv_temp.copy()

            # Multi-image
            image_tensor = process_images(
                [Image.open(image_path).convert('RGB')
                 for image_path in images_path],
                self.processor, self.model.config
            ).to(device)

            # NOTE: handle the special cases in CLEVR-Change dataset
            question = question.replace(
                '<ImageHere><ImageHere>', '<ImageHere>\n<ImageHere>\n')
            input_prompt = question.replace(
                '<ImageHere>', self.single_img_tokens)

            conv.append_message(conv.roles[0], input_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt=prompt,
                tokenizer=self.tokenizer,
                image_token_index=IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    use_cache=True,
                    stopping_criteria=[KeywordsStoppingCriteria(
                        self.keywords, self.tokenizer, input_ids)],
                    **gen_kwargs
                )
            answer = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True).strip()
            answers.append(answer)

        return answers


class LLaVA_next(BaseWorker):

    def init_components(self, config):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            config.model_dir, device_map='cuda', torch_dtype=torch.float16, use_flash_attention_2=True)
        self.processor = LlavaNextProcessor.from_pretrained(config.model_dir, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        self.single_img_tokens = self.processor.image_token

        self.model.eval()

    def forward(self, questions, image_paths, device, gen_kwargs):

        answers = []
        print("Actual images being used: ",np.mean([len(i) for i in image_paths]))
        for question, images_path in zip(questions, image_paths):

            # NOTE: handle the special cases in CLEVR-Change dataset
            question = question.replace(
                '<ImageHere><ImageHere>', '<ImageHere>\n<ImageHere>\n')
            input_prompt = question.replace(
                '<ImageHere>', self.single_img_tokens)
            
            conv = [
                {

                    "role": "user",
                    "content": [
                        {"type": "text", 
                         "text": input_prompt},
                        # *([{"type": "image"}] * len(images_path))
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
            # Multi-image
            images = [Image.open(image_path).convert('RGB')
                 for image_path in images_path]

            inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(device)
            input_ids_len = inputs['input_ids'].shape[1]
            print("processed inputs ", inputs['input_ids'].shape)

            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            output_ids = self.model.generate(
                **inputs,
                use_cache=True,
                **gen_kwargs
            )
                # print("model generate")
            # print("output")
            # print(output_ids.shape, output_ids.device)
            output_ids = output_ids[0, input_ids_len:]
            answer = self.processor.decode(output_ids, skip_special_tokens=True).strip()

            answers.append(answer)

        return answers


class LLaVA_oneVision(BaseWorker):

    def init_components(self, config):
        from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            config.model_dir, device_map='cuda', torch_dtype=torch.float16, use_flash_attention_2=True)
        self.processor = LlavaOnevisionProcessor.from_pretrained(config.model_dir, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        self.single_img_tokens = "<|im_start|>"+self.processor.image_token+"<|im_end|>"

        self.model.eval()

        # generator = HuggingFaceLocalGenerator(model=config.model_dir,
        #                               generation_kwargs={
        #                                 "max_new_tokens": 512,
        #                                 "temperature": 1,
        #                                 },
        #                                 huggingface_pipeline_kwargs={"device_map": "auto", 
        #                                                              })

        # generator.warm_up()

    def forward(self, questions, image_paths, device, gen_kwargs):

        answers = []
        print("Actual images being used: ",np.mean([len(i) for i in image_paths]))
        for question, images_path in zip(questions, image_paths):

            # NOTE: handle the special cases in CLEVR-Change dataset
            question = question.replace(
                '<ImageHere><ImageHere>', '<ImageHere>\n<ImageHere>\n')
            input_prompt = question.replace(
                '<ImageHere>', self.single_img_tokens)
            # print(f"######## \n {images_path}")
            
            conv = [
                {

                    "role": "user",
                    "content": [
                        {"type": "text", 
                         "text": input_prompt},
                        # *([{"type": "image"}] * len(images_path))
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
            # Multi-image
            images = [Image.open(image_path).convert('RGB')
                 for image_path in images_path]
            # print(prompt)
            # print("images loaded")
            # print(conv)
            only_text = self.processor.tokenizer(prompt, return_tensors='pt')
            # print("only_text inputs ", only_text['input_ids'].shape)

            inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(device)
            input_ids_len = inputs['input_ids'].shape[1]
            print("processed inputs ", inputs['input_ids'].shape)

            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            output_ids = self.model.generate(
                **inputs,
                use_cache=True,
                **gen_kwargs
            )
                # print("model generate")
            # print("output")
            # print(output_ids.shape, output_ids.device)
            output_ids = output_ids[0, input_ids_len:]
            answer = self.processor.decode(output_ids, skip_special_tokens=True).strip()
            answers.append(answer)
            del output_ids, inputs, prompt, images
            gc.collect()
            torch.cuda.empty_cache()

        return answers

class LLaVA_oneVision_Sink(BaseWorker):

    def init_components(self, config):
        from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            config.model_dir, device_map='cuda', torch_dtype=torch.float16, use_flash_attention_2=True)
        self.processor = LlavaOnevisionProcessor.from_pretrained(config.model_dir, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        self.single_img_tokens = "<|im_start|>"+self.processor.image_token+"<|im_end|>"

        self.model.eval()

    def forward(self, questions, image_paths, device, gen_kwargs):

        answers = []
        print("Actual images being used: ",np.mean([len(i) for i in image_paths]))
        for question, images_path in zip(questions, image_paths):

            # NOTE: handle the special cases in CLEVR-Change dataset
            question = question.replace(
                '<ImageHere><ImageHere>', '<ImageHere>\n<ImageHere>\n')
            input_prompt = question.replace(
                '<ImageHere>', self.single_img_tokens)
            # print(f"######## \n {images_path}")
            
            conv = [
                {

                    "role": "user",
                    "content": [
                        {"type": "text", 
                         "text": input_prompt},
                        # *([{"type": "image"}] * len(images_path))
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
            # Multi-image
            images = [Image.open(image_path).convert('RGB')
                 for image_path in images_path]
            # print(prompt)
            # print("images loaded")
            # print(conv)
            only_text = self.processor.tokenizer(prompt, return_tensors='pt')
            # print("only_text inputs ", only_text['input_ids'].shape)

            inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(device)
            input_ids_len = inputs['input_ids'].shape[1]
            print("processed inputs ", inputs['input_ids'].shape)

            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                past_key_values = SinkCache(85072, 512)
                output_ids = self.model.generate(
                    **inputs,
                    use_cache=True,
                    past_key_values = past_key_values,
                    **gen_kwargs
                )
                # print("model generate")
            # print("output")
            # print(output_ids.shape, output_ids.device)
            output_ids = output_ids[0, input_ids_len:]
            answer = self.processor.decode(output_ids, skip_special_tokens=True).strip()
            answers.append(answer)
            del output_ids, inputs, prompt, images, past_key_values
            gc.collect()
            torch.cuda.empty_cache()

        return answers



class LLaVA_oneVision_cam(BaseWorker):

    def init_components(self, config):
        from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
        
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            config.model_dir, device_map='cuda', torch_dtype=torch.float16, use_flash_attention_2=True)
        
        lm_config = self.model.language_model.config
        lm_config.start_ratio = 0.01
        lm_config.recent_ratio = 0.1
        lm_config.merge_token = True
        self.model.language_model = convert_kvcache_qwen_cam(self.model.language_model, lm_config)
        
        self.processor = LlavaOnevisionProcessor.from_pretrained(config.model_dir, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        self.single_img_tokens = "<|im_start|>"+self.processor.image_token+"<|im_end|>"

        self.model.eval()
    
    def clean_mask(self, model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                model._modules[name] = self.clean_mask(module)

            if isinstance(module, Qwen2AttentionCam):
                module._reset_masks()
        return model

    def forward(self, questions, image_paths, device, gen_kwargs):

        answers = []
        print("Actual images being used: ",np.mean([len(i) for i in image_paths]))
        for question, images_path in zip(questions, image_paths):

            # NOTE: handle the special cases in CLEVR-Change dataset
            question = question.replace(
                '<ImageHere><ImageHere>', '<ImageHere>\n<ImageHere>\n')
            input_prompt = question.replace(
                '<ImageHere>', self.single_img_tokens)
            # print(f"######## \n {images_path}")
            
            conv = [
                {

                    "role": "user",
                    "content": [
                        {"type": "text", 
                         "text": input_prompt},
                        # *([{"type": "image"}] * len(images_path))
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(conv, add_generation_prompt=True)
            # Multi-image
            images = [Image.open(image_path).convert('RGB')
                 for image_path in images_path]
            # print(prompt)
            # print("images loaded")
            # print(conv)
            only_text = self.processor.tokenizer(prompt, return_tensors='pt')
            # print("only_text inputs ", only_text['input_ids'].shape)

            inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(device)
            input_ids_len = inputs['input_ids'].shape[1]
            print("processed inputs ", inputs['input_ids'].shape)

            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                past_key_values = OffloadedCache()
                output_ids = self.model.generate(
                    **inputs,
                    use_cache=True,
                    past_key_values = past_key_values,
                    **gen_kwargs
                )

                # print("model generate")
            # print("output")
            # print(output_ids.shape, output_ids.device)
            output_ids = output_ids[0, input_ids_len:]
            answer = self.processor.decode(output_ids, skip_special_tokens=True).strip()
            answers.append(answer)
            del output_ids, inputs, prompt, images
            gc.collect()
            torch.cuda.empty_cache()
            self.clean_mask(self.model)

        return answers


