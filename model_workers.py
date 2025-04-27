# from workers.baseworker import *
# import sys

######################## Multi-image application ########################

# class LLaVA(BaseWorker):

#     def init_components(self, config):
#         sys.path.insert(0, '/path/to/LLaVA/packages/')
#         from llava.model.builder import load_pretrained_model
#         from llava.conversation import conv_templates, SeparatorStyle
#         from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

#         self.tokenizer, self.model, self.processor, context_len = load_pretrained_model(
#             model_path=config.model_dir,
#             model_base=None,
#             model_name=config.model_dir,
#             device_map='cuda',
#         )
        
#         if getattr(self.model.config, 'mm_use_im_start_end', False):
#             self.single_img_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
#         else:
#             self.single_img_tokens = DEFAULT_IMAGE_TOKEN

#         self.conv_temp = conv_templates["llava_llama_2"]
#         stop_str = self.conv_temp.sep if self.conv_temp.sep_style != SeparatorStyle.TWO else self.conv_temp.sep2
#         self.keywords = [stop_str]

#         self.model.eval()

#     def forward(self, questions, image_paths, device, gen_kwargs):
#         from llava.constants import IMAGE_TOKEN_INDEX
#         from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria

#         answers = []
#         for question,images_path in zip(questions, image_paths):
#             conv = self.conv_temp.copy()

#             # Multi-image
#             image_tensor = process_images(
#                 [Image.open(image_path).convert('RGB') for image_path in images_path],
#                 self.processor, self.model.config
#             ).to(device)

#             question = question.replace('<ImageHere><ImageHere>', '<ImageHere>\n<ImageHere>\n') # NOTE: handle the special cases in CLEVR-Change dataset
#             input_prompt = question.replace('<ImageHere>', self.single_img_tokens)

#             conv.append_message(conv.roles[0], input_prompt)
#             conv.append_message(conv.roles[1], None)
#             prompt = conv.get_prompt()
#             input_ids = tokenizer_image_token(
#                 prompt=prompt, 
#                 tokenizer=self.tokenizer, 
#                 image_token_index=IMAGE_TOKEN_INDEX, 
#                 return_tensors='pt'
#             ).unsqueeze(0).to(device)

#             with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#                 output_ids = self.model.generate(
#                     input_ids,
#                     images=image_tensor,
#                     use_cache=True,
#                     stopping_criteria=[KeywordsStoppingCriteria(self.keywords, self.tokenizer, input_ids)],
#                     **gen_kwargs
#                 )
#             answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
#             answers.append(answer)

#         return answers

from workers.baseworker import BaseWorker
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from modify_llama_cam2 import convert_kvcache_llama_cam

class LLaVA(BaseWorker):
    def init_components(self, config):
        model_id = config.model_dir  # e.g., 'llava-hf/llava-1.5-7b-hf'

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = self.processor.tokenizer  # for compatibility
        self.device = self.model.device
        self.model.eval()

    def forward(self, questions, image_paths, device, gen_kwargs):
        """
        :param questions: list of strings
        :param image_paths: list of list of image paths (multi-image per question)
        :param device: 'cuda' or 'cpu'
        :param gen_kwargs: generation settings
        """
        answers = []

        for question, paths in zip(questions, image_paths):
            # Prepare chat-like multimodal prompt
            parts = question.split("<ImageHere>")
            conversation_content = []

            for i, part in enumerate(parts):
                part = part.strip()
                if part:
                    conversation_content.append({"type": "text", "text": part})
                if i < len(paths):
                    conversation_content.append({"type": "image"})

            conversation = [{
                "role": "user",
                "content": conversation_content
            }]

            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            # Load and process all images
            images = [Image.open(p).convert("RGB") for p in paths]
            inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.device, torch.float16)
            # image_tensor = process_images(
            #     [Image.open(image_path).convert('RGB') for image_path in images_path],
            #     self.processor, self.model.config
            # ).to(device)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=gen_kwargs.max_new_tokens, do_sample=False)

            # Decode response (skip special/image tokens)
            decoded = self.processor.tokenizer.decode(outputs[0][2:], skip_special_tokens=True).strip()
            decoded = decoded.split("ASSISTANT:")[-1].strip()
            answers.append(decoded)

        return answers

class LLaVACAM(BaseWorker):
    def init_components(self, config):
        model_id = config.model_dir  # e.g., 'llava-hf/llava-1.5-7b-hf'

        # Load processor (handles both tokenizer and image processor)
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Load model
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Inject CAM-specific config values
        llama_config = self.model.config.text_config  # This is a LlamaConfig object

        setattr(llama_config, "start_ratio", 0.3)
        setattr(llama_config, "recent_ratio", 0.2)
        setattr(llama_config, "merge_token", True)

        # Apply CAM modification
        self.model.language_model = convert_kvcache_llama_cam(self.model.language_model, llama_config)[0]

        # Tokenizer (optional shortcut)
        self.tokenizer = self.processor.tokenizer
        self.device = self.model.device
        self.model.eval()

    def forward(self, questions, image_paths, device, gen_kwargs):
        """
        :param questions: list of strings
        :param image_paths: list of list of image paths (multi-image per question)
        :param device: 'cuda' or 'cpu'
        :param gen_kwargs: generation settings
        """
        answers = []

        for question, paths in zip(questions, image_paths):
            # Split question by <ImageHere> and build conversation
            parts = question.split("<ImageHere>")
            conversation_content = []

            for i, part in enumerate(parts):
                part = part.strip()
                if part:
                    conversation_content.append({"type": "text", "text": part})
                if i < len(paths):
                    conversation_content.append({"type": "image"})

            conversation = [{
                "role": "user",
                "content": conversation_content
            }]

            # Apply prompt template
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

            # Load and process images
            images = [Image.open(p).convert("RGB") for p in paths]
            inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.device, torch.float16)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs.max_new_tokens,
                    do_sample=False
                )

            # Decode and clean output
            decoded = self.processor.tokenizer.decode(outputs[0][2:], skip_special_tokens=True).strip()
            decoded = decoded.split("ASSISTANT:")[-1].strip()
            print(decoded)
            answers.append(decoded)

        return answers



class LLaVA_Sink(BaseWorker):

    def init_components(self, config):
        self.model = LlavaForConditionalGeneration.from_pretrained(
            config.model_dir,
            torch_dtype=torch.float16, device_map='cuda'
        )
        self.processor = AutoProcessor.from_pretrained(config.model_dir, use_fast=True)
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
                past_key_values = SinkCache(300, 100)
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
            torch.cuda.empty_cache()

        return answers
