from transformers import LlavaOnevisionForConditionalGeneration, AutoConfig, LlavaOnevisionProcessor
from PIL import Image
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "models/llava-onevision-qwen2-7b-ov-hf",
    # torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="cache_dir"
)


def descriptive_generation(model, processor, images, question, images_per_prompt=10):

    # chubnking images
    complete_description = ""
    num_images = len(images)
    image_chunks = []
    for i in range(0, num_images, images_per_prompt):
        chunk = images[i : i + images_per_prompt]
        image_chunks.append(chunk)
    
    # First description
    initial_prompt = f"""
Objective: Analyze the following images provided in this chunk, processed sequentially. Generate a structured description consisting of two parts:
1.  A Common Description summarizing the shared elements, setting, overall theme, or typical characteristics observed across most or all images in this chunk.
2.  Image-Specific Details for each image, listing only the concise details that are unique to that specific image or deviate significantly from the Common Description.

Instructions:

1.  Initial Scan: Briefly examine all images in this chunk to understand the overall content and identify recurring elements.
2.  Generate Common Description: Write a paragraph summarizing the elements that are shared or typical across the images in this chunk. This could include the main type of subject, the general setting, recurring actions, overall style, or mood. Label this section clearly as "Common Description (Chunk 1):".
3.  Generate Image-Specific Details: For each image provided (Image 1, Image 2, etc.), identify details that are not captured by the Common Description or represent significant variations. These should be concise bullet points or short descriptions. Focus on:
       Unique objects, people, or text present only in that image.
       Specific actions or states that differ from the norm described in the Common Description.
       Notable differences in composition, lighting, or viewpoint.
       Key sequential changes if the images depict a process or event unfolding.
       Label this section "Image-Specific Details (Chunk 1):" and preface each image's details with its sequence number (e.g., "Image 1 Specifics:", "Image 2 Specifics:").

4.  Output Format: Present the output clearly structured with the "Common Description (Chunk 1):" paragraph first, followed by the "Image-Specific Details (Chunk 1):" section containing the labeled specifics for each image. Ensure descriptions are accurate and objective.

There are {len(image_chunks[0])} images in current chunk.
Generate the structured description now."""


    conv = [
        {
            "role": "user",
            "content": [
                {"type": "text", 
                 "text": initial_prompt}
            ],
        },
    ]
    conv[0]['content'].extend(([{"type": "image"}] *len(image_chunks[0])))

    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    # prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    # prompt = initial_prompt
    inputs = processor(images=image_chunks[0], text=prompt, return_tensors='pt').to('cuda')
    input_len = inputs.input_ids.shape[-1]
    output = model.generate(**inputs, max_new_tokens=1000, use_cache=True, do_sample=True, temperature = 1.0, eos_token_id=processor.tokenizer.eos_token_id)
    description = processor.decode(output[0][input_len:], skip_special_tokens=True)
    complete_description = description

    for i in range(1, len(image_chunks)):
        subsequent_prompt = f"""
Context: You are analyzing a sequence of images in chunks. The analysis from previous chunks, structured into Common and Specific descriptions, is provided below.

Objective: Analyze the new set of images provided for the current chunk, processed sequentially. Identify new commonalities and new specific details within this chunk that were not present or significantly different from the entire "Previous Description". Then, append a new, similarly structured analysis (Common Description + Image-Specific Details) for this current chunk to the "Previous Description".

Instructions:

1.  Analyze New Images: Examine the new images provided in this chunk sequentially, keeping the "Previous Description" in mind to identify novelty.
2.  Generate Common Description (New Chunk): Identify shared elements, themes, or characteristics present across most or all images in this new chunk. Summarize these in a paragraph, focusing only on aspects that are new or represent a significant change/evolution compared to any information in the "Previous Description". Label this clearly (e.g., "Common Description (Chunk [Current Chunk Number]):"). If there are no significant new common elements in this chunk, state that briefly.
3.  Generate Image-Specific Details (New Chunk): For each new image in this chunk (use absolute numbering if possible, e.g., "Image 4 Specifics:", "Image 5 Specifics:", otherwise use relative like "New Chunk - Image 1 Specifics:"), list only the concise details that are:
       Unique to this image within the new chunk.
       Significantly different from the "Common Description (New Chunk)" generated in step 2.
       Represent new information or significant changes not present anywhere in the "Previous Description".
       Keep these details concise.
       Label this section clearly (e.g., "Image-Specific Details (Chunk [Current Chunk Number]):") and preface each image's details appropriately.

4.  Avoid Redundancy: Crucially, do not repeat information already captured in the "Previous Description" in either the new Common or Specific sections, unless describing a direct modification or evolution of a previously mentioned element.
5.  Output Format: Generate the updated comprehensive description by:
       First, reproducing the entire "Previous Description" exactly as provided.
       Second, clearly marking the start of the new analysis (e.g., "--- Analysis of Chunk [Current Chunk Number] ---").
       Third, appending the "Common Description (Chunk [Current Chunk Number]):" paragraph.
       Fourth, appending the "Image-Specific Details (Chunk [Current Chunk Number]):" section with its labeled specifics.
       The final output must be a single text block containing the previous description followed by the structured analysis of the new chunk.

Previous Description:
{description}

There are {len(image_chunks[i])} images in current chunk.
Generate the updated description by appending the structured analysis of the new sequential images now, focusing only on new information."""
        
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "text", 
                    "text": subsequent_prompt}
                ],
            },
        ]

        conv[0]['content'].extend(([{"type": "image"}] *len(image_chunks[i])))

        # prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = processor(images=image_chunks[i], text=prompt, return_tensors='pt').to('cuda')
        input_len = inputs.input_ids.shape[-1]
        output = model.generate(**inputs, max_new_tokens=1000, use_cache=True, do_sample=True, temperature = 1.0, eos_token_id=processor.tokenizer.eos_token_id)
        description = processor.decode(output[0][input_len:], skip_special_tokens=True)
        complete_description  = f"{complete_description} \n {description}"
        print(i)

    conv = [
            {
                "role": "user",
                "content": [
                    {"type": "text", 
                    "text": f"""Answer the following question based on the description of the images provided:
Description:
{complete_description}
Question:
{question}"""
                    }
                ],
            },
        ]
    
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors='pt').to('cuda')
    input_len = inputs.input_ids.shape[-1]
    output = model.generate(**inputs, max_new_tokens=512, use_cache=True, do_sample=True, temperature = 1.0, eos_token_id=processor.tokenizer.eos_token_id)
    answer = processor.decode(output[0][input_len:], skip_special_tokens=True)

    return answer
        

if __name__ == '__main__':
    processor = LlavaOnevisionProcessor.from_pretrained("models/llava-onevision-qwen2-7b-ov-hf", use_fast=True)
    # image_max_count = 2
    image_dir = "datasets/MileBench/ActionLocalization/images/0KZYF"
    images = [Image.open(os.path.join(image_dir,image_path)).convert('RGB')
                    for image_path in os.listdir(image_dir)] #[:image_max_count]

    with torch.no_grad():
        model.eval()
        output = descriptive_generation(model, processor, images, question="Step by step Describe what is happening in these images?")
        print('here')