import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO

# Load model
model_name = "THUDM/minigpt-4"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
model.eval()

# Sample image
image_url = "https://raw.githubusercontent.com/Vision-CAIR/minigpt-4/master/minigpt4/data/test.jpg"
image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0).cuda()

# Simulated long context D1-Dn
context_docs = [
    "This is a lively street with people celebrating a cultural event.",
    "Flags and decorations are visible in the image.",
    "Some individuals appear to be performing rituals.",
    "The weather looks bright and sunny.",
    "A gathering is taking place under colorful canopies."
]

# Summarization of each D_i → S_i
summaries = []
for i, doc in enumerate(context_docs):
    prompt = f"{doc} Summarize this:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=50)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    summaries.append(summary)

# Combine summaries and query
summary_text = " ".join(summaries)
query = "What is happening in this image?"
final_prompt = f"{summary_text} {query}"
inputs = tokenizer(final_prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)
answer = tokenizer.decode(output[0], skip_special_tokens=True)

# Display results
print("\nSummaries:")
for idx, s in enumerate(summaries):
    print(f"S{idx+1}: {s}")
print("\nFinal Answer to the Question:")
print(answer)
