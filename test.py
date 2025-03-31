from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pandas as pd
from tqdm import tqdm

# Load MiniGPT checkpoint
MODEL_NAME = "openbmb/MiniGPT-4"  # Change to your local model if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")

# Load inference pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

# Load all MileBench splits
all_splits = [
    'ActionLocalization_test', 'ActionLocalization_adv', 'ActionPrediction_test', 'ActionPrediction_adv',
    'ActionSequence_test', 'ActionSequence_adv', 'ALFRED_test', 'ALFRED_adv', 'CharacterOrder_test',
    'CharacterOrder_adv', 'CLEVR_Change_test', 'CLEVR_Change_adv', 'CounterfactualInference_test',
    'CounterfactualInference_adv', 'DocVQA_test', 'DocVQA_adv', 'EgocentricNavigation_test',
    'EgocentricNavigation_adv', 'GPR1200_test', 'IEdit_test', 'IEdit_adv', 'ImageNeedleInAHaystack_test',
    'MMCoQA_test', 'MMCoQA_adv', 'MovingAttribute_test', 'MovingAttribute_adv', 'MovingDirection_test',
    'MovingDirection_adv', 'MultiModalQA_test', 'MultiModalQA_adv', 'nuscenes_test', 'nuscenes_adv',
    'ObjectExistence_test', 'ObjectExistence_adv', 'ObjectInteraction_test', 'ObjectInteraction_adv',
    'ObjectShuffle_test', 'ObjectShuffle_adv', 'OCR_VQA_test', 'OCR_VQA_adv', 'SceneTransition_test',
    'SceneTransition_adv', 'SlideVQA_test', 'SlideVQA_adv', 'Spot_the_Diff_test', 'Spot_the_Diff_adv',
    'StateChange_test', 'StateChange_adv', 'TextNeedleInAHaystack_test', 'TQA_test', 'TQA_adv',
    'WebQA_test', 'WebQA_adv', 'WikiVQA_test', 'WikiVQA_adv'
]

results = []

for split in all_splits:
    print(f"\n🔹 Processing split: {split}")
    try:
        dataset = load_dataset("FreedomIntelligence/MileBench", split=split)
    except Exception as e:
        print(f"❌ Failed to load {split}: {e}")
        continue

    # Take a small sample to test
    sample_data = dataset.select(range(min(5, len(dataset))))  # Adjust as needed

    for item in tqdm(sample_data, desc=f"🧪 Testing {split}"):
        prompt = item.get("question", "") or item.get("query", "") or str(item)
        if not prompt:
            continue

        try:
            generated = text_generator(prompt)
            response = generated[0]["generated_text"]
        except Exception as e:
            response = f"[Error] {e}"

        results.append({
            "split": split,
            "input": prompt,
            "output": response
        })

# Save results
df = pd.DataFrame(results)
df.to_csv("minigpt_milebench_results.csv", index=False)
print("✅ All done. Results saved to 'minigpt_milebench_results.csv'")
