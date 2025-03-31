from datasets import load_dataset
import pandas as pd

# List of all MileBench splits
splits = [
    'ActionLocalization_test', 'ActionLocalization_adv',
    'ActionPrediction_test', 'ActionPrediction_adv',
    'ActionSequence_test', 'ActionSequence_adv',
    'ALFRED_test', 'ALFRED_adv',
    'CharacterOrder_test', 'CharacterOrder_adv',
    'CLEVR_Change_test', 'CLEVR_Change_adv',
    'CounterfactualInference_test', 'CounterfactualInference_adv',
    'DocVQA_test', 'DocVQA_adv',
    'EgocentricNavigation_test', 'EgocentricNavigation_adv',
    'GPR1200_test',
    'IEdit_test', 'IEdit_adv',
    'ImageNeedleInAHaystack_test',
    'MMCoQA_test', 'MMCoQA_adv',
    'MovingAttribute_test', 'MovingAttribute_adv',
    'MovingDirection_test', 'MovingDirection_adv',
    'MultiModalQA_test', 'MultiModalQA_adv',
    'nuscenes_test', 'nuscenes_adv',
    'ObjectExistence_test', 'ObjectExistence_adv',
    'ObjectInteraction_test', 'ObjectInteraction_adv',
    'ObjectShuffle_test', 'ObjectShuffle_adv',
    'OCR_VQA_test', 'OCR_VQA_adv',
    'SceneTransition_test', 'SceneTransition_adv',
    'SlideVQA_test', 'SlideVQA_adv',
    'Spot_the_Diff_test', 'Spot_the_Diff_adv',
    'StateChange_test', 'StateChange_adv',
    'TextNeedleInAHaystack_test',
    'TQA_test', 'TQA_adv',
    'WebQA_test', 'WebQA_adv',
    'WikiVQA_test', 'WikiVQA_adv'
]

# Store sample data
split_samples = {}

for split in splits:
    print(f"Loading {split}...")
    try:
        dataset = load_dataset("FreedomIntelligence/MileBench", split=split)
        # Take first 2 examples for preview
        sample = dataset.select(range(min(2, len(dataset))))
        split_samples[split] = sample
        print(f"Loaded {split} with {len(dataset)} samples.")
    except Exception as e:
        print(f"Failed to load {split}: {e}")

# Combine all split samples into one DataFrame
combined_df = pd.concat(
    [sample.to_pandas().assign(split_name=split) for split, sample in split_samples.items()],
    ignore_index=True
)

# Save to CSV for inspection or logging
combined_df.to_csv("milebench_sample_preview.csv", index=False)
print("\n✅ Preview saved to 'milebench_sample_preview.csv'")
