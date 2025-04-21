from collections import defaultdict
import json # For loading the results if they are in a JSON string/file

def calculate_milebench_scores(results_dict):
    """
    Calculates aggregated MILEBENCH scores (like T1, S1, Overall) 
    from a detailed results dictionary, based on the mapping in Table 6 
    of the MILEBENCH paper.

    Args:
        results_dict (dict): A nested dictionary containing the detailed 
                             evaluation results for a single model, structured 
                             similarly to the provided example.

    Returns:
        dict: A dictionary containing the calculated scores for T1-T4, 
              S1-S5, N1, N2, I1, Realistic Overall, and Diagnostic Overall,
              formatted as percentages (0-100). Returns None if the input
              dict structure is unexpected.
    """

    # Mapping from the subtask names in the results dict to Table 2 categories
    # Based on Table 6 (Appendix B.1)
    subtask_to_category_map = {
        # T-1: Action Understanding and Prediction
        "ActionLocalization": "T1",
        "ActionPrediction": "T1",
        "ActionSequence": "T1",
        # T-2: Object and Scene Understanding
        "ObjectExistence": "T2",
        "ObjectInteraction": "T2",
        "MovingAttribute": "T2",
        "ObjectShuffle": "T2",
        # T-3: Visual Navigation and Spatial Localization
        "EgocentricNavigation": "T3",
        "MovingDirection": "T3",
        # T-4: Counterfactual Reasoning and State Change
        "CounterfactualInference": "T4",
        "StateChange": "T4",
        "CharacterOrder": "T4",
        "SceneTransition": "T4",
        # S-1: Knowledge Grounded QA
        "WebQA": "S1",
        "TQA": "S1",         # Textbook QA
        "MultiModalQA": "S1",# Complex Multimodal QA
        "WikiVQA": "S1",     # Long Text with Images QA
        # S-2: Text-Rich Images QA
        "SlideVQA": "S2",    # Slide QA
        "OCR-VQA": "S2",     # OCR QA
        "DocVQA": "S2",      # Document QA
        # S-3: Visual Relation Inference
        "Spot-the-Diff": "S3", # Visual Change Captioning (Spot-the-Diff)
        "CLEVR-Change": "S3",  # Visual Change Captioning (CLEVR-Change)
        "IEdit": "S3",         # Visual Relationship Expressing
        # S-4: Dialogue
        "MMCoQA": "S4",        # Multimodal Dialogue
        "ALFRED": "S4",        # Conversational Embodied Dialogue
        # S-5: Space Understanding
        "nuscenes": "S5",      # Space Understanding
        # N-1: Text Needle In A Haystack
        "TextNeedleInAHaystack": "N1",
        # N-2: Image Needle In A Haystack
        "ImageNeedleInAHaystack": "N2",
        # I-1: Image Retrieval
        "GPR1200": "I1",       # Image Retrieval
    }

    category_sums = defaultdict(float)
    category_counts = defaultdict(int)
    
    # Extract the actual model results (assuming one model per dict)
    model_name = list(results_dict.keys())[0]
    model_results = results_dict[model_name]

    # Iterate through the main categories (Realistic Temporal, Realistic Semantic, Diagnostic)
    for main_category, sub_results in model_results.items():
        if not isinstance(sub_results, dict):
            print(f"Warning: Expected dict for {main_category}, got {type(sub_results)}. Skipping.")
            continue
            
        # Iterate through specific subtasks within the main category
        for subtask_name, metrics in sub_results.items():
            if subtask_name not in subtask_to_category_map:
                # print(f"Warning: Subtask '{subtask_name}' not found in mapping. Skipping.")
                continue

            category_code = subtask_to_category_map[subtask_name]

            # Get the primary score (Accuracy or Rouge-L f) and convert to percentage
            score = 0.0
            found_score = False
            if "Accuracy" in metrics:
                score = metrics["Accuracy"] * 100
                found_score = True
            elif "Rouge-L f" in metrics:
                 # ROUGE-L is often reported 0-1, Table 2 looks like 0-100
                 # Adjust if the input ROUGE is already 0-100
                score = metrics["Rouge-L f"] * 100 
                found_score = True

            if found_score:
                category_sums[category_code] += score
                category_counts[category_code] += 1
            else:
                 print(f"Warning: No primary score ('Accuracy' or 'Rouge-L f') found for subtask '{subtask_name}'. Skipping.")


    # Calculate averages for each category
    final_scores = {}
    all_category_codes = list(set(subtask_to_category_map.values())) # T1, T2,... I1

    for code in all_category_codes:
        if category_counts[code] > 0:
            final_scores[code] = round(category_sums[code] / category_counts[code], 1)
        else:
            final_scores[code] = 0.0 # Or None, or handle as needed if a task is missing

    # Calculate Overall Scores
    realistic_tasks = ["T1", "T2", "T3", "T4", "S1", "S2", "S3", "S4", "S5"]
    diagnostic_tasks = ["N1", "N2", "I1"]

    realistic_sum = sum(final_scores.get(task, 0.0) for task in realistic_tasks)
    realistic_count = sum(1 for task in realistic_tasks if task in final_scores) # Count tasks actually calculated
    final_scores["Realistic Overall"] = round(realistic_sum / realistic_count, 1) if realistic_count > 0 else 0.0

    diagnostic_sum = sum(final_scores.get(task, 0.0) for task in diagnostic_tasks)
    diagnostic_count = sum(1 for task in diagnostic_tasks if task in final_scores) # Count tasks actually calculated
    final_scores["Diagnostic Overall"] = round(diagnostic_sum / diagnostic_count, 1) if diagnostic_count > 0 else 0.0

    # Reorder keys to roughly match Table 2 columns for better readability
    ordered_keys = [
        "T1", "T2", "T3", "T4", "S1", "S2", "S3", "S4", "S5", 
        "N1", "N2", "I1", "Realistic Overall", "Diagnostic Overall"
    ]
    
    ordered_final_scores = {key: final_scores.get(key, 0.0) for key in ordered_keys}


    return ordered_final_scores

# --- Example Usage ---

with open('outputs/result.json', 'r') as f:
    scores = json.load(f)

calculated_scores = calculate_milebench_scores(scores)

# Print the results
print("Calculated MILEBENCH Scores for llava-onevision-qwen2-7b-ov-hf:")
print(json.dumps(calculated_scores, indent=4))

# You can format this further to resemble the table row
print("\nFormatted like Table 2 row:")
row_values = [str(calculated_scores.get(k, 'N/A')) for k in calculated_scores.keys()]
print(" | ".join(row_values))