import megfile
import os
import pandas as pd
from collections import defaultdict
import sys
import numpy as np
import math
import json
import glob

def analyze_scores(json_lines, language):    
    group_prompt_following_scores = {}
    group_subject_consistency_scores = {}
    group_overall_scores = {}

    for task_type in json_lines.keys():
        prompt_following_scores = []
        subject_consistency_scores = []
        overall_scores = []

        for json_line in json_lines[task_type]:
            if json_line['instruction_language'] != language:
                continue

            prompt_following_score = json_line['PF_score']
            subject_consistency_score = json_line['SC_score']
            overall_score = math.sqrt(prompt_following_score * subject_consistency_score)
            
            prompt_following_scores.append(prompt_following_score)
            subject_consistency_scores.append(subject_consistency_score)
            overall_scores.append(overall_score)

        group_prompt_following_scores[task_type] = np.mean(prompt_following_scores)
        group_subject_consistency_scores[task_type] = np.mean(subject_consistency_scores)
        group_overall_scores[task_type] = np.mean(overall_scores)
    
    return group_prompt_following_scores, group_subject_consistency_scores, group_overall_scores

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="/results/")
    parser.add_argument("--backbone", type=str, default="gpt4dot1")
    parser.add_argument("--model_name", type=str, default="OmniGen2")
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()

    result_json_files = glob.glob(os.path.join(args.save_path, args.model_name, args.backbone, "**/*.jsonl"))
    print(f"{result_json_files=}")
    print(f"{len(result_json_files)=}")

    result_json_lines = defaultdict(list)
    for result_json_file in result_json_files:
        with open(result_json_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                task_type = os.path.basename(os.path.dirname(result_json_file))
                result_json_lines[task_type].append(data)
    

    group_prompt_following_scores, group_subject_consistency_scores, group_overall_scores = analyze_scores(result_json_lines, language=args.language)

    for task_type in group_prompt_following_scores.keys():
        print(f"{task_type}: {group_prompt_following_scores[task_type]:.3f}, {group_subject_consistency_scores[task_type]:.3f}, {group_overall_scores[task_type]:.3f}")

    print(f"Average: {np.mean(list(group_prompt_following_scores.values())):.3f}, {np.mean(list(group_subject_consistency_scores.values())):.3f}, {np.mean(list(group_overall_scores.values())):.3f}")