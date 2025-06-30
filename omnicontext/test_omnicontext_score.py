import math
import datasets
import argparse

import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import glob
from collections import defaultdict

from PIL import Image

from omnicontext.omnicontext_score import OmniContextScore


def process_single_item(item, vie_score, max_retries=5):
    instruction = item['instruction']
    key = item['key']
    instruction_language = item['instruction_language']
    
    input_images = item['input_images']
    output_image = Image.open(item['output_image_path']).convert("RGB")

    ori_img_sizes = [input_image.size for input_image in input_images]

    new_img_sizes = []
    for ori_img_size in ori_img_sizes:
        if ori_img_size[0] * ori_img_size[1] > 1024 * 1024:
            ratio = math.sqrt(1024 * 1024 / (ori_img_size[0] * ori_img_size[1]))
            new_img_size = (int(ori_img_size[0] * ratio), int(ori_img_size[1] * ratio))
        else:
            new_img_size = ori_img_size
        new_img_size = (new_img_size[0] // 16 * 16, new_img_size[1] // 16 * 16)
        new_img_sizes.append(new_img_size)
    input_images = [input_image.resize(new_img_size) for input_image, new_img_size in zip(input_images, new_img_sizes)]
    
    result_dict = {
        'key': key,
        'task_type': item['task_type'],
        'instruction': instruction,
        'instruction_language': instruction_language,
        'output_image_path': item['output_image_path'],
    }

    if item['task_type'].find('scene') != -1:
        with_scene = True
    else:
        with_scene = False
    score_dict = vie_score.evaluate(input_images + [output_image], instruction, with_scene=with_scene)
    
    print(f"{score_dict=}", flush=True)

    result_dict['PF_score'] = score_dict['PF_scores']['score']
    result_dict['PF_score_reason'] = score_dict['PF_scores']['reasoning']
    result_dict['SC_score'] = score_dict['SC_scores']['score']
    result_dict['SC_score_reason'] = score_dict['SC_scores']['reasoning']

    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--openai_url", type=str, default="https://api.openai.com/v1/chat/completions")
    parser.add_argument("--openai_key", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=100)
    args = parser.parse_args()


    omnicontext_score = OmniContextScore(args.openai_url, args.openai_key)

    test_dataset = datasets.load_dataset(args.test_data, split="train")

    sub_datasets = defaultdict(list)
    for example in test_dataset:
        task_type = example['task_type']
        sub_datasets[task_type].append(example)

    all_result_list = []
    for task_type, sub_data in sub_datasets.items():
        result_list = []
        json_path = os.path.join(args.result_dir, args.model_name, "gpt4dot1", task_type, "score.jsonl")

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line)
                    result_list.append(result)
            print(f"Loaded {json_path} for {task_type}, length： {len(result_list)}")
            continue

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for item in sub_data:
                key = item["key"]
                output_image_path = os.path.join(args.result_dir, args.model_name, "fullset", task_type, f"{key}.png")

                item['output_image_path'] = output_image_path

                if not os.path.exists(output_image_path):
                    print(f"Output image not found: {output_image_path}, skip")
                    continue

                future = executor.submit(process_single_item, item, omnicontext_score)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), unit="image", desc=f"Processing {task_type}"):
                result = future.result()
                if result:
                    result_list.append(result)

        all_result_list.extend(result_list)
                
        # Save group-specific CSV
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            for result in result_list:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"Saved {json_path} for {task_type}, length： {len(result_list)}")

    combined_json_path = os.path.join(args.result_dir, args.model_name, "gpt4dot1", "combined_score.jsonl")

    os.makedirs(os.path.dirname(combined_json_path), exist_ok=True)
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        for result in all_result_list:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')