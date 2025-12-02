import dotenv

dotenv.load_dotenv(override=True)

from editscore import EditScore
import PIL
import os
import glob
import json

from PIL import Image
import argparse
from datasets import load_dataset
from shutil import copyfile

import hashlib  # 新增导入
import threading # 新增导入
from tqdm import tqdm

def generate_cache_key(pair):
    """为每个样本生成一个唯一的SHA256哈希键。"""
    instruction, input_image, output_image = pair
    # 将三个组件用特殊分隔符连接，确保不会因内容本身包含分隔符而混淆
    key_string = f"{instruction}|||{input_image}|||{output_image}"
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

def load_cache(cache_file):
    """从JSONL文件加载缓存到字典。"""
    cache = {}
    if not os.path.exists(cache_file):
        return cache
    with open(cache_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                cache[data['key']] = data['result']
            except json.JSONDecodeError:
                print(f"Warning: Skipping corrupted line in cache file: {line.strip()}")
    return cache

def append_to_cache(cache_file, key, result, lock):
    """将新的计算结果线程安全地追加到缓存文件。"""
    with lock:
        with open(cache_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'key': key, 'result': result}, ensure_ascii=False) + '\n')

def process_single_item(item, vie_score, args, max_retries=10000):
    instruction = item[0]
    input_image = item[1]
    output_image = item[2]
    
    for retry in range(max_retries):
        # try:
        pil_image_raw = Image.open(input_image).convert("RGB")

        pil_image_edited = Image.open(output_image).convert("RGB")
        pil_image_edited = pil_image_edited.resize((pil_image_raw.size[0], pil_image_raw.size[1]))

        text_prompt = instruction
        score = vie_score.evaluate(
            [pil_image_raw, pil_image_edited], text_prompt, echo_output=False
        )
        return item, score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--scorer", type=str, default="viescore", choices=["viescore", "editscore"])
    parser.add_argument(
        "--backbone",
        type=str,
        default="openai",
        choices=["openai", "qwen25vl", "qwen25vl_vllm", "internvl3_5"],
    )
    parser.add_argument("--model_name_or_path", type=str, default="gpt-4.1")
    parser.add_argument(
        "--openai_url", type=str, default="https://api.openai.com/v1/chat/completions"
    )
    parser.add_argument(
        "--key", type=str, default="sk-cB6h7HcCSDIp71gs6lFLZxKE0dOYOnJbxzES6kWXe1Wb2VHS"
    )
    parser.add_argument(
        "--context_version", type=str, default="v1", choices=["v1", "v2"]
    )
    parser.add_argument(
        "--prompt_version", type=str, default="default", choices=["default", "our", "editscore"]
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
    )
    parser.add_argument(    
        "--end_index",
        type=int,
        default=1212,
    )
    parser.add_argument(
        "--num_samples", type=int, default=1
    )
    parser.add_argument("--num_pass", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--score_range", type=int, default=10)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=1536)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--max_num_batched_tokens", type=int, default=1536)
    parser.add_argument("--enable_lora", action="store_true")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()

def main(args):
    scorer = EditScore(
        backbone=args.backbone,
        key=args.key,
        openai_url=args.openai_url,
        model_name_or_path=args.model_name_or_path,
        score_range=args.score_range,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_pass=args.num_pass,
        enable_lora=args.enable_lora,
        lora_path=args.lora_path,
        cache_dir=args.cache_dir,
    )

    dataset = load_dataset("stepfun-ai/GEdit-Bench", split='train')
    dataset = dataset.remove_columns(["input_image", "input_image_raw"])
    dataset = dataset.filter(lambda x: x["instruction_language"] == "en", num_proc=4)

    data_index_list = list(range(args.start_index, args.end_index))
    with tqdm(
            total=len(data_index_list),
            desc=f"Processing {len(data_index_list)}/{len(dataset)}",
            unit="image"
        ) as pbar:
        for idx in data_index_list:
            data_item = dataset[idx]

            task_type = data_item['task_type']
            instruction_language = data_item['instruction_language']

            key = data_item['key']
            instruction = data_item['instruction']
            input_image_path = f"{args.result_dir}/fullset/{task_type}/{instruction_language}/{key}_SRCIMG.png"
            input_image = Image.open(input_image_path).convert('RGB')

            best_score = -float('inf')
            best_output_image_path = None

            for turn in range(args.num_samples):
                output_image_path = f"{args.result_dir}/fullset/{task_type}/{instruction_language}/{key}{'_sample' + str(turn) if turn > 0 else ''}.png"
                output_image = Image.open(output_image_path).convert('RGB')

                score = scorer.evaluate(
                    [input_image, output_image], instruction, echo_output=False
                )['overall']

                if score > best_score:
                    best_score = score
                    best_output_image_path = output_image_path
                
                if (turn + 1) & (turn + 1 - 1) == 0:
                    save_input_image_path = f"{args.save_dir}_best{turn + 1}/fullset/{task_type}/{instruction_language}/{key}_SRCIMG.png"
                    save_output_image_path = f"{args.save_dir}_best{turn + 1}/fullset/{task_type}/{instruction_language}/{key}.png"

                    os.makedirs(os.path.dirname(save_input_image_path), exist_ok=True)
                    if not os.path.exists(save_input_image_path):
                        copyfile(input_image_path, save_input_image_path)
                    copyfile(best_output_image_path, save_output_image_path)
            pbar.update(1)

if __name__ == "__main__":
    args = parse_args()
    main(args)