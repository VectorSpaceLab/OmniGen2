from viescore import VIEScore
import PIL
import os

# import megfile
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import sys
import csv
import threading
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import accelerate
from accelerate import Accelerator
from accelerate.state import AcceleratorState

GROUPS = [
    "background_change",
    "color_alter",
    "material_alter",
    "motion_change",
    "ps_human",
    "style_change",
    "subject-add",
    "subject-remove",
    "subject-replace",
    "text_change",
    "tone_transfer",
]

def process_single_item(item, vie_score, args, turn, max_retries=10000):
    instruction = item["instruction"]
    key = item["key"]
    instruction_language = item["instruction_language"]
    save_path_fullset_source_image = f"{args.result_dir}/fullset/{group_name}/{instruction_language}/{key}_SRCIMG.png"
    save_path_fullset_result_image = (
        f"{args.result_dir}/fullset/{group_name}/{instruction_language}/{key}{'_sample' + str(turn) if turn > 0 else ''}.png"
    )

    src_image_path = save_path_fullset_source_image
    save_path_item = save_path_fullset_result_image

    for retry in range(max_retries):
        # try:
        pil_image_raw = Image.open(open(src_image_path, "rb")).convert("RGB")
        pil_image_edited = (
            Image.open(open(save_path_item, "rb"))
            .convert("RGB")
            .resize((pil_image_raw.size[0], pil_image_raw.size[1]))
        )

        text_prompt = instruction
        score_list = vie_score.evaluate(
            [pil_image_raw, pil_image_edited], text_prompt, echo_output=False
        )
        sementics_score, quality_score, overall_score = score_list

        return {
            "source_image": src_image_path,
            "edited_image": save_path_item,
            "instruction": instruction,
            "sementics_score": sementics_score,
            "quality_score": quality_score,
            "intersection_exist": item["Intersection_exist"],
            "instruction_language": item["instruction_language"],
        }

if __name__ == "__main__":
    accelerator = Accelerator()

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="/results/")
    parser.add_argument("--csv_dir", type=str, default="viescore_gpt-4.1")
    parser.add_argument(
        "--backbone", type=str, default="gpt-4.1", choices=["gpt-4.1", "gpt-5"]
    )
    parser.add_argument(
        "--openai_url", type=str, default="https://api.openai.com/v1/chat/completions"
    )
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument(
        "--key", type=str, required=True
    )
    parser.add_argument("--num_samples", type=int, default=1)

    args = parser.parse_args()

    backbone = args.backbone

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    vie_score = VIEScore(
        backbone=backbone, task="tie", key=args.key, openai_url=args.openai_url
    )
    max_workers = 20
    dataset = load_dataset("stepfun-ai/GEdit-Bench", split='train')
    dataset = dataset.remove_columns(["input_image", "input_image_raw"])
    dataset = dataset.filter(lambda x: x["instruction_language"] == "en", num_proc=4)

    data_index_list = list(
        range(
            AcceleratorState().process_index,
            len(dataset),
            AcceleratorState().num_processes,
        )
    )

    all_csv_list = defaultdict(list)  # Store all results for final combined CSV
    for group_name in GROUPS:
        for turn in range(args.num_samples):
            group_csv_list = []
            group_dataset_list = dataset.filter(
                lambda x: x["task_type"] == group_name, num_proc=4
            )

            # Load existing group CSV if it exists
            group_csv_path = os.path.join(
                args.result_dir, f"viescore_{args.backbone}", f"{group_name}_gpt_score{'_sample' + str(turn) if turn > 0 else ''}.csv"
            )

            processed_samples = set()

            if os.path.exists(group_csv_path):
                with open(group_csv_path, "r", newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    group_results = list(reader)
                    group_csv_list.extend(group_results)

                    for row in group_results:
                        sample_key = (row["source_image"], row["edited_image"])
                        processed_samples.add(sample_key)

                print(f"Loaded existing results for {group_name}")

            print(f"Processing group: {group_name}")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for item in group_dataset_list:
                    instruction = item["instruction"]
                    key = item["key"]
                    instruction_language = item["instruction_language"]
                    intersection_exist = item["Intersection_exist"]
                    sample_prefix = key
                    save_path_fullset_source_image = f"{args.result_dir}/fullset/{group_name}/{instruction_language}/{key}_SRCIMG.png"
                    save_path_fullset_result_image = f"{args.result_dir}/fullset/{group_name}/{instruction_language}/{key}{'_sample' + str(turn) if turn > 0 else ''}.png"

                    if not os.path.exists(
                        save_path_fullset_result_image
                    ) or not os.path.exists(save_path_fullset_source_image):
                        print(
                            f"Skipping {sample_prefix}: Source or edited image does not exist {save_path_fullset_result_image=}"
                        )
                        continue

                    # Check if this sample has already been processed
                    sample_key = (
                        save_path_fullset_source_image,
                        save_path_fullset_result_image,
                    )
                    exists = sample_key in processed_samples
                    if exists:
                        print(
                            f"Skipping already processed sample: {sample_prefix}",
                            flush=True,
                        )
                        continue

                    future = executor.submit(process_single_item, item, vie_score, args, turn)
                    futures.append(future)

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    unit="image",
                    desc=f"Processing {group_name} {turn}",
                ):
                    result = future.result()
                    if result:
                        group_csv_list.append(result)

            from accelerate.utils import gather_object

            group_csv_list = gather_object(group_csv_list)

            if accelerator.is_main_process:
                # Save group-specific CSV
                group_csv_path = os.path.join(
                    args.result_dir, f"viescore_{args.backbone}", f"{group_name}_gpt_score{'_sample' + str(turn) if turn > 0 else ''}.csv"
                )
                os.makedirs(os.path.dirname(group_csv_path), exist_ok=True)
                with open(group_csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    fieldnames = [
                        "source_image",
                        "edited_image",
                        "instruction",
                        "sementics_score",
                        "quality_score",
                        "intersection_exist",
                        "instruction_language",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in group_csv_list:
                        writer.writerow(row)
                all_csv_list[turn].extend(group_csv_list)

                print(f"Saved group CSV for {group_name}, length： {len(group_csv_list)}, file: {group_csv_path}")

            if accelerator.is_main_process and all_csv_list[turn]:
                # Save combined CSV
                combined_csv_path = os.path.join(
                    args.result_dir, f"viescore_{args.backbone}", f"combined_gpt_score{'_sample' + str(turn) if turn > 0 else ''}.csv"
                )
                os.makedirs(os.path.dirname(combined_csv_path), exist_ok=True)
                with open(combined_csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    fieldnames = [
                        "source_image",
                        "edited_image",
                        "instruction",
                        "sementics_score",
                        "quality_score",
                        "intersection_exist",
                        "instruction_language",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in all_csv_list[turn]:
                        writer.writerow(row)