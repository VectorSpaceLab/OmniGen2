import json
import os
import argparse

DESIRED_TASKS = [
    "background",
    "color_alter",
    "material_alter",
    "motion_change",
    # "ps_human",
    "style",
    "subject_add",
    "subject_remove",
    "subject_replace",
    "tone_transfer",
    # "text_change"
]


def main(args):
    filtered_json_lines = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)

            if json_line["task_type"] in DESIRED_TASKS:
                filtered_json_lines.append(json_line)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for json_line in filtered_json_lines:
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)