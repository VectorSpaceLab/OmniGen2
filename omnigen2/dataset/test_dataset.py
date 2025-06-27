from typing import Optional

import os
import random
import json
import yaml

from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from transformers import AutoTokenizer

class OminiGenTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_path: str,
        tokenizer,
        apply_chat_template: bool,
        chat_template_version: str = 'v1',
        dynamic_image_size: bool = True,
        dynamic_image_size_version: str = 'v1',
        max_pixels: Optional[int] = None,
        img_scale_num: int = 16,
        align_res: bool = False
    ):
        
        self.dynamic_image_size = dynamic_image_size
        self.dynamic_image_size_version = dynamic_image_size_version
        self.max_pixels = max_pixels
        self.img_scale_num = img_scale_num
        # logger.info(f"read dataset config from {config_path}")
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        # logger.info("DATASET CONFIG:")
        # logger.info(self.config)

        self.apply_chat_template = apply_chat_template
        self.chat_template_version = chat_template_version
        self.align_res = align_res

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        ann, group_indice_range = self._collect_annotations(self.config)

        self.ann = ann
        self.tokenizer = tokenizer
        
    def _collect_annotations(self, config, global_ratio: float = 1, return_merged=True):
        group_ann = {}
        for data in config["data"]:
            data_path, data_type = data["path"], data.get("type", "default")
            data_ext = os.path.splitext(data_path)[-1]
            if data_ext == ".json":
                # with open(meta_path) as f:
                #     meta_l = json.load(f)
                with open(data_path, 'r') as json_file:
                    f = json_file.read()
                    data_l = json.loads(f) 
            elif data_ext == ".jsonl":
                data_l = []
                with open(data_path) as f:
                    for i, line in enumerate(f):
                        try:
                            data_l.append(json.loads(line))
                        except json.decoder.JSONDecodeError as e:
                            # logger.error(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}")
                            raise e
            elif data_ext in [".yml", ".yaml"]:
                with open(data_path, "r") as f:
                    sub_config = yaml.load(f, Loader=yaml.FullLoader)
                    sub_group_ann = self._collect_annotations(sub_config, global_ratio=data.get("ratio", 1), return_merged=False)
            else:
                raise NotImplementedError(
                    f'Unknown data file extension: "{data_ext}". '
                    f"Currently, .json, .jsonl are supported. "
                    "If you are using a supported format, please set the file extension so that the proper parsing "
                    "routine can be called."
                )
            
            if data_ext in [".yml", ".yaml"]:
                for data_type, data_l in sub_group_ann.items():
                    if data_type not in group_ann:
                        group_ann[data_type] = []
                    group_ann[data_type] += data_l
            else:
                # cur_ratio = data.get("ratio", 1)
                # random.seed(0)
                # data_l = random.sample(data_l, int(len(data_l) * cur_ratio * global_ratio))
                # logger.info(f"{data_path}, type: {data_type}, len: {len(data_l)}, ratio: {cur_ratio}, len * ratio * global_ratio: {len(data_l) * cur_ratio * global_ratio}")
                if "root" in data:
                    for item in data_l:
                        for path_key in ["path", "image_url", "image", "image_path"]:
                            if path_key in item:
                                item[path_key] = os.path.join(data["root"], item[path_key])
                if data_type not in group_ann:
                    group_ann[data_type] = []
                group_ann[data_type] += data_l

        if return_merged:
            ann = sum(list(group_ann.values()), start=[])

            group_indice_range = {}
            start_pos = 0
            for data_type, data_l in group_ann.items():
                group_indice_range[data_type] = [start_pos, start_pos + len(data_l)]
                start_pos = start_pos + len(data_l)

            return ann, group_indice_range
        else:
            return group_ann
    
    def process_instruction(self, instruction):
        if self.apply_chat_template:
            instruction = [{"role": "user", "content": instruction}]
            instruction = self.tokenizer.apply_chat_template(instruction, tokenize=False, add_generation_prompt=False)
            if self.chat_template_version == 'v1':
                if "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." in instruction:
                    instruction = instruction.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
                else:
                    instruction = instruction.replace("You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
            elif self.chat_template_version == 'v2':
                if "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." in instruction:
                    instruction = instruction.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
                else:
                    instruction = instruction.replace("You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
                instruction = instruction.replace("You are a helpful assistant that generates high-quality images based on user instructions.", "You are a helpful AI that generates images with superior degree of image-text alignment based on user prompts. The generated image focuses on the main subject, with a blurred background.")
                instruction = instruction.replace("Hyper-Realistic photo. ", "")
        return instruction

    def process_item(self, data_item):
        # print(f"{data_item=}")
        instruction = self.process_instruction(data_item['instruction'])

        input_images_path = data_item['input_images']
        input_images = []

        for input_image_path in input_images_path:
            input_image = Image.open(input_image_path).convert("RGB")
            ratio = 1
            if self.dynamic_image_size:
                width, height = input_image.size
                cur_pixels = height * width
                # if cur_pixels > self.max_pixels:
                ratio = (self.max_pixels / cur_pixels) ** 0.5
            if self.dynamic_image_size_version in ['v2', 'v3']:
                ratio = min(ratio, 1.0)
            new_height, new_width = int(height * ratio) // self.img_scale_num * self.img_scale_num, int(width * ratio) // self.img_scale_num * self.img_scale_num
            input_image = input_image.resize((new_width, new_height), resample=Image.BICUBIC)

            input_images.append(input_image)
        
        input_images = [self.image_transform(input_image) for input_image in input_images]

        if len(input_images) == 1 and self.align_res:
            target_img_size = (input_images[0].shape[-1], input_images[0].shape[-2])
        else:
            target_img_size = data_item["target_img_size"]

        # print(f"{[input_image.shape for input_image in input_images]=}")
        # print(f"{target_img_size=}", flush=True)
        w, h = target_img_size
        cur_pixels = w * h
        ratio = 1
        # if cur_pixels > self.max_pixels:
        ratio = (self.max_pixels / cur_pixels) ** 0.5
        if self.dynamic_image_size_version == 'v3':
            ratio = min(ratio, 1.0)

        target_img_size = (int(w * ratio) // self.img_scale_num * self.img_scale_num, int(h * ratio) // self.img_scale_num * self.img_scale_num)

        data = {
            'task_type': data_item['task_type'],
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'target_img_size': target_img_size,
        }
        return data

    def __getitem__(self, index):
        data_item = self.ann[index]
        return self.process_item(data_item)
        
    def __len__(self):
        return len(self.ann)

class OminiGenTestCollator():
    def __init__(self, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __call__(self, batch):
        task_type = [data['task_type'] for data in batch]
        instruction = [data['instruction'] for data in batch]
        input_images_path = [data['input_images_path'] for data in batch]
        input_images = [data['input_images'] for data in batch]

        text_inputs = self.tokenizer(
            instruction,
            padding="longest",
            max_length=self.max_token_len,
            truncation=True,
            return_tensors="pt",
        )

        data = {
            "task_type": task_type,
            "text_ids": text_inputs.input_ids,
            "text_mask": text_inputs.attention_mask,
            "input_images": input_images, 
            "input_images_path": input_images_path
        }
        return data

if __name__ == '__main__':
    config_path = "/share_2/luoxin/projects/Ominigenv2/data_options/test.yml"
    # with open(config_path, "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # print(config)

    print(np.random.choice(a=[1, 2], p=[0.5, 0.5]))

    tokenizer = AutoTokenizer.from_pretrained("/share_2/shitao/projects/DiffusionGPT2/qwen_cache/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")

    data = OminiGenTestDataset(config_path, apply_chat_template=True, dynamic_image_size=True)
    collator = OminiGenTestCollator(tokenizer=tokenizer, max_token_len=768)
    print('----------------------')
    print(data[0])
    # batch_data = collator([data[0], data[1], data[2]])
    # print(batch_data)
