from typing import Optional

import os
import random
import yaml
import glob
from PIL import Image

import torch

from datasets import load_dataset, concatenate_datasets

from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor

class OmniGen2TestDataset(torch.utils.data.Dataset):
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."

    def __init__(
        self,
        config_path: str,
        tokenizer,
        use_chat_template: bool,
        max_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        img_scale_num: int = 16,
        align_res: bool = True
    ):
        
        self.max_pixels = max_pixels
        self.max_side_length = max_side_length
        self.img_scale_num = img_scale_num
        self.align_res = align_res

        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.use_chat_template = use_chat_template
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)

        data = self._collect_annotations(self.config)

        self.data = data
        self.tokenizer = tokenizer
        
    def _collect_annotations(self, config):
        json_datasets = []
        for data in config['data']:
            data_path, data_type = data['path'], data.get("type", "default")
            if os.path.isdir(data_path):
                jsonl_files = list(glob.glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)) + list(glob.glob(os.path.join(data_path, "**/*.json"), recursive=True))
                json_dataset = load_dataset('json', data_files=jsonl_files, cache_dir=None)['train']
            else:
                data_ext = os.path.splitext(data_path)[-1]
                if data_ext in [".json", ".jsonl"]:
                    json_dataset = load_dataset('json', data_files=data_path, cache_dir=None)['train']
                elif data_ext in [".yml", ".yaml"]:
                    with open(data_path, "r") as f:
                        sub_config = yaml.load(f, Loader=yaml.FullLoader)
                        json_dataset = self._collect_annotations(sub_config)
                else:
                    raise NotImplementedError(
                        f'Unknown data file extension: "{data_ext}". '
                        f"Currently, .json, .jsonl .yml .yaml are supported. "
                        "If you are using a supported format, please set the file extension so that the proper parsing "
                        "routine can be called."
                    )
            json_datasets.append(json_dataset)
        
        json_dataset = concatenate_datasets(json_datasets)
        return json_dataset
    
    def apply_chat_template(self, instruction, system_prompt):
        if self.use_chat_template:
            prompt = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": instruction},
            ]
            instruction = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return instruction
    
    def process_item(self, data_item):
        assert data_item['instruction'] is not None
        if 'input_images' in data_item and data_item['input_images'] is not None:
            input_images_path = data_item['input_images']
            input_images = []

            for input_image_path in input_images_path:
                input_image = Image.open(input_image_path).convert("RGB")
                input_images.append(input_image)
        else:
            input_images_path, input_images = None, None

        if input_images is not None and len(input_images) == 1 and self.align_res:
            target_img_size = (input_images[0].width, input_images[0].height)
        else:
            target_img_size = data_item["target_img_size"]

        w, h = target_img_size
        cur_pixels = w * h
        ratio = min(1, (self.max_pixels / cur_pixels) ** 0.5)

        target_img_size = (int(w * ratio) // self.img_scale_num * self.img_scale_num, int(h * ratio) // self.img_scale_num * self.img_scale_num)

        data = {
            'task_type': data_item['task_type'],
            'instruction': data_item['instruction'],
            'input_images_path': input_images_path,
            'input_images': input_images,
            'target_img_size': target_img_size,
        }
        return data

    def __getitem__(self, index):
        data_item = self.data[index]
        return self.process_item(data_item)
        
    def __len__(self):
        return len(self.data)

class OmniGen2Collator():
    def __init__(self, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __call__(self, batch):
        task_type = [data['task_type'] for data in batch]
        instruction = [data['instruction'] for data in batch]
        input_images_path = [data['input_images_path'] for data in batch]
        input_images = [data['input_images'] for data in batch]
        output_image = [data['output_image'] for data in batch]
        output_image_path = [data['output_image_path'] for data in batch]

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
            "input_images_path": input_images_path,
            "output_image": output_image,
            "output_image_path": output_image_path,
        }
        return data
