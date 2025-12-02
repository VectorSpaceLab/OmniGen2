from typing import Optional, Union, List

import os
import random
import time
import math
import re
import yaml
import glob
from PIL import Image

import torch
from torchvision import transforms

from datasets import load_dataset, concatenate_datasets

from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor

from accelerate.logging import get_logger
logger = get_logger(__name__)


def normalize_whitespace(input_string):
    # 替换连续的空格为一个空格，并处理与制表符或换行符相邻的情况
    input_string = re.sub(r' +\t+', '\n', input_string)
    input_string = re.sub(r' +\n+', '\n', input_string)
    input_string = re.sub(r'\t+ +', '\n', input_string)
    input_string = re.sub(r'\n+ +', '\n', input_string)

    # 替换连续的空格为一个空格
    input_string = re.sub(r' +', ' ', input_string)
    
    # 替换连续的制表符为一个制表符
    input_string = re.sub(r'\t+', '\t', input_string)
    
    # 替换连续的换行符为一个换行符
    input_string = re.sub(r'\n+', '\n', input_string)
    
    return input_string.strip()


class OmniGen2TrainDataset(torch.utils.data.Dataset):
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."
    SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."

    def __init__(
        self,
        config_path: str,
        tokenizer,
        num_workers: int,
        use_chat_template: bool,
        max_input_pixels: Optional[Union[int, List[int]]] = None,
        max_output_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        img_scale_num: int = 16,
        prompt_dropout_prob: float = 0.0,
        ref_img_dropout_prob: float = 0.0,
    ):
        self.max_input_pixels = max_input_pixels
        self.max_output_pixels = max_output_pixels

        self.max_side_length = max_side_length
        self.img_scale_num = img_scale_num
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob

        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.num_workers = num_workers

        self.use_chat_template = use_chat_template
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)

        data = self._collect_annotations(self.config)

        self.data = data
        self.tokenizer = tokenizer
        
    def _collect_annotations(self, config):
        total_samples = 0
        total_ratio = 0
        json_datasets = []

        ratio_type = config.get('ratio_type', 'outside_ratio')

        for data in config['data']:
            data_path, data_type = data['path'], data.get("type", "default")
            if os.path.isdir(data_path):
                jsonl_files = list(glob.glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)) + list(glob.glob(os.path.join(data_path, "**/*.json"), recursive=True))
                json_dataset = load_dataset('json', data_files=jsonl_files, cache_dir=None, num_proc=self.num_workers)['train']
                logger.info(f"Loaded {len(json_dataset)} samples from {data_path}", main_process_only=False)
            else:
                data_ext = os.path.splitext(data_path)[-1]
                if data_ext in [".json", ".jsonl"]:
                    json_dataset = load_dataset('json', data_files=data_path, cache_dir=None, num_proc=self.num_workers)['train']
                    logger.info(f"Loaded {len(json_dataset)} samples from {data_path}", main_process_only=False)
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
            total_ratio += data['ratio']
            total_samples += len(json_dataset)
            json_datasets.append(json_dataset)
        
        start_time = time.time()
        for data, json_dataset in zip(config['data'], json_datasets):
            if ratio_type == 'inside_ratio':
                ratio = data['ratio']
            else:
                ratio = data['ratio'] / total_ratio
            if ratio != 1:
                target_size = int(len(json_dataset) * ratio) # normalize the ratio
                if target_size <= len(json_dataset):
                    # Random selection without replacement
                    indices = random.sample(range(len(json_dataset)), target_size)
                else:
                    # Oversample with replacement
                    indices = random.choices(range(len(json_dataset)), k=target_size)
                json_dataset = json_dataset.select(indices)
        logger.info(f"Time taken to select samples: {time.time() - start_time} seconds", main_process_only=False)

        start_time = time.time()
        json_dataset = concatenate_datasets(json_datasets)
        logger.info(f"Time taken to concatenate datasets: {time.time() - start_time} seconds")
        return json_dataset
    
    def clean_data_item(self, data_item):
        data_item['task_type'] = data_item['task_type'] if 'task_type' in data_item and data_item['task_type'] else ""

        task_type = data_item['task_type']
        instruction = data_item['instruction']

        if 'instruction_long' in data_item and data_item['instruction_long'] is not None:
            if random.random() < 0.8 and len(data_item['instruction_long']) > 10:
                instruction = data_item['instruction_long']
        
        if 'instruction_short' in data_item and data_item['instruction_short'] is not None:
            if random.random() < 0.3 and len(data_item['instruction_short']) > 10:
                instruction = data_item['instruction_short']

        if any([t2i_task_type in task_type for t2i_task_type in ['text_to_image', 't2i']]):
            if all(key in data_item and data_item[key] is not None for key in ['instruction', 'instruction_zh', 'instruction_short', 'instruction_short_zh']):
                instruction = random.choice(
                    [
                        data_item[key]
                        for key in [
                            "instruction",
                            "instruction_zh",
                            "instruction_short",
                            "instruction_short_zh",
                        ]
                        if len(data_item[key]) > 10
                    ]
                )
            elif 'instruction_zh' in data_item:
                if "instruction_tag" in data_item and data_item['instruction_tag'] is not None and len(data_item['instruction_tag']) > 3 and random.random() < 0.08:
                    instruction = data_item['instruction_tag']
                elif random.random() < 0.5 and data_item['instruction_zh'] is not None and len(data_item['instruction_zh']) > 10:
                    instruction = data_item['instruction_zh']
                else:
                    instruction = data_item['instruction']
        
            if "caption_qwenvl2_5" in data_item and data_item['caption_qwenvl2_5'] is not None:
                randn_num = random.random()
                if len(data_item["caption_qwenvl2_5"]) < 1000 and randn_num < 0.4 and len(data_item['caption_qwenvl2_5']) > 10:
                    instruction = data_item['caption_qwenvl2_5']
                elif len(data_item['caption_qwenvl2_5_zh']) < 1000 and randn_num < 0.99 and len(data_item['caption_qwenvl2_5_zh']) > 10:
                    instruction = data_item['caption_qwenvl2_5_zh']
                else:
                    instruction = data_item['instruction']
                    
            if "Hyper-Realistic photo. Photo of " in instruction:
                instruction = instruction.replace("Hyper-Realistic photo. Photo of ", "")
            if "Hyper-Realistic photo. " in instruction:
                instruction = instruction.replace("Hyper-Realistic photo. ", "")

        prefixs = ["The image portrays ", "The image depicts ", "The image captures ", "The image highlights ", "The image shows ", "这张图片展示了"]
        if random.random() < 0.5:
            for p in prefixs:
                if p in data_item['instruction']:
                    data_item['instruction'] = data_item['instruction'].replace(p, "")
                    break

        if "Hyper-Realistic photo. Photo of " in data_item['instruction']:
            data_item['instruction'] = data_item['instruction'].replace("Hyper-Realistic photo. Photo of ", "")
        if "Hyper-Realistic photo. " in data_item['instruction']:
            data_item['instruction'] = data_item['instruction'].replace("Hyper-Realistic photo. ", "")

        unnecessary_words = ["<img>", "</img>", "<|image_1|>", "<|image_2|>", "<|image_3|>", "<|image_4|>", "<|image_5|>",
                             "<|img_1|>", "<|img_2|>", "<|img_3|>", "<|img_4|>", "<|img_5|>"]
        for word in unnecessary_words:
            data_item['instruction'] = data_item['instruction'].replace(word, '')
        
        instruction = normalize_whitespace(instruction)

        return data_item
    
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
        data_item = self.clean_data_item(data_item)

        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction = self.apply_chat_template(data_item['instruction'], self.SYSTEM_PROMPT)

        if not drop_ref_img and 'input_images' in data_item and data_item['input_images'] is not None:
            input_images_path = data_item['input_images']
            input_images = []
            input_images_pil = []

            max_input_pixels = self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels

            for input_image_path in input_images_path:
                input_image_pil = Image.open(input_image_path).convert("RGB")
                input_image = self.image_processor.preprocess(input_image_pil, max_pixels=max_input_pixels, max_side_length=self.max_side_length)
                input_image_pil_resized = self.image_processor.postprocess(input_image, output_type="pil")[0]
                input_images.append(input_image)
                input_images_pil.append(input_image_pil_resized)
        else:
            input_images_path, input_images, input_images_pil = None, None, None

        target_img_size = data_item.get('target_img_size', (512, 512))

        if input_images_pil is not None and len(input_images_pil) == 1:
            target_img_size = (input_images_pil[0].width, input_images_pil[0].height)
        
        w, h = target_img_size
        cur_pixels = w * h
        ratio = min(1, (self.max_output_pixels / cur_pixels) ** 0.5)

        target_img_size = (int(w * ratio) // self.img_scale_num * self.img_scale_num, int(h * ratio) // self.img_scale_num * self.img_scale_num)

        # output_image_path = data_item['output_image']
        # output_image = Image.open(output_image_path).convert("RGB")
        # output_image = self.image_processor.preprocess(output_image, max_pixels=self.max_output_pixels, max_side_length=self.max_side_length)

        data = {
            'task_type': data_item['task_type'],
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'input_images_pil': input_images_pil,
            'target_img_size': target_img_size,
            'meta_data': data_item['meta_data'],
            # 'output_image': output_image,
            # 'output_image_path': output_image_path,
        }
        return data

    def __getitem__(self, index):
        max_retries = 100

        current_index = index
        for attempt in range(max_retries):
            try:
                data_item = self.data[current_index]
                return self.process_item(data_item)
            except Exception as e:
                print("error when loading data: ", e)
                if attempt == max_retries - 1:
                    raise e
                else:
                    # Try a different index for the next attempt
                    current_index = random.randint(0, len(self.data) - 1)
                    continue
        
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
        input_images_pil = [data['input_images_pil'] for data in batch]
        target_img_size = [data['target_img_size'] for data in batch]
        meta_data = [data['meta_data'] for data in batch]
        # output_image = [data['output_image'] for data in batch]
        # output_image_path = [data['output_image_path'] for data in batch]

        text_inputs = self.tokenizer(
            instruction,
            padding="longest",
            max_length=self.max_token_len,
            truncation=True,
            return_tensors="pt",
        )

        data = {
            "task_type": task_type,
            "instruction": instruction,
            "text_ids": text_inputs.input_ids,
            "text_mask": text_inputs.attention_mask,
            "input_images": input_images, 
            "input_images_path": input_images_path,
            "input_images_pil": input_images_pil,
            "target_img_size": target_img_size,
            "meta_data": meta_data,
            # "target_img_size": target_img_size,
            # "output_image": output_image,
            # "output_image_path": output_image_path,
        }
        return data


class RepeatedDistributedBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_repeats: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.num_repeats = num_repeats
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        self.samples_per_iter = self.num_replicas * batch_size
        assert self.samples_per_iter % self.num_repeats == 0, f"k can not div n*b, k{num_repeats}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.unique_samples_per_iter = self.samples_per_iter // self.num_repeats

        if self.drop_last and len(self.dataset) % self.unique_samples_per_iter != 0:  # type: ignore[arg-type]
            self.num_batches = len(self.dataset) // self.unique_samples_per_iter  # type: ignore[arg-type]
        else:
            self.num_batches = math.ceil(len(self.dataset) / self.unique_samples_per_iter)  # type: ignore[arg-type]
        
        self.total_size = self.num_batches * self.unique_samples_per_iter
        self.batch_size = self.unique_samples_per_iter * self.num_repeats
        self.epoch=0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        indices = indices[: self.total_size]

        for i in range(self.num_batches):
            start = i * self.unique_samples_per_iter
            end = start + self.unique_samples_per_iter
            batch_indices = indices[start:end]
            
            batch_indices = batch_indices * self.num_repeats

            shuffled_indices = torch.randperm(len(batch_indices), generator=g).tolist()
            shuffled_samples = [batch_indices[j] for j in shuffled_indices]
            
            yield shuffled_samples
    
    def __len__(self):
        return self.num_batches
    
    def set_epoch(self, epoch):
        self.epoch = epoch