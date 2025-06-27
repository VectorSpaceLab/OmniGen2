import os
import datasets
from datasets import load_dataset, ClassLabel, concatenate_datasets
import torch
import numpy as np
import random
from PIL import Image
import json
import copy
# import torchvision.transforms as T
from torchvision import transforms
import pickle
import time
import re
import yaml
from transformers import AutoTokenizer
from ..utils import normalize_whitespace


from accelerate.logging import get_logger
# logger = logging.getLogger(__name__)
logger = get_logger(__name__)


def crop_arr(pil_image, max_image_size, img_scale_num):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * max_image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if max(*pil_image.size) > max_image_size:
        scale = max_image_size / max(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    if min(*pil_image.size) < img_scale_num:
        scale = img_scale_num / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    
    arr = np.array(pil_image)
    crop_y1 = (arr.shape[0] % img_scale_num) // 2
    crop_y2 = arr.shape[0] % img_scale_num - crop_y1

    crop_x1 = (arr.shape[1] % img_scale_num) // 2
    crop_x2 = arr.shape[1] % img_scale_num - crop_x1

    arr = arr[crop_y1:arr.shape[0]-crop_y2, crop_x1:arr.shape[1]-crop_x2]
    
    return Image.fromarray(arr)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def crop_arr_fixratio(pil_image, max_image_size, img_scale_num):
    fixratio = [[16, 9], [1, 1]]
    raise NotImplementedError


class OmniGen2TrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_path: str, 
        image_size: int,
        tokenizer,
        apply_chat_template: bool,
        return_raw_image: bool = False,
        dynamic_image_size: bool = True,
        prompt_dropout_prob: float = 0.0,
        ref_img_dropout_prob: float = 0.0,
        output_min_side_length: int = 0,
        output_min_pixels: int = 0,
    ):
        self.dynamic_image_size = dynamic_image_size
        self.mode = mode
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob
        self.apply_chat_template_on_null_prompt = apply_chat_template_on_null_prompt
        self.specified_null_chat_template = specified_null_chat_template
        self.null_prompt_version = null_prompt_version
        self.output_min_side_length = output_min_side_length
        self.output_min_pixels = output_min_pixels
        self.max_size_op_edit = max_size_op_edit
        
        logger.info(f"read dataset config from {config_path}", main_process_only=False)
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("DATASET CONFIG:", main_process_only=False)
        logger.info(self.config, main_process_only=False)

        self.tokenizer = tokenizer
        self.apply_chat_template = apply_chat_template

        self.max_prompt_length = self.config["max_prompt_length"]
        self.condition_dropout_prob = self.config["condition_dropout_prob"]

        self.max_image_size = image_size
        self.max_input_image_size = self.config["max_input_image_size"]
        assert isinstance(self.max_input_image_size, list)

        self.img_scale_num = 16 

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        self.return_raw_image = return_raw_image

        self.multi_datasets_name = []
        self.multi_datasets = []
        self.multi_datasets_ratio = []
        self.dataset_len = 0
        for i in range(len(self.config['data'])):
            task_data = self.config['data'][i]
            self.multi_datasets_name.append(task_data['type'])
            self.multi_datasets.append(self.get_data(task_data['path']))
            self.multi_datasets_ratio.append(task_data['ratio'])

            self.dataset_len += len(self.multi_datasets[-1])

        logger.info(f"Total dataset len: {self.dataset_len}", main_process_only=False)
        
    def get_data(self, json_path):
        all_tasks = []
        for line in open(json_path, 'r'):
            line = line.rstrip('\n')
            if len(line) > 0:
                all_tasks.append(json.loads(line))

        all_dataset = []
        for task in all_tasks:
            # temp_dataset = self.load_files(task['json_file'], cache_dir='/share_2/shitao/projects/DiffusionGPT2/processed_data/cache_dir')
            temp_dataset = self.load_files(task['json_file'])
            if temp_dataset[0]['task_type'] == 'text_to_image' and 'input_images' in temp_dataset[0]:
                temp_dataset = temp_dataset.remove_columns(['input_images'])
            if 'index' in temp_dataset[0]:
                temp_dataset = temp_dataset.remove_columns(['index'])

            if task['ratio'] > 1:
                for _ in range(int(task['ratio'])):
                    all_dataset.append(temp_dataset)
            elif task['ratio'] < 1:
                temp_num = len(temp_dataset)
                temp_sample_num = int(len(temp_dataset) * task['ratio'])
                all_dataset.append(temp_dataset.select(list(range(temp_sample_num))))
            else:
                all_dataset.append(temp_dataset)

        logger.info("Start concatenate dataset", main_process_only=False)
        start_time = time.time()
        concatenated_dataset = concatenate_datasets(all_dataset)
        logger.info(f"concatenate dataset cost {time.time() - start_time}", main_process_only=False)
        return concatenated_dataset

    def load_files(self, path, cache_dir: str=None):
        logger.info(f"Loading file {path}", main_process_only=False)
        if os.path.isdir(path):
            data = load_dataset('json', data_files=[os.path.join(path, x) for x in os.listdir(path) if ('.json' in x or '.jsonl' in x) and 'preprocessed_length' not in x ], cache_dir=cache_dir)['train']
        else:
            data = load_dataset('json', data_files=path, cache_dir=cache_dir)['train']
        return data
    
    def process_example(self, example, data_name):
        task_type, instruction, output_image = (example['task_type'], example['instruction'], example['output_image'])
        if 'input_images' in example:
            input_images =  example['input_images']
            if input_images is not None:
                for i in range(len(input_images)):
                    if 'mat_v1_crop_1img_face' in input_images[i]:
                        input_images[i] = input_images[i].replace('mat_v1_crop_1img_face', 'mat_v1_crop_1img_face_v2')
        else:
            input_images = None

        if 'instruction_long' in example and example['instruction_long'] is not None:
            if random.random() < 0.8 and len(example['instruction_long']) > 10:
                instruction = example['instruction_long']
        
        if 'instruction_short' in example and example['instruction_short'] is not None:
            if random.random() < 0.3 and len(example['instruction_short']) > 10:
                instruction = example['instruction_short']

        if any(t2i_task_type in task_type for t2i_task_type in ['text_to_image', 't2i']):
            if all(key in example and example[key] is not None for key in ['instruction', 'instruction_zh', 'instruction_short', 'instruction_short_zh']):
                instruction = random.choice(
                    [
                        example[key]
                        for key in [
                            "instruction",
                            "instruction_zh",
                            "instruction_short",
                            "instruction_short_zh",
                        ]
                        if len(example[key]) > 10
                    ]
                )
            elif 'instruction_zh' in example:
                if "instruction_tag" in example and example['instruction_tag'] is not None and len(example['instruction_tag']) > 3 and random.random() < 0.08:
                    instruction = example['instruction_tag']
                elif random.random() < 0.5 and example['instruction_zh'] is not None and len(example['instruction_zh']) > 10:
                    instruction = example['instruction_zh']
                else:
                    instruction = example['instruction']
        
            if "caption_qwenvl2_5" in example and example['caption_qwenvl2_5'] is not None:
                randn_num = random.random()
                if len(example["caption_qwenvl2_5"]) < 1000 and randn_num < 0.4 and len(example['caption_qwenvl2_5']) > 10:
                    instruction = example['caption_qwenvl2_5']
                elif len(example['caption_qwenvl2_5_zh']) < 1000 and randn_num < 0.99 and len(example['caption_qwenvl2_5_zh']) > 10:
                    instruction = example['caption_qwenvl2_5_zh']
                else:
                    instruction = example['instruction']
            
            if "Hyper-Realistic photo. Photo of " in instruction:
                instruction = instruction.replace("Hyper-Realistic photo. Photo of ", "")
            if "Hyper-Realistic photo. " in instruction:
                instruction = instruction.replace("Hyper-Realistic photo. ", "")
            
        if task_type == 'segementation':
            instruction = instruction.replace('"', '')

        ai_instructions = [". The image looks like it was generated by AI."]
        # if 'midjourney' in task_type or 'citivai' in task_type or 'generated_by_ai' in task_type or 'g_ai' in task_type or data_name == 'ai' or 'civitai_hq' in task_type:
        if 'generated_by_ai' in task_type or 'g_ai' in task_type:
            if random.random() < 0.5:
                instruction = instruction + random.choice(ai_instructions)
            # print(data_name, instruction)
        
        prefix_str = prefix = ["The image portrays ", "The image depicts ", "The image captures ", "The image highlights ", "The image shows ", "这张图片展示了"]
        if "text_to_image" in task_type or "t2i" in task_type:
            if random.random() < 0.5:
                for p in prefix_str:
                    if p in instruction:
                        instruction = instruction.replace(p, "")
                        break
        
        if 'ai' in data_name:
            dof_str = [
                "Foreground in focus, background blurred.",
                "Shallow depth of field.",
                "Bokeh background, sharp foreground.",
                "Subject in sharp focus, background out of focus.",
                "Clear foreground, soft background blur.",
                "Portrait mode effect (AI often understands this from phone cameras).",
                "Focused on the foreground, heavily blurred background.",
                "Prominent foreground, defocused background.",
                "Cinematic depth of field, foreground in focus.",
                "Blur the background.",
                "Create a soft focus in the background.",
                "Make the background less distinct.",
                "Apply a background blur effect.",
                "Fade the background into a blur.",
            ]
            instruction += " " + random.choice(dof_str)
        
        if "gpt4o" in data_name:
            dof_str = [
                "The overall tone of the image is little yellowish.",
            ]
            instruction += " " + random.choice(dof_str)

        if "/share_2/shitao/datasets/PT/BLIP3o-60k/data/" in output_image:
            output_image = output_image.replace("/share_2/shitao/datasets/PT/BLIP3o-60k/data/", "/share_2/chenyuan/data/Awesome_gpt4o_images/PT/BLIP3o-60k/imgs_flux_1_cool_shift10/")
            output_image = output_image.replace("jpg", "png")
        if "BLIP3o" in output_image:
            assert "chenyuan" in output_image
            # print(output_image)
        
        unnecessary_words = ["<img>", "</img>", "<|image_1|>", "<|image_2|>", "<|image_3|>", "<|image_4|>", "<|image_5|>",
                             "<|img_1|>", "<|img_2|>", "<|img_3|>", "<|img_4|>", "<|img_5|>"]
        for word in unnecessary_words:
            instruction = instruction.replace(word, '')
        
        instruction = normalize_whitespace(instruction)
        return task_type, instruction, input_images, output_image

    def image_augmentation(self, images, noise_prob:float=0.6, flip_prob:float=0.0):
        def add_noise(inputs, noise_factor=0.3):
            noisy = inputs + torch.randn_like(inputs) * noise_factor
            noisy = torch.clip(noisy, 0., 1.)
            return noisy

        aug_images = []
        for img in images:
            if random.random() < flip_prob:
                img = transforms.RandomHorizontalFlip(p=1)(img)

            if random.random() < noise_prob:
                img = add_noise(transforms.ToTensor()(img), random.choice([0.3, 0.4, 0.5, 0.6]))
                img = transforms.ToPILImage()(img)
            
            if random.random() < 0.2:
                img = transforms.RandomRotation(random.randint(1, 350))(img)
            if random.random() < 0.2:
                img = transforms.Grayscale(num_output_channels=3)(img)

            aug_images.append(img)    
        return aug_images

    def get_example(self):
        cur_dataset_idx = np.random.choice(a=list(range(len(self.multi_datasets_ratio))), p=self.multi_datasets_ratio)
        cur_dataset = self.multi_datasets[cur_dataset_idx]
        index = random.randint(0, len(cur_dataset)-1)
        return self.multi_datasets_name[cur_dataset_idx], cur_dataset[index]

    def getitem(self, example, data_name):
        task_type, instruction, input_images_path, output_image_path = self.process_example(example, data_name)

        if 'op_edit' in data_name:
            max_size = self.max_size_op_edit
        else:
            max_size = self.max_image_size

        if random.random() < self.condition_dropout_prob:
            instruction = '<cfg>'
            instruction = self.process_instruction(instruction)
            input_images = None
        else:
            if input_images_path is not None and len(input_images_path) > 0:
                input_images = [Image.open(x).convert('RGB') for x in input_images_path]
                if 'multi-iamge,grit' in task_type or 'multi-images,humman' in task_type or task_type == 'multi-iamge,openimage':
                    if 'msdiff' not in task_type:
                        input_images = self.image_augmentation(input_images, noise_prob=0.1, flip_prob=0.3)
                
                if len(input_images) == 1: max_size = min(max_size, self.max_input_image_size[0])
                if len(input_images) == 2: max_size = min(max_size, self.max_input_image_size[1])
                if len(input_images) == 3: max_size = min(max_size, self.max_input_image_size[2])
                if len(input_images) >= 4: max_size = min(max_size, self.max_input_image_size[3])
                max_size = min(max_size, self.max_image_size)
                input_images = [self.process_image(x, max_size=max_size) for x in input_images]
            else:
                input_images = None
        
        output_image = Image.open(output_image_path).convert('RGB')
        raw_image = None
        if self.return_raw_image:
            raw_image = transforms.ToTensor()(output_image)

        out_size = self.max_image_size
        if 'op_edit' in data_name and self.max_size_op_edit is not None:
            out_size = min(out_size, self.max_size_op_edit)

        output_image = self.process_image(output_image, max_size=out_size, output_min_length=self.output_min_length, output_min_pixels=self.output_min_pixels)

        if self.apply_chat_template:
            instruction = [{"role": "user", "content": instruction}]
            instruction = self.tokenizer.apply_chat_template(instruction, tokenize=False, add_generation_prompt=False)
            if "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." in instruction:
                instruction = instruction.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
            else:
                instruction = instruction.replace("You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
        
            if data_name == 'ai':
                instruction = instruction.replace("You are a helpful assistant that generates high-quality images based on user instructions.", "You are a helpful AI that generates images with superior degree of image-text alignment based on user prompts. The generated image focuses on the main subject, with a blurred background.")
                instruction = instruction.replace("Hyper-Realistic photo. ", "")
                
        if random.random() < self.prompt_dropout_prob:
            if self.null_prompt_version == 'v1':
                instruction = ""
            elif self.null_prompt_version == 'v2':
                instruction = "<image>"
                
            if self.apply_chat_template_on_null_prompt:
                instruction = [{"role": "user", "content": instruction}]
                instruction = self.tokenizer.apply_chat_template(instruction, tokenize=False, add_generation_prompt=False)
                if self.specified_null_chat_template:
                    if "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." in instruction:
                        instruction = instruction.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant that generates images.")
                    else:
                        instruction = instruction.replace("You are a helpful assistant.", "You are a helpful assistant that generates images.")
                else:
                    if "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." in instruction:
                        instruction = instruction.replace("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
                    else:
                        instruction = instruction.replace("You are a helpful assistant.", "You are a helpful assistant that generates high-quality images based on user instructions.")
            if random.random() < self.ref_img_dropout_prob:
                input_images = None

        return instruction, input_images, input_images_path, output_image, output_image_path, raw_image

    def process_image(self, image, max_size:int=None, output_min_length:int=0, output_min_pixels:int=0):
        if self.dynamic_image_size:
            image = crop_arr(image, max_size, self.img_scale_num)
        else:
            image = center_crop_arr(image, max_size)
        output = self.image_transform(image)

        sizes = [output.size(-2), output.size(-1)]
        if max(sizes) > 3072 or min(sizes) < (self.img_scale_num * 2) or min(sizes) < output_min_length or sizes[0] * sizes[1] < output_min_pixels:
            raise RuntimeError(f"error size: {image.size}")
        return output

    def __getitem__(self, index):
        # data_name, example = self.get_example()
        # return self.getitem(example, data_name)
        if self.mode == 'train':
            for _ in range(12):
                try:
                    data_name, example = self.get_example()
                    return_data = self.getitem(example, data_name)
                    input_ids = self.tokenizer.encode(return_data[0])
                    if len(input_ids) > self.max_prompt_length:
                        raise RuntimeError(f"cur number of tokens={len(input_ids)}, larger than max_prompt_length={self.max_prompt_length}")
                    return return_data
                except Exception as e:
                    print("error when loading data: ", e)
                    print(data_name, example['task_type'])
            raise RuntimeError("Too many bad data.")
        else:
            data_name, example = self.multi_datasets_name[0], self.multi_datasets[0][index]
            return_data = self.getitem(example, data_name)
            input_ids = self.tokenizer.encode(return_data[0])
            if len(input_ids) > self.max_prompt_length:
                raise RuntimeError(f"cur number of tokens={len(input_ids)}, larger than max_prompt_length={self.max_prompt_length}")
            
            return_data = return_data + (example['target_img_size'],)
            return return_data
        
    def __len__(self):
        return self.dataset_len

class OmniGen2Collator:
    def __init__(self, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __call__(self, batch):
        instruction = [x[0] for x in batch]
        input_images = [x[1] for x in batch]
        input_images_path = [x[2] for x in batch]
        output_image = [x[3] for x in batch]
        output_image_path = [x[4] for x in batch]
        raw_images = [x[5] for x in batch] if batch[0][5] is not None else None

        text_inputs = self.tokenizer(
            instruction,
            padding="longest",
            max_length=self.max_token_len,
            truncation=True,
            return_tensors="pt",
        )

        data = {
            "text_ids": text_inputs.input_ids,
            "text_mask": text_inputs.attention_mask,
            "input_images": input_images,
            "input_images_path": input_images_path,
            "output_image": output_image,
            "output_image_path": output_image_path,
            "raw_images": raw_images,
        }
        return data