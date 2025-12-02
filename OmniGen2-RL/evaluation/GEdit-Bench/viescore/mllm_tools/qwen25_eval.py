import os
import torch
import time
from PIL import Image
from typing import List
from transformers import AutoModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from transformers import AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer
import requests
from io import BytesIO
import random
import numpy as np
import base64
import magic
import megfile

def process_image(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def convert_image_to_base64(file_content):
    mime_type = magic.from_buffer(file_content, mime=True)
    base64_encoded_data = base64.b64encode(file_content).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Qwen25():
    def __init__(self) -> None:     
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        self.model = AutoModelForCausalLM.from_pretrained(
            "/share_2/luoxin/modelscope/hub/models/Qwen/Qwen2.5-72B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("/share_2/luoxin/modelscope/hub/models/Qwen/Qwen2.5-72B-Instruct")

        print(f"Using {attn_implementation} for attention implementation")

    def get_parsed_output(self, input_string):
        set_seed(42)
        # Prepare the inputs
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"""Please rectify following json string to correct format. Expected format: 
            {
            "score" : [...],
            "reasoning" : "..."
            }
            Below is the input string:
            {input_string}"""}
        ]
                
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Process inputs
        inputs = self.tokenizer([text], return_tensors="pt").to(model.device)
        inputs = inputs.to("cuda")

        generation_config = {
            "max_new_tokens": 512,
            "num_beams": 1,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
        }

        generated_ids = self.model.generate(**inputs, **generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        return output_text[0] if output_text else ""

if __name__ == "__main__":
    model = Qwen25()
    prompt = model.prepare_prompt(
        ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"], 
        'Describe the image in detail.'
    )
    res = model.get_parsed_output(prompt)
    print("result : \n", res)