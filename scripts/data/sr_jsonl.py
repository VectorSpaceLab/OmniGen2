import dotenv

dotenv.load_dotenv(override=True)

import os
import sys
import json
import argparse
import random
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_tensor, to_pil_image
from accelerate.utils.random import set_seed

from transformers import AutoTokenizer

from omnigen.dataset.jsonl_dataset import JsonlDataset
from omnigen.models.rrdbnet_arch import RRDBNet

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=12
    )       
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=2
    )
    parser.add_argument(
        "--save_file",
        type=str,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=998244353
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0
    )
    parser.add_argument(
        "--end_index",
        type=int,
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=20
    )
    parser.add_argument(
        "--task",
        type=str,
        default="human"
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="mixed"
    )
    parser.add_argument(
        "--num_load_images",
        type=int,
        default=None
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="v1"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=16384
    )
    
    args = parser.parse_args()
    return args


def main(args, root_dir):
    set_seed(args.seed)

    if args.data_source == "real":
        data_files = [
            "/share/shitao/wyz/datasets/Images/folder0/jsonls/caption_qwenvl2_5_new_preprocessed_preprocessed_length3.jsonl",
            "/share/shitao/wyz/datasets/Images/folder1/jsonls/caption_qwenvl2_5_new_preprocessed_preprocessed_length3.jsonl",
            "/share/shitao/wyz/datasets/Images/folder2/jsonls/caption_qwenvl2_5_new_preprocessed_preprocessed_length3.jsonl",
            "/share/shitao/wyz/datasets/Images/folder3/jsonls/caption_qwenvl2_5_new_preprocessed_preprocessed_length3.jsonl",
            "/share_2/shitao/datasets/Images/folder4/jsonls/caption_qwenvl2_5_new_preprocessed_preprocessed_length3.jsonl",
            "/share/shitao/wyz/datasets/Images/folder5/jsonls/caption_qwenvl2_5_new_preprocessed_preprocessed_length3.jsonl",
            "/share/shitao/wyz/datasets/Images/folder7/jsonls/caption_qwenvl2_5_preprocessed_length3.jsonl",
        ]

        data_dir = "/share/shitao/wyz/datasets/Images"
        save_dir = "/share_2/luoxin/datasets/image_enhancement"

    if args.num_load_images is None:
        args.num_load_images = args.end_index
    
    num_images = args.end_index - args.start_index

    print(f"{args.start_index=} {args.end_index=} {args.num_load_images=}")

    dataset = JsonlDataset(data_files, num_images=args.num_load_images, largest_side=1024, read_keys=["output_image"])
    print(f"total loaded datas: {len(dataset)}")

    dataset = Subset(dataset, range(args.start_index, args.end_index))

    print(f"total processing datas: {len(dataset)}")

    assert num_images == len(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: x,
    )

    json_file = os.path.join(args.save_file)
    os.makedirs(os.path.dirname(json_file), exist_ok=True)

    sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    state_dict = torch.load("pretrained_models/RealESRGAN_x4plus.pth")['params_ema']
    # state_dict.pop('conv_first.weight')
    sr_model.load_state_dict(state_dict, strict=False)
    sr_model = sr_model.eval().to("cuda")
    
    with tqdm(total=num_images, desc=f'Processing {num_images}: from {args.start_index} to {args.end_index}', unit='image') as pbar:
        for batch in dataloader:
            data = batch[0]
            # input_images_path = data["input_images_path"]
            output_image_path = data["output_image_path"]

            # input_images = data["input_images"]
            output_image = data["output_image"]
            output_image_ori = output_image

            # input_images_sr = []
            # for input_image in input_images:
            #     input_image = to_tensor(input_image).unsqueeze(0).to("cuda")
            #     sr_image = sr_model(input_image)
            #     sr_image = to_pil_image(sr_image.squeeze(0).cpu())
            #     input_images_sr.append(sr_image)
            
            ori_size = output_image.size
            # output_image = output_image.resize((ori_size[0] // 4 * 4, ori_size[1] // 4 * 4))
            output_image = output_image.resize((ori_size[0] // 4, ori_size[1] // 4))
            output_image_tensor = to_tensor(output_image).unsqueeze(0).to("cuda")
            output_image_sr = sr_model(output_image_tensor)
            output_image_sr = output_image_sr.clamp(0, 1)
            output_image_sr = to_pil_image(output_image_sr.squeeze(0).cpu())
            output_image_sr.resize(ori_size)

            # input_images_sr_path = []
            # for input_image_sr, input_image_path in zip(input_images_sr, input_images_path):
            #     relpath = os.path.relpath(data_dir, input_image_path)
            #     save_path = os.path.join(save_dir, relpath)
            #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #     input_image_sr.save(save_path)
            #     input_images_sr_path.append(save_path)

            output_image_sr_path = os.path.join(save_dir, os.path.relpath(output_image_path, data_dir))
            os.makedirs(os.path.dirname(output_image_sr_path), exist_ok=True)
            output_image_sr.save(output_image_sr_path, quality=100)

            name, ext = os.path.splitext(output_image_sr_path)
            print(name, ext)
            output_image_ori.save(name + "_ori" + ext, quality=100)

            json_line = data['json_data']
            # json_line['input_images'] = input_images_sr_path
            json_line['output_image'] = output_image_sr_path

            with open(json_file, 'a') as f:
                f.write(json.dumps(json_line) + '\n')

if __name__ == "__main__":
    args = parse_args()
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    main(args, root_dir)