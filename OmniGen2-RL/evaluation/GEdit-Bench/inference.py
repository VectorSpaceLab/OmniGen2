import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
import sys
from typing import List, Tuple

from PIL import Image, ImageOps

from omegaconf import OmegaConf
from tqdm import tqdm

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator
from accelerate import init_empty_weights

from datasets import load_dataset

from transformers import AutoProcessor, AutoModelForVision2Seq

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.hooks import apply_group_offloading

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


def parse_args(root_dir: str) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OmniGen2 image generation script.")
    parser.add_argument(
        "--load_from_pipeline",
        action="store_true",
        help="Load from pipeline.",
    )
    parser.add_argument(
        "--pipeline_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of experiment.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--transformer_lora_path",
        type=str,
        default=None,
        help="Path to transformer LoRA weights.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "euler_maruyama", "dpmsolver++"],
        help="Scheduler to use.",
    )
    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=50,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generation."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width."
    )
    parser.add_argument(
        "--max_input_image_pixels",
        type=int,
        default=1048576,
        help="Maximum number of pixels for each input image."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help="Data type for model weights."
    )
    parser.add_argument(
        "--text_guidance_scale",
        type=float,
        default=5.0,
        help="Text guidance scale."
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=2.0,
        help="Image guidance scale."
    )
    parser.add_argument(
        "--cfg_range_start",
        type=float,
        default=0.0,
        help="Start of the CFG range."
    )
    parser.add_argument(
        "--cfg_range_end",
        type=float,
        default=1.0,
        help="End of the CFG range."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt for generation."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Path to save results."
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload."
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable sequential CPU offload."
    )
    parser.add_argument(
        "--enable_group_offload",
        action="store_true",
        help="Enable group offload."
    )
    parser.add_argument(
        "--time_shift_base_res",
        type=int,
        default=320,
        help="Time shift base resolution."
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
        "--use_ori_neg_prompt_template",
        action="store_true",
        help="Use original negative prompt template."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None
    )
    args = parser.parse_args()
    
    if args.root_dir is None:
        args.root_dir = root_dir
    return args

def load_pipeline(args: argparse.Namespace, accelerator: Accelerator, weight_dtype: torch.dtype) -> OmniGen2Pipeline:
    if args.load_from_pipeline:
        pipeline = OmniGen2Pipeline.from_pretrained(
            args.pipeline_path,
            torch_dtype=weight_dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.pipeline_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
            local_files_only=True,
        )
    else:
        experiment_name = args.experiment_name
        experiment_dir = os.path.join(args.root_dir, 'experiments', experiment_name)

        conf = OmegaConf.load(os.path.join(experiment_dir, f"{experiment_name}.yml"))

        with init_empty_weights():
            transformer = OmniGen2Transformer2DModel(**conf.model.arch_opt)

        state_dict = torch.load(os.path.join(experiment_dir, args.model_path), mmap=True, weights_only=True)
        state_dict = torch.load(os.path.join(experiment_dir, args.model_path), mmap=True, weights_only=True)
        missing, unexpect = transformer.load_state_dict(state_dict, assign=True, strict=False)
        if len(missing) > 0 or len(unexpect) > 0:
            print(f"missed parameters: {missing}")
            print(f"unexpected parameters: {unexpect}")
        
        vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")

        mllm = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

        pipeline = OmniGen2Pipeline(
            transformer=transformer,
            vae=vae,
            mllm=mllm,
            processor=processor,
            scheduler=FlowMatchEulerDiscreteScheduler(),
        )

    if args.transformer_lora_path:
        print(f"LoRA weights loaded from {args.transformer_lora_path}")
        pipeline.load_lora_weights(args.transformer_lora_path,
                                   weight_name="pytorch_lora_weights.safetensors",
                                   local_files_only=True)

    if args.scheduler == "dpmsolver++":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler
    elif args.scheduler == "euler_maruyama":
        from omnigen2.schedulers.scheduling_flow_match_euler_maruyama_discrete import FlowMatchEulerMaruyamaDiscreteScheduler
        scheduler = FlowMatchEulerMaruyamaDiscreteScheduler(
            num_train_timesteps=args.num_inference_step,
            sigma_schedule="v3"
        )
        pipeline.scheduler = scheduler
    elif args.scheduler == "euler":
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=args.num_inference_step,
            time_shift_base_res=args.time_shift_base_res
        )
        pipeline.scheduler = scheduler

    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_group_offload:
        apply_group_offloading(pipeline.transformer, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.mllm, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.vae, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    else:
        pipeline = pipeline.to(device=accelerator.device)
    pipeline = pipeline.to(dtype=weight_dtype)
    return pipeline


def run(args: argparse.Namespace, 
        accelerator: Accelerator, 
        pipeline: OmniGen2Pipeline, 
        instruction: str, 
        negative_prompt: str, 
        input_images: List[Image.Image],
        target_img_size: Tuple[int, int],
        seed: int) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    if args.use_ori_neg_prompt_template:
        negative_prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": negative_prompt},
        ]
        negative_prompt = pipeline.processor.tokenizer.apply_chat_template(
            negative_prompt, tokenize=False, add_generation_prompt=False
        )

        negative_prompt_embeds, negative_prompt_attention_mask = pipeline._get_qwen2_prompt_embeds(
            prompt=negative_prompt, device=accelerator.device, max_sequence_length=1024
        )

        results = pipeline(
            prompt=[instruction],
            input_images=[input_images],
            size=[(target_img_size[0], target_img_size[1])],
            num_inference_steps=args.num_inference_step,
            max_sequence_length=1024,
            text_guidance_scale=args.text_guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            cfg_range=(args.cfg_range_start, args.cfg_range_end),
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
    else:
        results = pipeline(
            prompt=[instruction],
            input_images=[input_images],
            size=[(target_img_size[0], target_img_size[1])],
            num_inference_steps=args.num_inference_step,
            max_sequence_length=1024,
            text_guidance_scale=args.text_guidance_scale,
            image_guidance_scale=args.image_guidance_scale,
            cfg_range=(args.cfg_range_start, args.cfg_range_end),
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
    return results

def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return to_pil_image(canvas)

def main(args: argparse.Namespace, root_dir: str) -> None:
    """Main function to run the image generation process."""
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != 'fp32' else 'no')

    # Set weight dtype
    weight_dtype = torch.float32
    if args.dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.dtype == 'bf16':
        weight_dtype = torch.bfloat16

    # Load pipeline and process inputs
    pipeline = load_pipeline(args, accelerator, weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    test_dataset = load_dataset("stepfun-ai/GEdit-Bench", split='train')
    
    filtered_test_dataset = [
        item for item in test_dataset
        if item['instruction_language'] != 'cn'
    ]
    test_dataset = filtered_test_dataset

    process_index = Accelerator().process_index
    
    data_index = list(range(args.start_index, args.end_index))

    with tqdm(
            total=len(data_index),
            desc=f"process_index {process_index}: Processing {len(data_index)}/{len(test_dataset)}",
            unit="image",
            disable=not accelerator.is_main_process,
        ) as pbar:
        
        for idx in data_index:
            data_item = test_dataset[idx]

            task_type = data_item['task_type']
            instruction_language = data_item['instruction_language']

            key = data_item['key']
            instruction = data_item['instruction']
            input_image = data_item['input_image']

            ori_img_size = input_image.size
            new_img_size = (ori_img_size[0] // 16 * 16, ori_img_size[1] // 16 * 16)
            input_images = [input_image.resize(new_img_size)]

            for turn in range(args.num_samples):
                results = run(args, accelerator, pipeline, instruction, args.negative_prompt, input_images, new_img_size, args.seed + idx + turn * len(test_dataset))
                output_image = results.images[0]
                output_image = output_image.resize(ori_img_size)

                sub_dir = os.path.join(args.result_dir, "fullset", task_type, instruction_language)
                os.makedirs(sub_dir, exist_ok=True)

                if turn > 0:
                    output_image.save(os.path.join(sub_dir, f"{key}_sample{turn}.png"))
                else:
                    input_image.save(os.path.join(sub_dir, f"{key}_SRCIMG.png"))
                    output_image.save(os.path.join(sub_dir, f"{key}.png"))

            pbar.update(1)

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
    args = parse_args(root_dir)
    main(args, root_dir)