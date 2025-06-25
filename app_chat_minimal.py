import dotenv

dotenv.load_dotenv(override=True)

import gradio as gr
import os
import argparse
import random
from datetime import datetime
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from accelerate import Accelerator
from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import OmniGen2ChatPipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from omnigen2.utils.img_util import create_collage

NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

pipeline = None
accelerator = None

def load_pipeline_minimal(accelerator, weight_dtype, args):
    """Minimal pipeline loading with maximum memory efficiency"""
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor
    from diffusers.models.autoencoders import AutoencoderKL
    import gc
    
    print("🍎 Loading OmniGen2 with Apple Silicon optimizations...")
    
    # Aggressive memory management
    if accelerator.device.type == 'mps':
        torch.mps.empty_cache()
        print("Cleared MPS cache")
    
    # Load components with maximum memory efficiency
    print("Loading transformer with low memory usage...")
    transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if accelerator.device.type != 'mps' else None,
    )
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.model_path,
        subfolder="vae", 
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )
    
    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler()
    
    print("Loading MLLM (this may take a while)...")
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        subfolder="mllm",
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if accelerator.device.type != 'mps' else None,
    )
    
    print("Loading processor...")
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_path,
        subfolder="processor",
        use_fast=False,
    )
    
    # Clear cache after each major component
    if accelerator.device.type == 'mps':
        torch.mps.empty_cache()
        gc.collect()
    
    print("Constructing pipeline...")
    pipeline = OmniGen2ChatPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        mllm=mllm,
        processor=processor
    )
    
    # Use sequential CPU offload for maximum memory efficiency on MPS
    print("Enabling sequential CPU offload for maximum memory efficiency...")
    pipeline.enable_sequential_cpu_offload()
    
    if accelerator.device.type == 'mps':
        torch.mps.empty_cache()
        print("Pipeline configured for MPS with sequential CPU offload")
    
    return pipeline

def run_simple(instruction, progress=gr.Progress()):
    """Simplified run function for testing"""
    
    if not instruction.strip():
        return None, "Please enter a prompt"
    
    try:
        # Set seed
        seed = random.randint(0, 2**16 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)  # Use CPU generator for MPS compatibility
        
        def progress_callback(cur_step, timesteps):
            frac = (cur_step + 1) / float(timesteps)
            progress(frac)
        
        print(f"Generating with prompt: {instruction}")
        
        results = pipeline(
            prompt=instruction,
            input_images=None,
            width=512,  # Smaller size for testing
            height=512,
            num_inference_steps=20,  # Fewer steps for testing
            max_sequence_length=512,  # Reduced sequence length
            text_guidance_scale=3.5,
            image_guidance_scale=1.0,
            negative_prompt=NEGATIVE_PROMPT,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
            step_func=progress_callback,
        )
        
        progress(1.0)
        
        if results.text.startswith("<|img|>"):
            vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
            output_image = create_collage(vis_images)
            return output_image, None
        else:
            return None, results.text
            
    except Exception as e:
        return None, f"Error: {str(e)}"

def main(args):
    global accelerator, pipeline
    
    # Simple interface for testing
    with gr.Blocks(title="🍎 OmniGen2 Apple Silicon Test") as demo:
        gr.Markdown("# 🍎 OmniGen2 Apple Silicon Minimal Test")
        gr.Markdown("Simplified interface for testing Apple Silicon compatibility")
        
        with gr.Row():
            instruction = gr.Textbox(
                label="Prompt",
                placeholder="A cat sitting on a table",
                lines=2
            )
        
        generate_btn = gr.Button("Generate", variant="primary")
        
        with gr.Row():
            output_image = gr.Image(label="Generated Image")
            output_text = gr.Textbox(label="Response", lines=4)
        
        # Initialize pipeline
        print("Initializing accelerator...")
        accelerator = Accelerator(mixed_precision="no")  # Force no mixed precision
        weight_dtype = torch.float32  # Force float32
        
        print("Loading pipeline...")
        pipeline = load_pipeline_minimal(accelerator, weight_dtype, args)
        
        generate_btn.click(
            run_simple,
            inputs=[instruction],
            outputs=[output_image, output_text]
        )
    
    demo.launch(share=args.share, server_port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="OmniGen2/OmniGen2")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(args)