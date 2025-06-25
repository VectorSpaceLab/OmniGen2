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
from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from omnigen2.utils.img_util import create_collage

NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

pipeline = None
accelerator = None
save_images = False


def load_pipeline(accelerator, weight_dtype, args):
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor
    from diffusers.models.autoencoders import AutoencoderKL
    
    # Load individual components manually to avoid remote code
    print("Loading transformer...")
    transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.model_path,
        subfolder="vae", 
        torch_dtype=weight_dtype,
    )
    
    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler()
    
    print("Loading MLLM...")
    mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        subfolder="mllm",
        torch_dtype=weight_dtype,
    )
    
    print("Loading processor...")
    processor = Qwen2VLProcessor.from_pretrained(
        args.model_path,
        subfolder="processor",
        use_fast=False,  # Explicitly use slow processor for compatibility
    )
    
    # Manually construct the pipeline
    print("Constructing pipeline...")
    pipeline = OmniGen2ChatPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        mllm=mllm,
        processor=processor
    )
    
    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(accelerator.device)
        # For MPS, ensure all components are properly moved and synced
        if accelerator.device.type == 'mps':
            torch.mps.synchronize()
            print(f"Pipeline moved to MPS device successfully")
    return pipeline

def run(
    instruction,
    width_input,
    height_input,
    scheduler,
    num_inference_steps,
    image_input_1,
    image_input_2,
    image_input_3,
    negative_prompt,
    guidance_scale_input,
    img_guidance_scale_input,
    cfg_range_start,
    cfg_range_end,
    num_images_per_prompt,
    max_input_image_side_length,
    max_pixels,
    seed_input,
    progress=gr.Progress(),
):
    input_images = [image_input_1, image_input_2, image_input_3]
    input_images = [img for img in input_images if img is not None]

    if len(input_images) == 0:
        input_images = None

    if seed_input == -1:
        seed_input = random.randint(0, 2**16 - 1)

    generator = torch.Generator(device=accelerator.device).manual_seed(seed_input)

    def progress_callback(cur_step, timesteps):
        frac = (cur_step + 1) / float(timesteps)
        progress(frac)

    if scheduler == 'euler':
        pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
    elif scheduler == 'dpmsolver':
        pipeline.scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=width_input,
        height=height_input,
        max_input_image_side_length=max_input_image_side_length,
        max_pixels=max_pixels,
        num_inference_steps=num_inference_steps,
        max_sequence_length=1024,
        text_guidance_scale=guidance_scale_input,
        image_guidance_scale=img_guidance_scale_input,
        cfg_range=(cfg_range_start, cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        output_type="pil",
        step_func=progress_callback,
    )

    progress(1.0)

    if results.text.startswith("<|img|>"):
        vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
        output_image = create_collage(vis_images)

        if save_images:
            # Create outputs directory if it doesn't exist
            output_dir = os.path.join(ROOT_DIR, "outputs_gradio")
            os.makedirs(output_dir, exist_ok=True)

            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

            # Generate unique filename with timestamp
            output_path = os.path.join(output_dir, f"{timestamp}.png")
            # Save the image
            output_image.save(output_path)

            # Save All Generated Images
            if len(results.images) > 1:
                for i, image in enumerate(results.images):
                    image_name, ext = os.path.splitext(output_path)
                    image.save(f"{image_name}_{i}{ext}")

        return output_image, None
    else:
        return None, results.text

def get_example():
    case = [
        [
            "Please briefly describe this image.",
            1024,
            1024,
            'euler',
            50,
            "example_images/1e5953ff5e029bfc81bb0a1d4792d26d.jpg",
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "A stylish woman walking down a Parisian street",
            1024,
            1024,
            'euler',
            50,
            None,
            None,
            None,
            NEGATIVE_PROMPT,
            3.5,
            1.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
        [
            "Add a fisherman hat to the woman's head",
            1024,
            1024,
            'euler',
            50,
            os.path.join(ROOT_DIR, "example_images/flux5.png"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            0.0,
            1.0,
            1,
            2048,
            1024 * 1024,
            0,
        ],
    ]
    return case

def main(args):
    # Enhanced CSS for modern styling
    css = """
    .main-container { 
        max-width: 1400px; 
        margin: 0 auto; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .generate-btn { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 18px !important;
        padding: 15px 30px !important;
        border-radius: 12px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2) !important;
    }
    
    .generate-btn:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    .settings-group {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #dee2e6;
    }
    
    .output-section {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .quick-settings {
        background: #e3f2fd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }
    
    .advanced-settings {
        background: #fff3e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #ff9800;
    }
    
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4caf50;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    """
    
    with gr.Blocks(css=css, title="🎨 OmniGen2 Studio", theme=gr.themes.Soft()) as demo:
        # Header
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 30px; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
            <h1 style="margin: 0 0 15px 0; font-size: 2.5em; font-weight: 300;">
                🎨 OmniGen2 Multimodal AI Studio
            </h1>
            <p style="margin: 0; font-size: 1.2em; opacity: 0.9;">
                Advanced Text-to-Image Generation • Visual Understanding • Image Editing
            </p>
            <div style="margin-top: 20px;">
                <a href="https://arxiv.org/abs/2506.18871" target="_blank" style="color: white; text-decoration: none; margin: 0 15px; opacity: 0.8; transition: opacity 0.3s;">📄 Research Paper</a>
                <a href="https://github.com/VectorSpaceLab/OmniGen2" target="_blank" style="color: white; text-decoration: none; margin: 0 15px; opacity: 0.8; transition: opacity 0.3s;">💻 Original Source</a>
                <a href="https://github.com/zettai-seigi/OmniGen2" target="_blank" style="color: white; text-decoration: none; margin: 0 15px; opacity: 0.8; transition: opacity 0.3s;">🍎 Apple Silicon Optimized</a>
            </div>
        </div>
        """)
        
        # Quick Tips
        with gr.Row():
            gr.HTML("""
            <div style="background: linear-gradient(145deg, #e8f5e8, #f1f8e9); padding: 20px; border-radius: 12px; border-left: 5px solid #4caf50;">
                <h3 style="color: #2e7d32; margin-top: 0;">💡 Pro Tips for Best Results</h3>
                <ul style="color: #388e3c; margin-bottom: 0; line-height: 1.6;">
                    <li><strong>Image Quality:</strong> Use high-resolution images (512x512 minimum)</li>
                    <li><strong>Be Specific:</strong> "Add the bird from first image to the garden in second image"</li>
                    <li><strong>Guidance Scale:</strong> Image editing (1.3-2.0) | In-context generation (2.0-3.0)</li>
                    <li><strong>Language:</strong> English prompts currently work best</li>
                </ul>
            </div>
            """)
            
        with gr.Row():
            # Left Column - Inputs
            with gr.Column(scale=3):
                # Main Prompt Section
                with gr.Group():
                    gr.Markdown("### ✍️ **Creative Prompt**")
                    instruction = gr.Textbox(
                        label="🎯 Describe your vision",
                        placeholder="Create a beautiful landscape with mountains and a lake at sunset...",
                        lines=4,
                        info="💡 Reference uploaded images using 'first image', 'second image', etc."
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            negative_prompt = gr.Textbox(
                                label="🚫 Negative Prompt",
                                placeholder="What to avoid in the image...",
                                value=NEGATIVE_PROMPT,
                                lines=2,
                            )
                        with gr.Column(scale=1):
                            seed_input = gr.Number(
                                label="🎲 Seed",
                                value=-1,
                                info="Use -1 for random"
                            )

                # Image Upload Section
                with gr.Group():
                    gr.Markdown("### 🖼️ **Reference Images**")
                    with gr.Row():
                        image_input_1 = gr.Image(
                            label="🥇 First Reference", 
                            type="pil",
                            height=180
                        )
                        image_input_2 = gr.Image(
                            label="🥈 Second Reference", 
                            type="pil", 
                            height=180
                        )
                        image_input_3 = gr.Image(
                            label="🥉 Third Reference", 
                            type="pil",
                            height=180
                        )

                # Generate Button
                generate_button = gr.Button(
                    "🎨 Generate Masterpiece",
                    elem_classes=["generate-btn"],
                    size="lg"
                )

                # Quick Settings
                with gr.Group(elem_classes=["quick-settings"]):
                    gr.Markdown("### ⚡ **Quick Settings**")
                    with gr.Row():
                        width_input = gr.Slider(
                            label="📐 Width", 
                            minimum=256, 
                            maximum=1024, 
                            value=1024, 
                            step=64
                        )
                        height_input = gr.Slider(
                            label="📏 Height", 
                            minimum=256, 
                            maximum=1024, 
                            value=1024, 
                            step=64
                        )
                    
                    with gr.Row():
                        text_guidance_scale_input = gr.Slider(
                            label="📝 Text Guidance", 
                            minimum=1.0, 
                            maximum=8.0, 
                            value=5.0, 
                            step=0.1,
                            info="How closely to follow the text prompt"
                        )
                        image_guidance_scale_input = gr.Slider(
                            label="🖼️ Image Guidance", 
                            minimum=1.0, 
                            maximum=3.0, 
                            value=2.0, 
                            step=0.1,
                            info="How closely to follow reference images"
                        )

                # Advanced Settings (Collapsible)
                with gr.Accordion("🔧 Advanced Settings", open=False):
                    with gr.Group(elem_classes=["advanced-settings"]):
                        with gr.Row():
                            scheduler_input = gr.Dropdown(
                                label="🔄 Scheduler",
                                choices=["euler", "dpmsolver"],
                                value="euler",
                                info="Algorithm for denoising"
                            )
                            num_inference_steps = gr.Slider(
                                label="🔢 Steps", 
                                minimum=20, 
                                maximum=100, 
                                value=50, 
                                step=5,
                                info="More steps = higher quality, slower"
                            )
                        
                        with gr.Row():
                            cfg_range_start = gr.Slider(
                                label="CFG Start",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.0,
                                step=0.1,
                            )
                            cfg_range_end = gr.Slider(
                                label="CFG End",
                                minimum=0.0,
                                maximum=1.0,
                                value=1.0,
                                step=0.1,
                            )
                        
                        with gr.Row():
                            num_images_per_prompt = gr.Slider(
                                label="🎯 Images per prompt",
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                            )
                            max_input_image_side_length = gr.Slider(
                                label="Max input size",
                                minimum=256,
                                maximum=2048,
                                value=1024,
                                step=256,
                            )

            # Right Column - Output
            with gr.Column(scale=2):
                with gr.Group(elem_classes=["output-section"]):
                    gr.Markdown("### 🎨 **Generated Content**")
                    
                    # Status indicator
                    gr.HTML("""
                    <div style="margin-bottom: 15px;">
                        <span class="status-indicator"></span>
                        <span style="color: #4caf50; font-weight: 500;">Ready to create</span>
                    </div>
                    """)
                    
                    save_images_checkbox = gr.Checkbox(
                        label="💾 Auto-save generated images", 
                        value=False,
                        info="Images will be saved to outputs_gradio/"
                    )
                    
                    output_image = gr.Image(
                        label="🖼️ Generated Image",
                        height=400,
                        show_download_button=True
                    )
                    
                    output_text = gr.Textbox(
                        label="📝 Model Response",
                        placeholder="AI responses and descriptions will appear here...",
                        lines=4,
                        interactive=False,
                        show_copy_button=True
                    )

        # Load pipeline
        global accelerator, pipeline
        bf16 = False
        accelerator = Accelerator(mixed_precision="bf16" if bf16 else "no")
        weight_dtype = torch.bfloat16 if bf16 else torch.float32
        pipeline = load_pipeline(accelerator, weight_dtype, args)

        # Connect components
        max_pixels = gr.State(1024 * 1024)  # Hidden state for max_pixels
        
        generate_button.click(
            run,
            inputs=[
                instruction, width_input, height_input, scheduler_input,
                num_inference_steps, image_input_1, image_input_2, image_input_3,
                negative_prompt, text_guidance_scale_input, image_guidance_scale_input,
                cfg_range_start, cfg_range_end, num_images_per_prompt,
                max_input_image_side_length, max_pixels, seed_input
            ],
            outputs=[output_image, output_text]
        )

        # Update save_images global when checkbox changes
        def update_save_setting(save_enabled):
            global save_images
            save_images = save_enabled
            return f"✅ Auto-save: {'Enabled' if save_enabled else 'Disabled'}"
        
        save_images_checkbox.change(
            update_save_setting,
            inputs=[save_images_checkbox],
            outputs=[]
        )


        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 10px; color: #666;">
            <p>🔬 Research by VectorSpaceLab | 🍎 Apple Silicon Optimization by <a href="https://github.com/zettai-seigi/OmniGen2" target="_blank" style="color: #666; text-decoration: none;">zettai-seigi</a> | ⚡ Powered by OmniGen2</p>
        </div>
        """)

    demo.launch(share=args.share, server_port=args.port, allowed_paths=[ROOT_DIR])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="OmniGen2/OmniGen2")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true")
    args = parser.parse_args()
    main(args)