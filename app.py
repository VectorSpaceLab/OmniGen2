import dotenv

dotenv.load_dotenv(override=True)

import gradio as gr

import os
import argparse
import random

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.utils.img_util import resize_image

NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pipeline(accelerator, weight_dtype):
    pipeline = OmniGen2Pipeline.from_pretrained(
        "OmniGen2/OmniGen2",
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
    pipeline = pipeline.to(accelerator.device, dtype=weight_dtype)
    return pipeline


def run(
    instruction,
    width_input,
    height_input,
    num_inference_steps,
    image_input_1,
    image_input_2,
    image_input_3,
    negative_prompt,
    guidance_scale_input,
    img_guidance_scale_input,
    num_images_per_prompt,
    max_input_image_size,
    seed_input,
    progress=gr.Progress(),
):
    input_images = [image_input_1, image_input_2, image_input_3]
    input_images = [img for img in input_images if img is not None]
    if len(input_images) == 0:
        input_images = None

    if input_images is not None:
        # input_images = [crop_arr(x, max_input_image_size, 16) for x in input_images]
        input_images = [
            resize_image(x, max_input_image_size * max_input_image_size, 16)
            for x in input_images
        ]

    if input_images is not None and len(input_images) == 1:
        width_input, height_input = input_images[0].size

    if seed_input == -1:
        seed_input = random.randint(0, 2**16 - 1)
    generator = torch.Generator(device=accelerator.device).manual_seed(seed_input)

    def progress_callback(cur_step, timesteps):
        frac = (cur_step + 1) / float(timesteps)
        progress(frac)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=width_input,
        height=height_input,
        num_inference_steps=num_inference_steps,
        max_sequence_length=1024,
        text_guidance_scale=guidance_scale_input,
        image_guidance_scale=img_guidance_scale_input,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        output_type="pil",
        step_func=progress_callback,
    )

    progress(1.0)

    vis_images = [to_tensor(image) * 2 - 1 for image in results.images]

    # Concatenate input images of different sizes horizontally
    max_height = max(img.shape[-2] for img in vis_images)
    total_width = sum(img.shape[-1] for img in vis_images)
    canvas = torch.zeros((3, max_height, total_width), device=vis_images[0].device)

    current_x = 0
    for i, img in enumerate(vis_images):
        h, w = img.shape[-2:]
        # Place image at the top of canvas
        canvas[:, :h, current_x : current_x + w] = img * 0.5 + 0.5
        current_x += w
    output_image = to_pil_image(canvas)

    if save_images:
        # Save All Generated Images
        from datetime import datetime

        # Create outputs directory if it doesn't exist
        os.makedirs("outputs_gradio", exist_ok=True)
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_path = os.path.join("outputs_gradio", f"{timestamp}.png")
        # Save the image
        output_image.save(output_path)

    return output_image


def get_example():
    case = [
        [
            "A dark wizard conjuring spells in an ancient cave",
            1024,
            1024,
            50,
            None,
            None,
            None,
            NEGATIVE_PROMPT,
            3.5,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "The sun rises slightly, the dew on the rose petals in the garden is clear, a crystal ladybug is crawling to the dew, the background is the early morning garden, macro lens.",
            1024,
            1024,
            50,
            None,
            None,
            None,
            NEGATIVE_PROMPT,
            3.5,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "A snow maiden with pale translucent skin, frosty white lashes, and a soft expression of longing",
            1024,
            1024,
            50,
            None,
            None,
            None,
            NEGATIVE_PROMPT,
            3.5,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Add a fisherman hat to the woman's head",
            1024,
            1024,
            50,
            os.path.join(ROOT_DIR, "example_images/flux5.png"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "replace the sword with a hammer.",
            1024,
            1024,
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/d8f8f44c64106e7715c61b5dfa9d9ca0974314c5d4a4a50418acf7ff373432bb.png",
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Extract the character from the picture and fill the rest of the background with white.",
            # "Transform the sculpture into jade",
            1024,
            1024,
            50,
            os.path.join(
                ROOT_DIR, "example_images/46e79704-c88e-4e68-97b4-b4c40cd29826.png"
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Make he smile",
            1024,
            1024,
            50,
            os.path.join(
                ROOT_DIR, "example_images/vicky-hladynets-C8Ta0gwPbQg-unsplash.jpg"
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Change the background to classroom",
            1024,
            1024,
            50,
            os.path.join(ROOT_DIR, "example_images/ComfyUI_temp_mllvz_00071_.png"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Raise his hand",
            1024,
            1024,
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/289089159-a6d7abc142419e63cab0a566eb38e0fb6acb217b340f054c6172139b316f6596.png",
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Generate a photo of an anime-style figurine placed on a desk. The figurine model should be based on the character photo provided in the attachment, accurately replicating the full-body pose, facial expression, and clothing style of the character in the photo, ensuring the entire figurine is fully presented. The overall design should be exquisite and detailed, soft gradient colors and a delicate texture, leaning towards a Japanese anime style, rich in details, with a realistic quality and beautiful visual appeal.",
            1024,
            1024,
            50,
            os.path.join(ROOT_DIR, "example_images/RAL_0315.JPG"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Change the dress to blue.",
            1024,
            1024,
            50,
            os.path.join(ROOT_DIR, "example_images/1.png"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Remove the cat",
            1024,
            1024,
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/386724677-589d19050d4ea0603aee6831459aede29a24f4d8668c62c049f413db31508a54.png",
            ),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "In a cozy café, the anime figure is sitting in front of a laptop, smiling confidently.",
            1024,
            1024,
            50,
            os.path.join(ROOT_DIR, "example_images/ComfyUI_00254_.png"),
            None,
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Create a wedding figure based on the girl in the first image and the man in the second image. Set the background as a wedding hall, with the man dressed in a suit and the girl in a white wedding dress. Ensure that the original faces remain unchanged and are accurately preserved. The man should adopt a realistic style, whereas the girl should maintain their classic anime style.",
            1024,
            1024,
            50,
            os.path.join(ROOT_DIR, "example_images/1_20241127203215.png"),
            os.path.join(ROOT_DIR, "example_images/000050281.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            3.0,
            1,
            1024,
            0,
        ],
        [
            "Let the girl and the boy get married in the church.",
            1024,
            1024,
            50,
            os.path.join(ROOT_DIR, "example_images/8FtFUxRzXqaguVRGzkHvN.png"),
            os.path.join(ROOT_DIR, "example_images/01194-20240127001056_1024x1536.png"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            3.0,
            1,
            1024,
            0,
        ],
        [
            "Let the man form image1 and the woman from image2 kiss and hug",
            1024,
            1024,
            50,
            os.path.join(ROOT_DIR, "example_images/1280X1280.png"),
            os.path.join(ROOT_DIR, "example_images/000077066.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Please let the person in image 2 hold the toy from the first image in a parking lot.",
            1024,
            1024,
            50,
            os.path.join(ROOT_DIR, "example_images/04.jpg"),
            os.path.join(ROOT_DIR, "example_images/000365954.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Make the girl pray in the second image.",
            1024,
            682,
            50,
            os.path.join(ROOT_DIR, "example_images/000440817.jpg"),
            os.path.join(ROOT_DIR, "example_images/000119733.jpg"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Add the bird from image 1 to the desk in image 2",
            1024,
            682,
            50,
            os.path.join(
                ROOT_DIR,
                "example_images/996e2cf6-daa5-48c4-9ad7-0719af640c17_1748848108409.png",
            ),
            os.path.join(ROOT_DIR, "example_images/00066-10350085.png"),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Replace the apple in the first image with the cat from the second image",
            1024,
            780,
            50,
            os.path.join(ROOT_DIR, "example_images/apple.png"),
            os.path.join(
                ROOT_DIR,
                "example_images/468404374-d52ec1a44aa7e0dc9c2807ce09d303a111c78f34da3da2401b83ce10815ff872.png",
            ),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
        [
            "Replace the woman in the second image with the woman from the first image",
            1024,
            747,
            50,
            os.path.join(
                ROOT_DIR, "example_images/byward-outfitters-B97YFrsITyo-unsplash.jpg"
            ),
            os.path.join(
                ROOT_DIR, "example_images/6652baf6-4096-40ef-a475-425e4c072daf.png"
            ),
            None,
            NEGATIVE_PROMPT,
            5.0,
            2.0,
            1,
            1024,
            0,
        ],
    ]
    return case


def run_for_examples(
    instruction,
    width_input,
    height_input,
    num_inference_steps,
    image_input_1,
    image_input_2,
    image_input_3,
    negative_prompt,
    text_guidance_scale_input,
    image_guidance_scale_input,
    num_images_per_prompt,
    max_input_image_size,
    seed_input,
):
    return run(
        instruction,
        width_input,
        height_input,
        num_inference_steps,
        image_input_1,
        image_input_2,
        image_input_3,
        negative_prompt,
        text_guidance_scale_input,
        image_guidance_scale_input,
        num_images_per_prompt,
        max_input_image_size,
        seed_input,
    )


description = """
The model mainly supports English, with slight support for Chinese.
Increase the `image_guidance_scale` if you need more consistency with the reference image. For image editing task, we recommend to set it between 1.3 and 2.0; for in-context generateion task, a higher image_guidance_scale will maintian more details in input images, and we recommend to set it between 2.0 and 3.0.

"""

article = """
citation to be added
"""

# Gradio
with gr.Blocks() as demo:
    gr.Markdown(
        "# OmniGen2: Unified Image Generation [paper](https://arxiv.org/abs/2409.11340) [code](https://github.com/VectorSpaceLab/OmniGen2)"
    )
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            # text prompt
            instruction = gr.Textbox(
                label='Enter your prompt. Use "first/second image" or “第一张图/第二张图” as reference.',
                placeholder="Type your prompt here...",
            )

            with gr.Row(equal_height=True):
                # input images
                image_input_1 = gr.Image(label="First Image", type="pil")
                image_input_2 = gr.Image(label="Second Image", type="pil")
                image_input_3 = gr.Image(label="Third Image", type="pil")

            generate_button = gr.Button("Generate Image")

            negative_prompt = gr.Textbox(
                label="Enter your negative prompt",
                placeholder="Type your negative prompt here...",
                value=NEGATIVE_PROMPT,
            )

            # slider
            height_input = gr.Slider(
                label="Height", minimum=256, maximum=1024, value=1024, step=128
            )
            width_input = gr.Slider(
                label="Width", minimum=256, maximum=1024, value=1024, step=128
            )

            text_guidance_scale_input = gr.Slider(
                label="Text Guidance Scale",
                minimum=1.0,
                maximum=8.0,
                value=5.0,
                step=0.1,
            )

            image_guidance_scale_input = gr.Slider(
                label="Image Guidance Scale",
                minimum=1.0,
                maximum=3.0,
                value=2.0,
                step=0.1,
            )

            num_inference_steps = gr.Slider(
                label="Inference Steps", minimum=20, maximum=100, value=50, step=1
            )

            num_images_per_prompt = gr.Slider(
                label="Number of images per prompt",
                minimum=1,
                maximum=4,
                value=1,
                step=1,
            )

            seed_input = gr.Slider(
                label="Seed", minimum=-1, maximum=2147483647, value=0, step=1
            )
            max_input_image_size = gr.Slider(
                label="max_input_image_size",
                minimum=256,
                maximum=1024,
                value=1024,
                step=256,
            )

        with gr.Column():
            with gr.Column():
                # output image
                output_image = gr.Image(label="Output Image")
                save_images = gr.Checkbox(label="Save generated images", value=False)

    bf16 = True
    accelerator = Accelerator(mixed_precision="bf16" if bf16 else "no")
    weight_dtype = torch.bfloat16 if bf16 else torch.float32

    pipeline = load_pipeline(accelerator, weight_dtype)

    # click
    generate_button.click(
        run,
        inputs=[
            instruction,
            width_input,
            height_input,
            num_inference_steps,
            image_input_1,
            image_input_2,
            image_input_3,
            negative_prompt,
            text_guidance_scale_input,
            image_guidance_scale_input,
            num_images_per_prompt,
            max_input_image_size,
            seed_input,
        ],
        outputs=output_image,
    )

    gr.Examples(
        examples=get_example(),
        fn=run_for_examples,
        inputs=[
            instruction,
            width_input,
            height_input,
            num_inference_steps,
            image_input_1,
            image_input_2,
            image_input_3,
            negative_prompt,
            text_guidance_scale_input,
            image_guidance_scale_input,
            num_images_per_prompt,
            max_input_image_size,
            seed_input,
        ],
        outputs=output_image,
    )

    gr.Markdown(article)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OmniGen")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app")
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to use for the Gradio app"
    )
    args = parser.parse_args()

    # launch
    demo.launch(share=args.share, server_port=args.port)