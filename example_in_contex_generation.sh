# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

model_path="OmniGen2/OmniGen2"
python inference.py \
--model_path $model_path \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 5.0 \
--image_guidance_scale 2.0 \
--instruction "Please let the person in image 2 hold the toy from the first image in a parking lot." \
--input_image_path example_images/04.jpg example_images/000365954.jpg \
--output_image_path outputs/output_in_context_generation.png \
--num_images_per_prompt 1