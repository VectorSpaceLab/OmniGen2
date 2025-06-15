# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

model_path="OmniGen2/OmniGen2"
python inference.py \
--model_path $model_path \
--num_inference_step 50 \
--height 2048 \
--width 1024 \
--text_guidance_scale 3.5 \
--instruction "A curly-haired man in a red shirt is drinking tea." \
--output_image_path outputs/output_t2i.png \
--num_images_per_prompt 1