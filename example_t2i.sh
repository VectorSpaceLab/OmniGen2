# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

model_path="OmniGen2/OmniGen2"
python inference.py \
--model_path $model_path \
--num_inference_step 50 \
--height 1024 \
--width 1024 \
--text_guidance_scale 4.0 \
--instruction "The sun rises slightly, the dew on the rose petals in the garden is clear, a crystal ladybug is crawling to the dew, the background is the early morning garden, macro lens." \
--output_image_path outputs/output_t2i.png \
--num_images_per_prompt 1