# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

model_path="OmniGen2/OmniGen2"
python inference_chat.py \
--model_path $model_path \
--instruction "Please describe this image briefly." \
--input_image_path example_images/02.jpg