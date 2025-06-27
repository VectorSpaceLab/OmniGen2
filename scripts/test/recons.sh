# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.11+pytorch2.6+cu124

# 用gemma2-it作为text encoder

experiments=(
shitao_recons_256
)

shift=(
# 0.25
# 0.5
1.0
# 1.5
# 2.0
# 2.5
# 3.0
)

guidance_scale=(
# 1.0
# 1.5
# 2.0
# 2.5
# 3.0
4.0
# 5.0
# 6.0
# 7.0
)

step=28000

for((i=0;i<${#experiments[@]};i++))
do
    for((j=0;j<${#guidance_scale[@]};j++))
    do
        for((k=0;k<${#shift[@]};k++))
        do
            accelerate launch --multi_gpu \
            --main_process_port 40500 \
            --num_machines 1 \
            --mixed_precision no \
            --dynamo_backend no \
            shitao_test.py --experiment_name ${experiments[i]} \
            --model_path checkpoint-${step}/pytorch_model_fsdp.bin \
            --data_path data_options/debug_shitao_index.yml \
            --num_inference_step 50 \
            --height 256 \
            --width 256 \
            --guidance_scale ${guidance_scale[j]} \
            --visualize_input_image \
            --dynamic_size \
            --result_dir experiments/${experiments[i]}/results_${step}_gs${guidance_scale[j]}_shift${shift[k]}
        done
    done
done