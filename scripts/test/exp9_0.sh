# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.11+pytorch2.6+cu124

experiments=(
ominigenv2_ps512_bs128_ft_new_edit
)

shift=(
# 0.25
# 0.5
# 1.0
# 1.5
2.0
# 2.5
3.0
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

step=18000

for((i=0;i<${#experiments[@]};i++))
do
    for((j=0;j<${#guidance_scale[@]};j++))
    do
        for((k=0;k<${#shift[@]};k++))
        do
            accelerate launch \
            --main_process_port 40900 \
            --num_machines 1 \
            --mixed_precision no \
            --dynamo_backend no \
            shitao_test.py --experiment_name ${experiments[i]} \
            --model_path checkpoint-${step}/pytorch_model_fsdp.bin \
            --data_path data_options/debug_shitao.yml \
            --num_inference_step 50 \
            --time_shift_scale ${shift[k]} \
            --height 512 \
            --width 512 \
            --guidance_scale ${guidance_scale[j]} \
            --visualize_input_image \
            --dynamic_size \
            --visualize_progress \
            --result_dir experiments/${experiments[i]}/results_${step}_gs${guidance_scale[j]}_shift${shift[k]}
        done
    done
done