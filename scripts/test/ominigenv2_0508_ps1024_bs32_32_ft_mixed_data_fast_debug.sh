# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.11+pytorch2.6+cu124

experiments=(
ominigenv2_0507_ps512_bs1024_ft_mixed_data
)

shift=(
# 0.25
# 0.5
# 1.0
# 1.5
# 2.0
# 2.5
3.0
# 4.0
# 5.0
# 6.0
# 7.0
)

guidance_scale=(
# 0.0
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

negative_prompt_chat_template_version=(
# v1
v2
# v3
# v4
# v5
# v6
)

step=1000

for((i=0;i<${#experiments[@]};i++))
do
    for((j=0;j<${#guidance_scale[@]};j++))
    do
        for((k=0;k<${#shift[@]};k++))
        do
            for((l=0;l<${#negative_prompt_chat_template_version[@]};l++))
            do
                accelerate launch --multi_gpu \
                --main_process_port 40604 \
                --num_machines 1 \
                --mixed_precision no \
                --dynamo_backend no \
                shitao_test.py --experiment_name ${experiments[i]} \
                --model_path checkpoint-${step}/pytorch_model_fsdp.bin \
                --data_path data_options/test.yml \
                --num_inference_step 50 \
                --time_shift_scale ${shift[k]} \
                --height 1024 \
                --width 1024 \
                --guidance_scale ${guidance_scale[j]} \
                --visualize_input_image \
                --dynamic_size \
                --apply_chat_template_negative_prompt \
                --negative_prompt_chat_template_version ${negative_prompt_chat_template_version[l]} \
                --negative_prompt "" \
                --result_dir experiments/${experiments[i]}/results_${step}_gs${guidance_scale[j]}_shift${shift[k]}_apply_chat_template_negative_prompt_${negative_prompt_chat_template_version[l]}_neg_prompt_null
            done
        done
    done
done