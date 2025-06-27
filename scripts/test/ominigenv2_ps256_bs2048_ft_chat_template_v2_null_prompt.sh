# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.11+pytorch2.6+cu124

experiment_name=ominigenv2_ps256_bs2048_ft

test_data=(
data_options/test_t2i_en.yml
data_options/test_t2i_zh.yml
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

step=12000

for((i=0;i<${#test_data[@]};i++))
do
    for((j=0;j<${#guidance_scale[@]};j++))
    do
        for((k=0;k<${#shift[@]};k++))
        do
            accelerate launch --multi_gpu \
            --main_process_port 40604 \
            --num_machines 1 \
            --mixed_precision no \
            --dynamo_backend no \
            shitao_test.py --experiment_name ${experiment_name} \
            --model_path checkpoint-${step}/pytorch_model_fsdp.bin \
            --data_path ${test_data[i]} \
            --num_inference_step 50 \
            --time_shift_scale ${shift[k]} \
            --height 256 \
            --width 256 \
            --guidance_scale ${guidance_scale[j]} \
            --visualize_input_image \
            --dynamic_size \
            --apply_chat_template_negative_prompt \
            --negative_prompt_chat_template_version v2 \
            --negative_prompt "" \
            --result_dir experiments/${experiment_name}/results_${step}_gs${guidance_scale[j]}_shift${shift[k]}_apply_chat_template_negative_prompt_v2_null_neg_prompt
        done
    done
done