# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.11+pytorch2.6+cu124

experiment_name=ominigenv2_0527_ps1024_bs512_mix


shift=(
# 0.25
# 0.5
# 1.0
# 1.5
3.0
# 2.5
# 3.0
# 4.0
# 5.0
# 6.0
# 7.0
)

ref_guidance_scale=(
# 0.0
1.0
1.2
# 1.3
1.5
# 1.8
2.0
# 2.5
# 3.0
# 4.0
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

data_path=(
# data_options/test_t2i.yml
data_options/test_edit.yml
# data_options/test_subject.yml
)

step=(
# 500
# 1000
# 1500
# 2000
# 1500
# 2000
3000
# 5000
# 8000
)

dynamic_image_size_version=(
# v1 # 原来的，输入和输出都缩放到指定分辨率
# v2 # 输出缩放到指定分辨率，输入只有大于指定分辨率时才缩放
v3 # 输入和输出只有大于指定分辨率时才缩放
)

for((i=0;i<${#data_path[@]};i++))
do
    for((j=0;j<${#ref_guidance_scale[@]};j++))
    do
        for((k=0;k<${#shift[@]};k++))
        do
            for((l=0;l<${#negative_prompt_chat_template_version[@]};l++))
            do
                for((m=0;m<${#step[@]};m++))
                do
                    for((n=0;n<${#dynamic_image_size_version[@]};n++))
                    do
                        accelerate launch --multi_gpu \
                        --main_process_port 40604 \
                        --num_machines 1 \
                        --mixed_precision no \
                        --dynamo_backend no \
                        shitao_test.py --experiment_name ${experiment_name} \
                        --model_path checkpoint-${step[m]}/pytorch_model_fsdp.bin \
                        --data_path ${data_path[i]} \
                        --num_inference_step 50 \
                        --time_shift_scale ${shift[k]} \
                        --height 1024 \
                        --width 1024 \
                        --guidance_scale 4.0 \
                        --ref_guidance_scale ${ref_guidance_scale[j]} \
                        --visualize_input_image \
                        --dynamic_size \
                        --apply_chat_template_negative_prompt \
                        --negative_prompt_chat_template_version ${negative_prompt_chat_template_version[l]} \
                        --negative_prompt "" \
                        --dynamic_image_size_version ${dynamic_image_size_version[n]} \
                        --align_res \
                        --result_dir experiments/${experiment_name}/results_${step[m]}_gs4.0_ref_gs${ref_guidance_scale[j]}_shift${shift[k]}_apply_chat_template_negative_prompt_${negative_prompt_chat_template_version[l]}_neg_prompt_null_dynamic_image_size_version_${dynamic_image_size_version[n]}_align_res
                    done
                done
            done
        done
    done
done