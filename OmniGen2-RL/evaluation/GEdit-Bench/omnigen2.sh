# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

experiment_name=omnigen2_edit_rl_4machine_editscore7b_avg8
step=700
RANK=0
WORLD_SIZE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment_name=*)
            experiment_name="${1#*=}"
            shift
            ;;
        --step=*)
            step="${1#*=}"
            shift
            ;;
        --rank=*)
            RANK="${1#*=}"
            shift
            ;;
        --world_size=*)
            WORLD_SIZE="${1#*=}"
            shift
            ;;
        *)
            echo "未知参数: $1"
            shift
            ;;
    esac
done

# 输出配置
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"

global_shift_index=0
total_num_images=606

num_gpus_per_machine=$(python -c "import torch; print(torch.cuda.device_count())")
# Calculate images per machine, rounding up to ensure all data is covered
num_images_per_machine=$(( (total_num_images + WORLD_SIZE - 1) / WORLD_SIZE ))
shift_index=$((RANK * num_images_per_machine))

if [ $((total_num_images - shift_index)) -lt $num_images_per_machine ]; then
    num_images_per_machine=$((total_num_images - shift_index))
fi

# Calculate base number of images per GPU (for first 7 GPUs)
num_images_per_gpu=$(( (num_images_per_machine + num_gpus_per_machine - 1) / num_gpus_per_machine ))

text_guidance_scale=5.0
image_guidance_scale=1.5

for ((i=0; i<num_gpus_per_machine; i++)); do
    if [ $i -lt $((num_gpus_per_machine - 1)) ]; then
        # First 7 GPUs process equal amounts
        start_idx=$((global_shift_index + i * num_images_per_gpu + shift_index))
        end_idx=$((start_idx + num_images_per_gpu))
    else
        # Last GPU processes remaining data
        start_idx=$((global_shift_index + (num_gpus_per_machine - 1) * num_images_per_gpu + shift_index))
        end_idx=$((global_shift_index + shift_index + num_images_per_machine))
    fi
    echo ${start_idx} ${end_idx}

    CUDA_VISIBLE_DEVICES=${i} WORLD_SIZE=1 nohup accelerate launch --num_processes 1 --num_machines 1 \
    evaluation/GEdit-Bench/inference.py \
    --load_from_pipeline \
    --pipeline_path OmniGen2/OmniGen2 \
    --transformer_lora_path experiments/${experiment_name}/checkpoint-${step}/transformer_lora \
    --num_inference_step 50 \
    --height 1024 \
    --width 1024 \
    --text_guidance_scale ${text_guidance_scale} \
    --image_guidance_scale ${image_guidance_scale} \
    --time_shift_base_res 168 \
    --negative_prompt "" \
    --use_ori_neg_prompt_template \
    --scheduler "euler" \
    --result_dir evaluation/GEdit-Bench/results/${experiment_name}/results_step${step}_ts${text_guidance_scale}_ig${image_guidance_scale} \
    --start_index ${start_idx} --end_index ${end_idx} \
    > logs/gedit_${experiment_name}_step${step}_ts${text_guidance_scale}_ig${image_guidance_scale}_${start_idx}_${end_idx}.log 2>&1 &
done