# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

RANK=0
MASTER_ADDR=1
MASTER_PORT=29500
WORLD_SIZE=1

# 处理命名参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank=*)
            RANK="${1#*=}"
            shift
            ;;
        --master_addr=*)
            MASTER_ADDR="${1#*=}"
            shift
            ;;
        --master_port=*)
            MASTER_PORT="${1#*=}"
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
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
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

guidance_scale=4.0

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
    evaluation/GEdit-Bench/inference_qwen_image.py \
    --pipeline_path Qwen/Qwen-Image-Edit \
    --num_inference_step 50 \
    --height 1024 \
    --width 1024 \
    --guidance_scale ${guidance_scale} \
    --negative_prompt " " \
    --result_dir evaluation/GEdit-Bench/results/Qwen-Image-Edit/results_gs${guidance_scale}_16samples \
    --start_index ${start_idx} --end_index ${end_idx} \
    --num_samples 16 \
    > logs/gedit_Qwen-Image-Edit_gs${guidance_scale}_16samples_${start_idx}_${end_idx}.log 2>&1 &
done