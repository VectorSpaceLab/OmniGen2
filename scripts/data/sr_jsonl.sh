# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.11+pytorch2.6+cu124

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu \
# --main_process_port 40101 \
# --num_machines 1 \
# --mixed_precision no \
# --dynamo_backend no \
# scripts/generate_gemini_prompt.py --task subject --start_index 5150000 --num_images 150000 --save_frequency 20 --data_source real

rank=${1:-0}  # 如果 $1 为空，则使用 "default1"
world_size=${2:-1}  # 如果 $2 为空，则使用 "default2"

global_shift_index=0

total_num_images=100

# Calculate images per machine, rounding up to ensure all data is covered
num_images_per_machine=$(( (total_num_images + world_size - 1) / world_size ))
shift_index=$((rank * num_images_per_machine))

if [ $((total_num_images - shift_index)) -lt $num_images_per_machine ]; then
    num_images_per_machine=$((total_num_images - shift_index))
fi

# Calculate base number of images per GPU (for first 7 GPUs)
num_images_per_gpu=$(( (num_images_per_machine + 7) / 8 ))

for i in {0..7}; do
    if [ $i -lt 7 ]; then
        # First 1 GPUs process equal amounts
        start_idx=$((global_shift_index + i * num_images_per_gpu + shift_index))
        end_idx=$((start_idx + num_images_per_gpu))
    else
        # Last GPU processes remaining data
        start_idx=$((global_shift_index + 7 * num_images_per_gpu + shift_index))
        end_idx=$((global_shift_index + shift_index + num_images_per_machine))
    fi
    echo $i, $start_idx, $end_idx
    CUDA_VISIBLE_DEVICES=${i} nohup python \
    scripts/data/sr_jsonl.py \
    --num_load_images $((global_shift_index + total_num_images + shift_index)) \
    --start_index ${start_idx} --end_index ${end_idx} \
    --save_frequency 20 \
    --data_source real \
    --save_file /share_2/luoxin/datasets/image_enhancement/real/${start_idx}_${end_idx}.jsonl \
    --prompt_version v3 > logs/sr_jsonl_${i}.log 2>&1 &
done

# sleep infinity