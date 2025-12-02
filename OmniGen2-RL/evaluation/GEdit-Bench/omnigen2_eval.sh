# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.12+pytorch2.7.1+cu126

experiment_name=omnigen2_edit_rl_4machine_editscore7b_avg8
step=700

# 处理命名参数
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
        *)
            echo "未知参数: $1"
            shift
            ;;
    esac
done

text_guidance_scale=5.0
image_guidance_scale=1.5

accelerate launch --num_processes 1 evaluation/GEdit-Bench/test_gedit_score.py \
--result_dir evaluation/GEdit-Bench/results/${experiment_name}/results_step${step}_ts${text_guidance_scale}_ig${image_guidance_scale} \
--backbone gpt-4.1 \
--openai_url https://api.openai.com/v1/chat/completions \
--max_workers 30 \
--key PUT-YOUR-KEY-HERE

python evaluation/GEdit-Bench/calculate_statistics.py \
--result_dir evaluation/GEdit-Bench/results/${experiment_name}/results_step${step}_ts${text_guidance_scale}_ig${image_guidance_scale}/viescore_gpt-4.1 \
--language en