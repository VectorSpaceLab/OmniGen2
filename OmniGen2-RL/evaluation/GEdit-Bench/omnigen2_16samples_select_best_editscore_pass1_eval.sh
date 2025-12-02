# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

best=(
1
2
4
8
16
)

for b in "${best[@]}"
do
    accelerate launch --num_processes 1 evaluation/GEdit-Bench/test_gedit_score.py \
    --result_dir evaluation/GEdit-Bench/results/OmniGen2/results_ts${text_guidance_scale}_ig${image_guidance_scale}_16samples_pass1_best${b} \
    --backbone gpt-4.1 \
    --openai_url https://api.openai.com/v1/chat/completions \
    --max_workers 30 \
    --key PUT-YOUR-KEY-HERE

    python evaluation/GEdit-Bench/calculate_statistics.py \
    --result_dir evaluation/GEdit-Bench/results/OmniGen2/results_ts${text_guidance_scale}_ig${image_guidance_scale}_16samples_pass1_best${b}/viescore_gpt-4.1 \
    --language en
done