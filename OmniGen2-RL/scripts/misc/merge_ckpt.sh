# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER
cd ../../

experiment_name=$1
step=$2

model_path=experiments/${experiment_name}/checkpoint-${step}
config_path=experiments/${experiment_name}/${experiment_name}.yml

python scripts/misc/merge_ckpt.py \
--config_path $config_path \
--model_path $model_path/pytorch_model_fsdp.bin \
--save_path $model_path/pytorch_model_fsdp_merged.bin