# !/bin/bash

# The following code will use all available visible GPUs.
accelerate launch --multi_gpu \
--main_process_port 41000 \
--num_machines 1 \
--mixed_precision no \
--dynamo_backend no \
inference.py --input_dir ./datasets/RAWSR/val_in --output_dir ./results/RAWSR/val_out --model_dir ./pretrained_models/LiteRAWFormer \
--self_ensemble self --dtype fp16

# accelerate launch --multi_gpu \
# --main_process_port 41000 \
# --num_machines 1 \
# --mixed_precision no \
# --dynamo_backend no \
# inference.py --input_dir ./datasets/RAWSR/test_in --output_dir ./results/RAWSR/test_out --model_dir ./pretrained_models/NTIRE2025_finnal \
# --self_ensemble self --dtype fp16
