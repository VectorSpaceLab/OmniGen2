# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)
cd ../

debug=false
RANK=0
MASTER_ADDR=1
MASTER_PORT=29500
WORLD_SIZE=1

# process named arguments
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
            echo "unknown argument: $1"
            shift
            ;;
    esac
done

echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"

num_gpu_cards=$(nvidia-smi -L | wc -l)
num_processes=$(($WORLD_SIZE * $num_gpu_cards))
# num_processes=2

echo "num_gpu_cards: $num_gpu_cards, num_processes: $num_processes"
echo $NCCL_DEBUG_FILE

experiment_name="omnigen2_edit_rl_4machine_editscore7b_avg8"

accelerate launch \
--machine_rank=$RANK \
--main_process_ip=$MASTER_ADDR \
--main_process_port=$MASTER_PORT \
--num_machines=$WORLD_SIZE \
--num_processes=$num_processes \
--use_fsdp \
--fsdp_offload_params false \
--fsdp_sharding_strategy HYBRID_SHARD \
--fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
--fsdp_transformer_layer_cls_to_wrap OmniGen2TransformerBlock \
--fsdp_state_dict_type SHARDED_STATE_DICT \
--fsdp_forward_prefetch false \
--fsdp_use_orig_params True \
--fsdp_cpu_ram_efficient_loading false \
--fsdp_sync_module_states True \
train.py --config options/${experiment_name}.yml