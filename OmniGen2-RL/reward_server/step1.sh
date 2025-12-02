# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

# uncomment this if you are using conda
# source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
# conda activate editscore

root_dir=$SHELL_FOLDER
machine_id=0
model_name=editscore_7B
config_path=${root_dir}/server_configs/editscore_7B.yml

# process named parameters
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config_path=*)
            config_path="${1#*=}"
            shift
            ;;
        --machine_id=*)
            machine_id="${1#*=}"
            shift
            ;;
        --model_name=*)
            model_name="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            shift
            ;;
    esac
done

export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_LOG_BATCHSIZE_INTERVAL=60

VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3 python start_multi_servers.py reward_server --config_path ${config_path} --model_name ${model_name} \
--machine_id ${machine_id}