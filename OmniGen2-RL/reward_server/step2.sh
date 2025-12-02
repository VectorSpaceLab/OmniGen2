# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

# uncomment this if you are using conda
# source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
# conda activate editscore

machine_id=0
config_path=examples/OmniGen2-RL/reward_server/server_configs/editscore_7B.yml
model_name=editscore_7B

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

python reward_proxy.py --config_path ${config_path} \
>logs/reward_proxy_${model_name}_machine${machine_id}.log 2>&1