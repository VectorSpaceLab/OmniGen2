# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

root_dir=$SHELL_FOLDER

config_path=${root_dir}/server_configs/editscore_7B.yml

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config_path=*)
            config_path="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            shift
            ;;
    esac
done

hosts_string=$(python ${root_dir}/scripts/misc/load_host.py --config_path=${config_path})
hosts=(${hosts_string})

SCRIPT="pkill python"

for i in "${!hosts[@]}"; do
    echo "$i ${hosts[$i]}"
    ssh -o StrictHostKeyChecking=no "${hosts[$i]}" "$SCRIPT" &
done

wait
echo "All tasks in remote machines killed"