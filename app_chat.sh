# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

# Try to locate conda.sh
CONDA_SH="$(dirname "$(which conda 2>/dev/null)" 2>/dev/null)/../etc/profile.d/conda.sh"

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate py3.11+pytorch2.6+cu124 || echo "⚠️  Warning: Failed to activate the conda environment."
else
    echo "⚠️  Warning: conda.sh not found. Make sure Conda is installed and in your PATH."
fi

RANK=0

# 处理命名参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rank=*)
            RANK="${1#*=}"
            shift
            ;;
        *)
            echo "未知参数: $1"
            shift
            ;;
    esac
done

# Set server_name to "0.0.0.0" if you want to access it from another computer.

CUDA_VISIBLE_DEVICES=${RANK} python app_chat.py \
--port $((7860 + RANK)) \
--server_name "127.0.0.1" \
--share