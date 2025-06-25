#!/bin/bash

set -e

# Source utility functions
source "$(dirname "$0")/utility.sh"

# Main installation logic
install_requirements() {
    local requirements_file="$1"
    local env_type="$2" # "conda" or "venv"

    echo "📦 Installing requirements from $requirements_file..."
    pip install --upgrade pip setuptools wheel build

    if [ "$env_type" = "conda" ]; then
        if grep -i "scipy" "$requirements_file" &> /dev/null; then
            echo "📦 Installing scipy and numpy with conda..."
            conda install -y scipy numpy
        fi
    fi

    while IFS= read -r requirement || [ -n "$requirement" ]; do
        if [ -n "$requirement" ] && [[ ! "$requirement" =~ ^[[:space:]]*# ]]; then
            # Sanitize requirement to remove inline comments, handling VCS URLs
            local sanitized_requirement
            if [[ "$requirement" == *git+* ]]; then
                sanitized_requirement=$(echo "$requirement" | awk '{print $1}')
            else
                sanitized_requirement=$(echo "$requirement" | sed 's/#.*//')
            fi
            
            if [ -z "$sanitized_requirement" ]; then continue; fi

            if [ "$env_type" = "conda" ]; then
                install_conda_requirement "$sanitized_requirement"
            else
                install_venv_requirement "$sanitized_requirement"
            fi
        fi
    done < "$requirements_file"

    touch .requirements_installed
    echo "✅ Requirements installed."
}

# The script can be executed directly, for example:
# ./install_requirements.sh requirements.txt venv
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: $0 <requirements_file> <env_type>"
        echo "env_type can be 'conda' or 'venv'"
        exit 1
    fi
    install_requirements "$1" "$2"
fi
