#!/bin/bash

# Utility functions for direnv_smart_setup

# Function to install packages with retries and fallbacks (for Conda)
install_conda_requirement() {
    local requirement="$1"
    local max_retries=3
    local retry_count=0

    # Skip scipy and numpy as we installed them with conda create
    if [[ "$requirement" == scipy* ]] || [[ "$requirement" == numpy* ]]; then
        return 0
    fi

    while [ $retry_count -lt $max_retries ]; do
        echo "Installing: \"$requirement\""
        
        case "$requirement" in
            "sentencepiece"*)
                if conda install -y -c conda-forge "$requirement"; then return 0; fi
                ;;
            "av"*)
                if conda install -y -c conda-forge "av"; then return 0; fi
                ;;
        esac

        if pip install "$requirement"; then
            return 0
        fi

        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo "⚠️  Retrying installation of \"$requirement\" (attempt $((retry_count + 1))/$max_retries)"
            sleep 2
        else
            echo "⚠️  Failed to install: \"$requirement\" after $max_retries attempts"
            return 1
        fi
    done
}

# Function to install packages with retries (for venv)
install_venv_requirement() {
    local requirement="$1"
    local max_retries=3
    local retry_count=0
    while [ $retry_count -lt $max_retries ]; do
        echo "Installing: \"$requirement\""
        if pip install "$requirement"; then
            return 0
        fi
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            echo "⚠️  Retrying installation of \"$requirement\" (attempt $((retry_count + 1))/$max_retries)"
            sleep 2
        else
            echo "⚠️  Failed to install: \"$requirement\" after $max_retries attempts"
            return 1
        fi
    done
}
