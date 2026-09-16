#!/usr/bin/env bash
set -e

ENV_NAME="ai_lib_actor_critic_mountain_car"

source "$(conda info --base)/etc/profile.d/conda.sh"

cd "$(dirname "$0")/../../.."


conda create \
    -n "$ENV_NAME" \
    -c conda-forge \
    python=3.10 \
    pip \
    gymnasium \
    -y

conda activate "$ENV_NAME"

python -m pip install \
    torch \
    --index-url https://download.pytorch.org/whl/cu130

python -m pip install -e .

python experiments/reinforcement_learning/actor_critic_mountain_car/train.py

conda deactivate
conda remove -n "$ENV_NAME" --all -y