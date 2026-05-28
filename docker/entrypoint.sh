#!/bin/bash
set -e

# Activate conda environment created at build time and forward commands
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate genpose2 || true
fi

if [ -d "/workspace/GenPose2" ] && [ -f "/workspace/GenPose2/runners/infer.py" ]; then
  cd /workspace/GenPose2
elif [ -f "/workspace/runners/infer.py" ]; then
  cd /workspace
fi

if [ "$#" -eq 0 ]; then
  exec /bin/bash
else
  exec "$@"
fi
