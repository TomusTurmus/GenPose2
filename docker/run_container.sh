#!/usr/bin/env bash
set -euo pipefail

DIR=$(pwd)/../

xhost + >/dev/null
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name genpose2 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$DIR:/workspace/GenPose2" \
  -v /home:/home \
  -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  --ipc=host \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  genpose2:latest bash -c "cd /workspace/GenPose2 && bash"