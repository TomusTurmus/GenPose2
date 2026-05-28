# GenPose2 Docker Support

Files added to help dockerize GenPose2 (CUDA 11.8 / Ubuntu 20.04 / Python 3.10.14):

- Dockerfile — image based on `nvidia/cuda:11.8` with Miniconda and PyTorch 2.1.0
- entrypoint.sh — activates the conda environment and runs commands
- docker-compose.yml — compose file with GPU access and useful volume mounts
- run_container.sh — helper launcher with X11 forwarding for OpenCV windows
- environment.yml — optional conda spec (approximate)

Build and run

1. Build image from the repository root so the Dockerfile can copy `requirements.txt`:

```bash
docker build -t genpose2:latest -f docker/Dockerfile .
```

2. Run with docker (GPU must be available via NVIDIA Container Toolkit):

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/GenPose2:/workspace/GenPose2 \
  -v $(pwd)/results:/workspace/GenPose2/results \
  -v $(pwd)/data:/workspace/GenPose2/data \
  genpose2:latest
```

For interactive GUI use inside Docker, first allow local X11 access on the host and then use the helper script:

```bash
xhost +local:
bash docker/run_container.sh
```

Or using docker-compose:

```bash
docker compose up --build
```

Notes and next steps

- The Dockerfile now builds a lightweight CUDA/PyTorch/PyTorch3D base image. The Compose setup mounts your local repo at `/workspace/GenPose2`, so source changes stay outside the image build.
- GUI windows need X11 forwarding. The helper script mirrors FoundationPose's container launch by mounting `/tmp/.X11-unix` and passing `DISPLAY`.
- Some native builds (nvdiffrast, foundationpose components, pointnet2) may still require manual tuning from inside a running container if you need them for a specific experiment.
- GPU access requires `nvidia-docker` / `nvidia-container-toolkit` installed on the host.
