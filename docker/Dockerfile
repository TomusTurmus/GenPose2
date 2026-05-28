FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=/opt/conda/bin:$CUDA_HOME/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates git curl \
    gcc-10 g++-10 cmake openexr libopenexr-dev libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender1 libgtk2.0-dev qt5-default ffmpeg procps \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -a -y

RUN $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment with the required Python
RUN $CONDA_DIR/bin/conda create -n genpose2 python=3.10.14 -y && \
    $CONDA_DIR/bin/conda clean -afy

# Ensure conda env binaries are on PATH
ENV PATH=$CONDA_DIR/envs/genpose2/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST=8.9

# Upgrade pip and install build helpers inside the conda environment
RUN conda run -n genpose2 python -m pip install --no-cache-dir --upgrade pip "setuptools<70" wheel ninja

# Install PyTorch for CUDA 11.8
RUN conda run -n genpose2 python -m pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 && \
    conda run -n genpose2 python -m pip install --force-reinstall "setuptools<70"

# Keep NumPy on the 1.x ABI line for pytorch3d and other compiled extensions
RUN conda run -n genpose2 python -m pip install --no-cache-dir "numpy<2"

# Install the project Python requirements into the conda environment so they
# are available in the final container image.
COPY requirements.txt /tmp/requirements.txt
RUN conda run -n genpose2 python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Install the camera/inference extras used by the real-time camera pipeline.
COPY requirements_camera.txt /tmp/requirements_camera.txt
RUN conda run -n genpose2 python -m pip install --no-cache-dir -r /tmp/requirements_camera.txt

# GenPose2 imports cutoop throughout training, evaluation, and inference.
RUN conda run -n genpose2 python -m pip install --no-cache-dir cutoop

# Compile the local PointNet++ CUDA extension required by the pose networks.
COPY networks/pts_encoder/pointnet2_utils/pointnet2 /tmp/pointnet2
RUN cd /tmp/pointnet2 && \
    conda run -n genpose2 python -m pip install --no-cache-dir --no-build-isolation .

# PyTorch3D is required by GenPose. Use the prebuilt Linux wheel for Python 3.10,
# CUDA 11.8, and PyTorch 2.1.0 to avoid the costly source build.
RUN conda run -n genpose2 python -m pip install --no-cache-dir iopath fvcore && \
        conda run -n genpose2 python -m pip install --no-cache-dir \
            pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html


WORKDIR /workspace

WORKDIR /workspace/GenPose2

ENV CUDA_HOME=/usr/local/cuda-11.8

# The compose file mounts the repository at /workspace/GenPose2, so keep the
# image focused on the shared CUDA/PyTorch/PyTorch3D runtime.
RUN mkdir -p /workspace/GenPose2

VOLUME ["/workspace/GenPose2/results", "/workspace/GenPose2/data", "/workspace/GenPose2/checkpoints"]

COPY /docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

CMD ["/bin/bash"]
