# sudo docker build --pull -t mv3dengine_22.04:cu118 -f docker/Dockerfile_base .
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set compute capability for nerfacc and tiny-cuda-nn
# See https://developer.nvidia.com/cuda-gpus and limit number to speed-up build
ENV TORCH_CUDA_ARCH_LIST="5.0 5.2 6.0 6.1 7.0 7.5 8.0 8.6 8.6+PTX 8.9 9.0+PTX"
ENV TCNN_CUDA_ARCHITECTURES=90;89;86;80;75;70;61;60;52;50

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/home/${USER_NAME}/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}

# apt install by root user
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda
RUN curl -so /miniconda.sh https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/repo/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

RUN  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/  \
&& conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
&& conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
&& conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ \
&& conda config --set show_channel_urls yes


# Create a Python 3.10 environment
RUN conda install -y conda-build \
 && conda create -y --name py310 python=3.10 \
 && conda clean -ya

ENV CONDA_DEFAULT_ENV=py310
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false


RUN pip install --upgrade pip setuptools ninja
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# # Install nerfacc and tiny-cuda-nn before installing requirements.txt
# # because these two installations are time consuming and error prone
RUN pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

RUN pip install git+https://github.com/NVlabs/nvdiffrast.git
RUN pip install git+https://github.com/ashawkey/envlight.git
RUN pip install git+https://github.com/openai/CLIP.git

COPY requirements.txt /tmp
RUN cd /tmp && pip install -r requirements.txt
# WORKDIR /home/${USER_NAME}/threestudio
WORKDIR /home

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install pymeshlab ipywidgets open_clip_torch==2.7.0 transformers==4.28.1