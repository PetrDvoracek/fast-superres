# pull base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# TODO change
ARG USERNAME=pedro
# TODO change
ARG USER_EMAIL=pedro.dvoracek@gmail.com
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=utility,compute,graphics \
    PATH=$PATH:/opt/conda/bin \
    DEBIAN_FRONTEND=noninteractive \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.7

WORKDIR /opt

RUN ln -sf /usr/share/zoneinfo/Europe/Prague /etc/localtime

RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    wget \
    git \
    tmux \
    x11-apps \
    tree \
    # libgl1 \
    build-essential \
    libnvinfer8 \
    libnvinfer-plugin8 \
    libopencv-dev \
    tzdata \
    locales \
    apt-utils &&\
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean \
    && groupadd --gid $USER_GID $USERNAME \
    # sudo start
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # sudo end
    # git configure
    && git config --global user.email "${USER_EMAIL}" \
    # git use main instead master branch name
    && git config --global init.defaultBranch main \
    # git avoid dubious ownership
    && git config --global --add safe.directory /workspaces/${WORKDIR_NAME}

RUN cd /usr/lib/x86_64-linux-gnu && \
    ln -s libnvinfer.so.8 libnvinfer.so.7 && \
    ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7

# install miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -P ./downloads/ && \
    bash ./downloads/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -b -p /opt/conda && \
    rm -rf ./downloads

USER $USERNAME

COPY zsh-in-docker.sh /tmp/zsh-in-docker.sh
RUN sh /tmp/zsh-in-docker.sh \
    -t amuse \
    -p git \
    -p extract \
    -p copybuffer

# install additional libraries
COPY requirements.txt .
RUN pip install -U pip && \
    pip install -r requirements.txt && \
    rm -rf ~/.cache/pip

ENV TZ="Europe/Prague" 

WORKDIR /opt/ml