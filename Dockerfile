# This Dockerfile can be used to build a Docker image suitable for tango projects.

ARG BASE_IMAGE=ubuntu:18.04
ARG PYTHON_VERSION=3.9

FROM ${BASE_IMAGE} as dev-base
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.9
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y "python=${PYTHON_VERSION}" ipython && \
    /opt/conda/bin/conda clean -ya

FROM conda as conda-installs
ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=11.3
ARG INSTALL_CHANNEL=pytorch
RUN /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -y "python=${PYTHON_VERSION}" pytorch torchvision torchtext "cudatoolkit=${CUDA_VERSION}" && \
    /opt/conda/bin/conda clean -ya
WORKDIR /stage
COPY requirements.txt requirements.txt
RUN /opt/conda/bin/pip install --no-cache-dir -r requirements.txt
COPY . .
RUN /opt/conda/bin/pip install --no-cache-dir --no-deps .[all]

FROM ${BASE_IMAGE} as official

LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN --mount=type=cache,id=apt-final,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

COPY --from=conda-installs /opt/conda /opt/conda

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

WORKDIR /workspace

ENTRYPOINT ["/opt/conda/bin/tango"]
