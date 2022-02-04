# This Dockerfile can be used to build a Docker image suitable for tango projects.

# It's important to use the right base image depending on the environment you're going
# to run your project in. If you need different version of CUDA, you'll have to change the base
# image accordingly and also potentially adjust the PyTorch install line below.
# See https://hub.docker.com/r/nvidia/cuda/tags for a list of available base images.
# Generally the "runtime" images are best suited for tango projects.
ARG cuda=11.5.1
FROM nvidia/cuda:${cuda}-cudnn8-runtime-ubuntu20.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /stage/

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment so we don't mess with system packages.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv /opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# Install torch ecosystem before everything else since this is the most time-consuming.
# If you need a different version of CUDA, change this instruction according to the official
# instructions from PyTorch: https://pytorch.org/
RUN pip install --no-cache-dir torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Now copy the `requirements.txt` to `/stage/requirements.txt/` and install them.
# We do this first because it's slow and each of these commands are cached in sequence.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Finally we can copy the tango source code and install tango.
# Alternatively, you could just install "ai2-tango" from PyPI directly by replacing
# these next two lines with `RUN pip install --no-cache-dir ai2-tango[all]`.
COPY . /stage/
RUN pip install --no-cache-dir --no-deps .[all]

WORKDIR /app/
RUN rm -rf /stage/

ENTRYPOINT ["/opt/venv/bin/tango"]
