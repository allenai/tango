# This Dockerfile can be used to build a Docker image suitable for tango projects.

ARG BASE_IMAGE=ghcr.io/allenai/pytorch:1.12.0-cuda11.3-python3.9
FROM ${BASE_IMAGE}

WORKDIR /stage

COPY requirements.txt requirements.txt
RUN /opt/conda/bin/pip install --no-cache-dir -r requirements.txt

COPY . .
RUN /opt/conda/bin/pip install --no-cache-dir --no-deps .[all]

WORKDIR /workspace

RUN rm -rf /stage/

ENTRYPOINT ["/opt/conda/bin/tango"]
