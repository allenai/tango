# This Dockerfile can be used to build a Docker image suitable for tango projects.

ARG BASE_IMAGE=ghcr.io/allenai/pytorch:2.0.0-cuda11.7-python3.10
FROM ${BASE_IMAGE}

WORKDIR /stage

COPY . .
RUN /opt/conda/bin/pip install --no-cache-dir .[all]

WORKDIR /workspace

RUN rm -rf /stage/

ENTRYPOINT ["/opt/conda/bin/tango"]
