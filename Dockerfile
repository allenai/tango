# This Dockerfile can be used to build a Docker image suitable for tango projects.

ARG BASE_IMAGE=ghcr.io/allenai/pytorch:1.13.0-cuda11.6-python3.9
FROM ${BASE_IMAGE}

WORKDIR /stage

COPY . .
RUN /opt/conda/bin/pip install --no-cache-dir .[all]

WORKDIR /workspace

RUN rm -rf /stage/

ENTRYPOINT ["/opt/conda/bin/tango"]
