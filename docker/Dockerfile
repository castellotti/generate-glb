# syntax=docker/dockerfile:1

# Base image with common dependencies
FROM --platform=$TARGETPLATFORM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY docker/requirements/ /app/requirements/

# Copy source code
COPY src/ /app/src/

# Create cache directory for models
RUN mkdir -p /root/.cache/huggingface

# CPU target
FROM base as cpu
RUN pip install --no-cache-dir \
    -r requirements/base.txt \
    -r requirements/cpu.txt \
    -r requirements/server.txt

# Default command (can be overridden)
CMD ["python", "/app/src/generate.py"]

# CUDA target using NVIDIA base image (amd64 only)
FROM --platform=linux/amd64 nvidia/cuda:12.8.0-runtime-ubuntu24.04 as cuda

# Set up Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements files
COPY docker/requirements/ /app/requirements/

# Copy source code
COPY src/ /app/src/

# Create cache directory for models
RUN mkdir -p /root/.cache/huggingface

# Install CUDA-enabled requirements
RUN pip install --no-cache-dir \
    -r requirements/base.txt \
    -r requirements/cuda.txt \
    -r requirements/server.txt

# Default command (can be overridden)
CMD ["python", "/app/src/generate.py"]

# Default to CPU target if not specified
FROM cpu as default