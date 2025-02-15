# syntax=docker/dockerfile:1

# Base image with common dependencies
FROM python:3.10-slim as base

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
RUN pip install --no-cache-dir -r requirements/base.txt -r requirements/cpu.txt
ENTRYPOINT ["python", "src/generate.py"]

# CUDA target using NVIDIA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as cuda

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
RUN pip install --no-cache-dir -r requirements/base.txt -r requirements/cuda.txt

ENTRYPOINT ["python", "src/generate.py"]

# Default to CPU target if not specified
FROM cpu as default