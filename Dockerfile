# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.8
FROM mcr.microsoft.com/windows/servercore:ltsc2019
FROM python:${PYTHON_VERSION}-slim as base
FROM continuumio/miniconda3:latest


# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /ies.pi.internship

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
COPY environment.yml .

RUN --mount=type=cache,target=/opt/conda/pkgs \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=environment.yml,target=environment.yml \
    conda env update -n base -f environment.yml && \
    conda clean --all -y

# Copy the source code into the container.
COPY . .

# # Expose the port that the application listens on.
# EXPOSE 8000

# # Run the application.
# # CMD ["python", "."] --bind=0.0.0.0:8000
