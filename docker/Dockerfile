# Base Dockerfile for TPU VM
FROM tensorflow/tensorflow:2.14.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    software-properties-common

# Copy requirements and install Python packages
COPY docker/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional TPU-specific packages
RUN pip install cloud-tpu-client==0.10 \
    tensorflow-estimator \
    tensorboard

# Set up working directory
WORKDIR /app
COPY . /app/

# TPU initialization script
COPY docker/scripts/setup_tpu.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/setup_tpu.sh

# Default command
CMD ["/bin/bash"] 