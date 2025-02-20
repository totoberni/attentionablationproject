FROM gcr.io/cloud-tpus/prebuilt-images/tensorflow/tf-2.15-pjrt-base:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY docker/requirements.txt /app/docker/requirements.txt
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -r docker/requirements.txt

# Install spacy model
RUN python -m spacy download en_core_web_lg

# Copy project files
COPY . /app/

# Set permissions for scripts
RUN chmod +x /app/docker/scripts/*.sh

# Create volume for data persistence
VOLUME /app/data

# Set working directory
WORKDIR /app

# Default command
CMD ["bash"] 