FROM gcr.io/cloud-tpus/prebuilt-images/tensorflow/tf-2.15-pjrt-base:latest

# Set non-interactive installation for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        git \
        wget \
        curl \
        software-properties-common \
        && \
    rm -rf /var/lib/apt/lists/*

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

# Install TensorFlow dependencies and other useful packages
RUN pip install --no-cache-dir -U \
    google-cloud-storage \
    wandb

# Set environment variables
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV PYTHONPATH="/usr/share/tpu/models:${PYTHONPATH}"
ENV NEXT_PLUGGABLE_DEVICE_USE_C_API=true
ENV TF_PLUGGABLE_DEVICE_LIBRARY_PATH=/lib/libtpu.so

# Copy project files
WORKDIR /app
COPY . /app/ 