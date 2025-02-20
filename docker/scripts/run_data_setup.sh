#!/bin/bash

# Load environment variables
ENV_FILE="../../.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi
source "$ENV_FILE"

# Function for error handling
handle_error() {
    local exit_code=$1
    local error_message=$2
    if [ $exit_code -ne 0 ]; then
        echo "Error: $error_message"
        exit $exit_code
    fi
}

echo "Starting data setup process..."

# Check and create GCS directories if needed
if ! gsutil ls "gs://${BUCKET_NAME}/raw_data" &> /dev/null; then
    echo "Creating GCS directory: gs://${BUCKET_NAME}/raw_data"
    gsutil mb -p "${PROJECT_ID}" "gs://${BUCKET_NAME}/raw_data" || true
else
    echo "GCS directory gs://${BUCKET_NAME}/raw_data exists."
fi

# Run preprocessing first
echo "Running preprocessing..."
"$(dirname "$0")/run_preprocess.sh"
handle_error $? "Preprocessing failed"

# Define the data setup command
DATA_SETUP_COMMAND="python3 /app/src/data/core/setup.py \
    --dataset_name=gutenberg2-dpo \
    --split=train \
    --config_path=/app/config/data_config.yaml \
    --batch_size=32 \
    --output_dir=gs://${BUCKET_NAME}/raw_data"

# Run data setup on TPU VM
echo "Running data setup on TPU VM..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --command="$DATA_SETUP_COMMAND"
handle_error $? "Data setup failed"

echo "Data setup completed successfully!" 