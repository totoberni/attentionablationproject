#!/bin/bash

# Load environment variables
ENV_FILE="../../.env"  # Path from docker/scripts to root .env
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

echo "Starting preprocessing..."

# Define preprocessing command with absolute paths
PREPROCESS_COMMAND="python3 /app/src/data/preprocess.py \
    --input-dir=gs://${BUCKET_NAME}/raw_data \
    --output-dir=gs://${BUCKET_NAME}/processed_data \
    --config-dir=/app/config \
    --tokenizer-path=gs://${BUCKET_NAME}/tokenizers \
    --cache-dir=gs://${BUCKET_NAME}/cache"

# Run preprocessing on TPU VM
echo "Running preprocessing on TPU VM..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --command="$PREPROCESS_COMMAND"
handle_error $? "Preprocessing failed"

echo "Preprocessing completed successfully!" 