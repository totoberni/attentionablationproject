#!/bin/bash
set -e

# Source environment variables from .env file (in the parent directory)
ENV_FILE="../.env"
if [ -f "$ENV_FILE" ]; then
  source "$ENV_FILE"
else
  echo "Error: .env file not found at $ENV_FILE."
  exit 1
fi

# --- Check for Required Variables ---
if [[ -z "${TPU_NAME}" || -z "${TPU_ZONE}" || -z "${PROJECT_ID}" || -z "${BUCKET_NAME}" ]]; then
  echo "Error: Required environment variables (TPU_NAME, TPU_ZONE, PROJECT_ID, BUCKET_NAME) are not set."
  exit 1
fi

# --- Verify GCS Bucket Access ---
if ! gsutil ls "gs://${BUCKET_NAME}" &> /dev/null; then
  echo "Error: Cannot access GCS bucket gs://${BUCKET_NAME}"
  exit 1
fi

# --- Construct the SSH Command ---
PREPROCESS_COMMAND="python3 /app/src/data/preprocess.py \
  --input-dir=gs://${BUCKET_NAME}/raw_data \
  --output-dir=gs://${BUCKET_NAME}/processed_data \
  --config-dir=/app/config \
  --tokenizer-path=gs://${BUCKET_NAME}/tokenizers \
  --cache-dir=gs://${BUCKET_NAME}/cache"

# The full gcloud command
echo "Starting preprocessing on TPU VM..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$TPU_ZONE" \
  --project="$PROJECT_ID" \
  --worker=all \
  --command="$PREPROCESS_COMMAND"

echo "Preprocessing completed. Data is available in gs://${BUCKET_NAME}/processed_data" 