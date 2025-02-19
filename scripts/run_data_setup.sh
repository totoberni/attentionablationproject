#!/bin/bash
set -e

# Source environment variables from .env file
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

# --- Create GCS Bucket Structure ---
echo "Setting up GCS bucket structure..."
for dir in raw_data processed_data tokenizers cache; do
    if ! gsutil ls "gs://${BUCKET_NAME}/${dir}" &> /dev/null; then
        echo "Creating gs://${BUCKET_NAME}/${dir}"
        gsutil mb -p "${PROJECT_ID}" "gs://${BUCKET_NAME}/${dir}" || true
    fi
done

# --- Construct the Data Setup Command ---
DATA_SETUP_COMMAND="python3 /app/src/data/setup.py \
  --dataset_name=gutenberg2-dpo \
  --split=train \
  --config_path=/app/config/data_config.yaml \
  --batch_size=32 \
  --output_dir=gs://${BUCKET_NAME}/raw_data"

# Execute data setup on TPU VM
echo "Starting data setup on TPU VM..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$TPU_ZONE" \
  --project="$PROJECT_ID" \
  --worker=all \
  --command="$DATA_SETUP_COMMAND"

# --- Run Preprocessing ---
echo "Starting preprocessing pipeline..."
./run_preprocess.sh

echo "Data setup and preprocessing completed successfully." 