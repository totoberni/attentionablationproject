#!/bin/bash
set -e

# --- .env File Handling (Relative Path) ---
ENV_FILE="../.env" # Path to .env relative to scripts/

# Source environment variables from .env file
if [ -f "$ENV_FILE" ]; then
  source "$ENV_FILE"
else
  echo "Error: .env file not found at $ENV_FILE. Please create it."
  exit 1
fi

# Check for required environment variables
if [[ -z "${TPU_NAME}" || -z "${TPU_ZONE}" || -z "${PROJECT_ID}" || -z "${QUEUED_RESOURCE_ID}" ]]; then
    echo "Error: Required environment variables (TPU_NAME, TPU_ZONE, PROJECT_ID or QUEUED_RESOURCE_ID) are not set in .env file."
    exit 1
fi

# Stop the TPU VM
echo "Stopping TPU VM '$TPU_NAME'..."
gcloud compute tpus tpu-vm stop $TPU_NAME --zone=$TPU_ZONE --project=$PROJECT_ID --quiet

# Delete the Queued Resource
echo "Deleting Queued Resource '$QUEUED_RESOURCE_ID'..."
gcloud compute tpus queued-resources delete $QUEUED_RESOURCE_ID --zone=$TPU_ZONE --project=$PROJECT_ID --quiet

echo "TPU VM '$TPU_NAME' and Queued Resource '$QUEUED_RESOURCE_ID' stopped and deleted." 