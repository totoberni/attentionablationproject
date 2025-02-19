#!/bin/bash
set -e

# Initialize TPU runtime
ctpu up --name=${TPU_NAME} --project=${PROJECT_ID} --zone=${ZONE} --tpu-size=${TPU_SIZE}

# Configure PyTorch/XLA for optimal performance
export XLA_USE_BF16=1
export TPU_NUM_DEVICES=8

# Set up storage bucket for checkpoints
gsutil -m cp -r ${LOCAL_CHECKPOINT_DIR} gs://${BUCKET_NAME}/checkpoints/

echo "TPU environment initialized successfully!" 