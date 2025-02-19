#!/bin/bash
set -e

# --- 1. Google Cloud Authentication and Configuration ---

if ! gcloud config list &> /dev/null; then
  echo "gcloud is not initialized. Please run 'gcloud init' and follow the prompts."
  gcloud init
fi

if ! gcloud auth list &> /dev/null; then
    echo "Authenticating with Google Cloud..."
    gcloud auth login
fi

# --- .env File Handling (Relative Path) ---
ENV_FILE="../.env"  # Path to .env relative to scripts/

if [ -f "$ENV_FILE" ]; then
  source "$ENV_FILE"
else
  echo "Error: .env file not found at $ENV_FILE. Please create it."
  exit 1
fi

gcloud auth configure-docker -q

# --- 2. Create TPU VM (Queued Resource) ---

if [[ -z "${TPU_NAME}" || -z "${TPU_ZONE}" || -z "${ACCELERATOR_TYPE}" || -z "${RUNTIME_VERSION}" || -z "${QUEUED_RESOURCE_ID}" ]]; then
  echo "Error: Required environment variables are not set in .env file."
  exit 1
fi

if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project)
    if [ -z "$PROJECT_ID" ]; then
        echo "Error: Could not determine PROJECT_ID."
        exit 1
    fi
    echo "PROJECT_ID set to: $PROJECT_ID"
fi

QUOTA_TYPE=${QUOTA_TYPE:-"on-demand"}
if [[ "$QUOTA_TYPE" != "reserved" && "$QUOTA_TYPE" != "best-effort" && "$QUOTA_TYPE" != "on-demand" ]]; then
  echo "Error: Invalid QUOTA_TYPE."
  exit 1
fi

if gcloud compute tpus queued-resources describe $QUEUED_RESOURCE_ID --zone=$TPU_ZONE --project=$PROJECT_ID &> /dev/null; then
    echo "Queued Resource '$QUEUED_RESOURCE_ID' already exists. Skipping creation."
else
    echo "Creating Queued Resource '$QUEUED_RESOURCE_ID'..."
    gcloud compute tpus queued-resources create $QUEUED_RESOURCE_ID \
        --node-id=$TPU_NAME \
        --project=$PROJECT_ID \
        --zone=$TPU_ZONE \
        --accelerator-type=$ACCELERATOR_TYPE \
        --runtime-version=$RUNTIME_VERSION \
        --$QUOTA_TYPE
fi

# --- 3. Verify TPU VM Creation ---

echo "Waiting for Queued Resource to be ACTIVE..."
while true; do
    STATUS=$(gcloud compute tpus queued-resources describe $QUEUED_RESOURCE_ID \
        --project=$PROJECT_ID \
        --zone=$TPU_ZONE \
        --format="value(state)")

    if [[ "$STATUS" == "ACTIVE" ]]; then
        echo "Queued Resource is ACTIVE. TPU VM is ready."
        break
    elif [[ "$STATUS" == "FAILED" ]]; then
        echo "Error: Queued Resource creation failed."
        exit 1
    else
        echo "Current status: $STATUS. Waiting..."
        sleep 10
    fi
done

# --- 4. Build, Tag, and Push Docker Image ---
if [ -z "$IMAGE_NAME" ]; then
  IMAGE_NAME="tpu-training-image"
  echo "Using default image name: $IMAGE_NAME"
fi

echo "Building Docker image..."
docker build -t $IMAGE_NAME ..  # Note the ".." to build from the parent directory

echo "Tagging Docker image..."
docker tag $IMAGE_NAME gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

echo "Pushing Docker image to GCR..."
docker push gcr.io/$PROJECT_ID/$IMAGE_NAME:latest

# --- 5. Update the TPU VM to use the Docker Image ---

echo "Updating TPU VM to use Docker image..."
# Stop the previously created tpu
gcloud compute tpus tpu-vm stop $TPU_NAME --zone=$TPU_ZONE --project=$PROJECT_ID --quiet

gcloud compute tpus tpu-vm update $TPU_NAME \
  --zone=$TPU_ZONE \
  --worker=all \
  --version=$RUNTIME_VERSION \
  --accelerator-type=$ACCELERATOR_TYPE \
  --container-image=gcr.io/$PROJECT_ID/$IMAGE_NAME:latest
#Start the tpu
gcloud compute tpus tpu-vm start $TPU_NAME --zone=$TPU_ZONE --project=$PROJECT_ID

echo "Setup complete. You can now SSH into the TPU VM using:"
echo "gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$TPU_ZONE --project=$PROJECT_ID"
echo "To run your scripts, use:"
echo "gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$TPU_ZONE --project=$PROJECT_ID --command='python3 /app/your_script.py' --worker=all" 