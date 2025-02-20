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

echo "Setting up TPU environment..."

# --- 1. Authenticate with Google Cloud ---
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    gcloud auth activate-service-account --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
    handle_error $? "Failed to authenticate with service account"
else
    echo "Warning: GOOGLE_APPLICATION_CREDENTIALS not set. Using default authentication."
fi

# --- 2. Set project and zone ---
gcloud config set project "$PROJECT_ID"
handle_error $? "Failed to set project"

gcloud config set compute/zone "$TPU_ZONE"
handle_error $? "Failed to set compute zone"

# --- 3. Configure Networking ---
echo "Configuring network..."

# Create a VPC network if it doesn't exist
if ! gcloud compute networks describe "$TPU_NETWORK" --project="$PROJECT_ID" &> /dev/null; then
    echo "Creating VPC network '$TPU_NETWORK'..."
    gcloud compute networks create "$TPU_NETWORK" --project="$PROJECT_ID" --subnet-mode=auto
    handle_error $? "Failed to create VPC network"
else
    echo "VPC network '$TPU_NETWORK' already exists."
fi

# Create a subnet if it doesn't exist
REGION=$(echo "$TPU_ZONE" | sed 's/.[^.]*$//')
if ! gcloud compute networks subnets describe "$TPU_SUBNET" --project="$PROJECT_ID" --region="$REGION" &> /dev/null; then
    echo "Creating subnet '$TPU_SUBNET' in region $REGION..."
    gcloud compute networks subnets create "$TPU_SUBNET" \
        --project="$PROJECT_ID" \
        --network="$TPU_NETWORK" \
        --range=10.240.0.0/16 \
        --region="$REGION"
    handle_error $? "Failed to create subnet"
else
    echo "Subnet '$TPU_SUBNET' already exists."
fi

# --- 4. Create Queued Resource ---
echo "Creating queued resource..."
gcloud compute tpus queued-resources create "$QUEUED_RESOURCE_ID" \
    --project="$PROJECT_ID" \
    --zone="$TPU_ZONE" \
    --accelerator-type="$ACCELERATOR_TYPE" \
    --runtime-version="$RUNTIME_VERSION" \
    --network="$TPU_NETWORK" \
    --subnetwork="$TPU_SUBNET" \
    --quota-type="$QUOTA_TYPE"
handle_error $? "Failed to create queued resource"

# --- 5. Create TPU VM ---
echo "Creating TPU VM..."
gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --project="$PROJECT_ID" \
    --zone="$TPU_ZONE" \
    --accelerator-type="$ACCELERATOR_TYPE" \
    --version="$RUNTIME_VERSION" \
    --network="$TPU_NETWORK" \
    --subnetwork="$TPU_SUBNET"
handle_error $? "Failed to create TPU VM"

# --- 6. Set up Docker on TPU VM ---
echo "Setting up Docker on TPU VM..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$TPU_ZONE" --command="sudo apt-get update && sudo apt-get install -y docker.io"
handle_error $? "Failed to install Docker"

# --- 7. Configure GCS bucket ---
if ! gsutil ls "gs://$BUCKET_NAME" &> /dev/null; then
    echo "Creating GCS bucket: gs://$BUCKET_NAME"
    gsutil mb -p "$PROJECT_ID" "gs://$BUCKET_NAME"
    handle_error $? "Failed to create GCS bucket"
else
    echo "GCS bucket gs://$BUCKET_NAME exists."
fi

# Create necessary subdirectories
for dir in "raw_data" "processed_data" "checkpoints" "logs" "cache" "tokenizers"; do
    if ! gsutil ls "gs://$BUCKET_NAME/$dir" &> /dev/null; then
        echo "Creating GCS directory: gs://$BUCKET_NAME/$dir"
        gsutil mb -p "$PROJECT_ID" "gs://$BUCKET_NAME/$dir" || true
    fi
done

echo "TPU environment setup completed successfully!" 