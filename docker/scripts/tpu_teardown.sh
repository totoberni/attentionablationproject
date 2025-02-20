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

echo "WARNING: This will delete the following resources:"
echo "- TPU VM: $TPU_NAME"
echo "- Queued Resource: $QUEUED_RESOURCE_ID"
echo "- Network: $TPU_NETWORK"
echo "- Subnet: $TPU_SUBNET"

read -p "Are you sure you want to proceed? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

echo "Starting teardown process..."

# Delete TPU VM
if gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$TPU_ZONE" &> /dev/null; then
    echo "Deleting TPU VM: $TPU_NAME..."
    gcloud compute tpus tpu-vm delete "$TPU_NAME" \
        --zone="$TPU_ZONE" \
        --quiet
    handle_error $? "Failed to delete TPU VM"
else
    echo "TPU VM $TPU_NAME does not exist."
fi

# Delete Queued Resource
if gcloud compute tpus queued-resources describe "$QUEUED_RESOURCE_ID" --zone="$TPU_ZONE" &> /dev/null; then
    echo "Deleting Queued Resource: $QUEUED_RESOURCE_ID..."
    gcloud compute tpus queued-resources delete "$QUEUED_RESOURCE_ID" \
        --zone="$TPU_ZONE" \
        --quiet
    handle_error $? "Failed to delete Queued Resource"
else
    echo "Queued Resource $QUEUED_RESOURCE_ID does not exist."
fi

# Delete subnet
REGION=$(echo "$TPU_ZONE" | sed 's/.[^.]*$//')
if gcloud compute networks subnets describe "$TPU_SUBNET" --region="$REGION" &> /dev/null; then
    echo "Deleting subnet: $TPU_SUBNET..."
    gcloud compute networks subnets delete "$TPU_SUBNET" \
        --region="$REGION" \
        --quiet
    handle_error $? "Failed to delete subnet"
else
    echo "Subnet $TPU_SUBNET does not exist."
fi

# Delete network
if gcloud compute networks describe "$TPU_NETWORK" &> /dev/null; then
    echo "Deleting network: $TPU_NETWORK..."
    gcloud compute networks delete "$TPU_NETWORK" --quiet
    handle_error $? "Failed to delete network"
else
    echo "Network $TPU_NETWORK does not exist."
fi

echo "Teardown completed successfully!" 