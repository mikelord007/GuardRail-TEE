# Copyright (c), Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
#!/bin/bash

# Check if an enclave is running
ENCLAVES=$(nitro-cli describe-enclaves)
if [ "$(echo "$ENCLAVES" | jq 'length')" -eq 0 ]; then
    echo "Error: No enclaves are currently running."
    echo "Please start an enclave first using: make run"
    exit 1
fi

# Gets the enclave id and CID
# expects there to be only one enclave running
ENCLAVE_ID=$(echo "$ENCLAVES" | jq -r ".[0].EnclaveID")
ENCLAVE_CID=$(echo "$ENCLAVES" | jq -r ".[0].EnclaveCID")

if [ -z "$ENCLAVE_ID" ] || [ "$ENCLAVE_ID" == "null" ]; then
    echo "Error: Could not get enclave ID"
    exit 1
fi

if [ -z "$ENCLAVE_CID" ] || [ "$ENCLAVE_CID" == "null" ]; then
    echo "Error: Could not get enclave CID"
    exit 1
fi

echo "Using enclave ID: $ENCLAVE_ID, CID: $ENCLAVE_CID"

# Kill any existing vsock-proxy and socat processes to avoid port conflicts
echo "Cleaning up existing processes..."
sudo pkill -f "vsock-proxy.*8101" || true
pkill -f "socat TCP4-LISTEN:3000" || true
sleep 2
# Secrets-block
# No secrets: create empty secrets.json for compatibility
echo '{}' > secrets.json
# No secrets: create empty secrets.json for compatibility
# No secrets: create empty secrets.json for compatibility
# No secrets: create empty secrets.json for compatibility
# No secrets: create empty secrets.json for compatibility
# No secrets: create empty secrets.json for compatibility
# No secrets: create empty secrets.json for compatibility
# This section will be populated by configure_enclave.sh based on secret configuration

cat secrets.json | socat - VSOCK-CONNECT:$ENCLAVE_CID:7777
socat TCP4-LISTEN:3000,reuseaddr,fork VSOCK-CONNECT:$ENCLAVE_CID:3000 &

# Add Hugging Face endpoints to vsock-proxy allowlist if not already present
for endpoint in huggingface.co cdn-lfs.huggingface.co hf.co; do
    if ! grep -q "$endpoint" /etc/nitro_enclaves/vsock-proxy.yaml 2>/dev/null; then
        echo "- {address: $endpoint, port: 443}" | sudo tee -a /etc/nitro_enclaves/vsock-proxy.yaml
    fi
done

# Start vsock-proxy for Hugging Face endpoints
# This forwards traffic from the enclave (VSOCK ports) to Hugging Face:443
echo "Starting vsock-proxy for Hugging Face endpoints..."
PORT=8101
for endpoint in huggingface.co cdn-lfs.huggingface.co hf.co; do
    echo "Starting vsock-proxy on port $PORT for $endpoint..."
    sudo vsock-proxy $PORT $endpoint 443 --config /etc/nitro_enclaves/vsock-proxy.yaml &
    PORT=$((PORT+1))
done
sleep 2

# Additional port configurations will be added here by configure_enclave.sh if needed
