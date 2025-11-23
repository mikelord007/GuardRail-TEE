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
# This section will be populated by configure_enclave.sh based on secret configuration

cat secrets.json | socat - VSOCK-CONNECT:$ENCLAVE_CID:7777
socat TCP4-LISTEN:3000,reuseaddr,fork VSOCK-CONNECT:$ENCLAVE_CID:3000 &

# Add api.weatherapi.com to vsock-proxy allowlist if not already present
if ! grep -q "api.weatherapi.com" /etc/nitro_enclaves/vsock-proxy.yaml 2>/dev/null; then
    echo "- {address: api.weatherapi.com, port: 443}" | sudo tee -a /etc/nitro_enclaves/vsock-proxy.yaml
fi

# Start vsock-proxy for api.weatherapi.com on port 8101
# This forwards traffic from the enclave (VSOCK port 8101) to api.weatherapi.com:443
echo "Starting vsock-proxy on port 8101..."
sudo vsock-proxy 8101 api.weatherapi.com 443 --config /etc/nitro_enclaves/vsock-proxy.yaml &
sleep 2

# Additional port configurations will be added here by configure_enclave.sh if needed
