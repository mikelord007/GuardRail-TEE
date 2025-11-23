# Copyright (c), Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Common types and utilities for the nautilus server.
"""

import subprocess
import json
import time
from typing import TypeVar, Generic, Dict, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
import requests
import yaml
import os
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.backends import default_backend
import binascii

from bcs import to_bytes

T = TypeVar('T')


# Intent scope enum
class IntentScope(IntEnum):
    ProcessData = 0


@dataclass
class IntentMessage(Generic[T]):
    """Intent message wrapper containing intent scope, timestamp, and data."""
    intent: IntentScope
    timestamp_ms: int
    data: T


@dataclass
class ProcessedDataResponse(Generic[T]):
    """Response containing the intent message and signature."""
    response: T
    signature: str


@dataclass
class ProcessDataRequest(Generic[T]):
    """Request containing the payload."""
    payload: T


@dataclass
class GetAttestationResponse:
    """Response for get attestation endpoint."""
    attestation: str  # Hex-encoded attestation document


@dataclass
class HealthCheckResponse:
    """Response for health check endpoint."""
    pk: str  # Hex-encoded public key
    endpoints_status: Dict[str, bool]


class EnclaveError(Exception):
    """Enclave error exception."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def to_signed_response(
    private_key: ed25519.Ed25519PrivateKey,
    payload: T,
    timestamp_ms: int,
    intent: IntentScope,
) -> ProcessedDataResponse[IntentMessage[T]]:
    """
    Sign the BCS bytes of the payload with the keypair.
    Returns ProcessedDataResponse with IntentMessage and signature.
    """
    intent_msg = IntentMessage(
        intent=intent,
        timestamp_ms=timestamp_ms,
        data=payload,
    )
    
    # Serialize IntentMessage to BCS bytes
    signing_payload = to_bytes(intent_msg)
    
    # Sign with Ed25519 (no backend parameter needed in newer cryptography versions)
    signature = private_key.sign(signing_payload)
    
    # Encode signature as hex
    signature_hex = binascii.hexlify(signature).decode('ascii')
    
    return ProcessedDataResponse(
        response=intent_msg,
        signature=signature_hex,
    )


def get_attestation(public_key_bytes: bytes) -> GetAttestationResponse:
    """
    Get attestation document from NSM using Rust helper binary.
    Returns attestation document hex-encoded.
    
    Args:
        public_key_bytes: Ed25519 public key bytes to include in attestation
    
    Returns:
        GetAttestationResponse with hex-encoded attestation document
    
    Raises:
        EnclaveError: If attestation fails
    """
    try:
        import subprocess
        
        # Encode public key as hex
        public_key_hex = binascii.hexlify(public_key_bytes).decode('ascii')
        
        # Try to find the Rust helper binary
        # In enclave: /nsm-attestation-helper
        # On host: might be in different location
        helper_paths = [
            "/nsm-attestation-helper",  # In enclave
            "nsm-attestation-helper",   # In PATH
            "./nsm-attestation-helper", # Current directory
        ]
        
        helper_binary = None
        for path in helper_paths:
            try:
                # Check if file exists and is executable
                if os.path.exists(path) and os.access(path, os.X_OK):
                    helper_binary = path
                    break
            except Exception as e:
                continue
        
        if helper_binary is None:
            tried_paths = ", ".join(helper_paths)
            raise EnclaveError(f"NSM attestation helper binary not found at any of: {tried_paths}. Make sure it's built and available.")
        
        # Call the Rust helper binary
        result = subprocess.run(
            [helper_binary, public_key_hex],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else f"Helper binary exited with code {result.returncode}"
            raise EnclaveError(f"NSM attestation failed: {error_msg}")
        
        attestation_hex = result.stdout.strip()
        if not attestation_hex:
            raise EnclaveError("Failed to get attestation document (empty response)")
        
        return GetAttestationResponse(
            attestation=attestation_hex
        )
    except FileNotFoundError as e:
        # This could be the binary not found, or /dev/nsm not found inside the binary
        raise EnclaveError(f"File not found error during attestation: {str(e)}. Helper binary: {helper_binary if 'helper_binary' in locals() else 'unknown'}")
    except subprocess.TimeoutExpired:
        raise EnclaveError("NSM attestation timed out")
    except Exception as e:
        raise EnclaveError(f"Attestation error: {str(e)}")


def health_check(public_key_bytes: bytes) -> HealthCheckResponse:
    """
    Health check endpoint that checks connectivity to all domains
    and returns the enclave's public key.
    """
    pk_hex = binascii.hexlify(public_key_bytes).decode('ascii')
    
    # Load allowed endpoints from YAML file
    endpoints_status = {}
    
    yaml_content = load_allowed_endpoints_yaml()
    if yaml_content:
        try:
            yaml_data = yaml.safe_load(yaml_content)
            if yaml_data and 'endpoints' in yaml_data:
                endpoints = yaml_data['endpoints']
                
                # Create HTTP client with timeout
                timeout = 5
                
                for endpoint in endpoints:
                    if isinstance(endpoint, str):
                        # Check connectivity to each endpoint
                        if ".amazonaws.com" in endpoint:
                            url = f"https://{endpoint}/ping"
                        else:
                            url = f"https://{endpoint}"
                        
                        try:
                            response = requests.get(url, timeout=timeout, verify=False)
                            if ".amazonaws.com" in endpoint:
                                # For AWS endpoints, check if response body contains "healthy"
                                is_reachable = "healthy" in response.text.lower()
                            else:
                                # For non-AWS endpoints, check for 200 status
                                is_reachable = response.status_code == 200
                        except Exception:
                            is_reachable = False
                        
                        endpoints_status[endpoint] = is_reachable
        except Exception as e:
            # Failed to parse YAML, continue with empty status
            pass
    
    return HealthCheckResponse(
        pk=pk_hex,
        endpoints_status=endpoints_status,
    )


def load_allowed_endpoints_yaml() -> Optional[str]:
    """Load allowed_endpoints.yaml from file or built-in.
    
    Tries multiple paths to support both host and enclave environments:
    - Host: src/nautilus-server/apps/{app}/allowed_endpoints.yaml
    - Enclave: /apps/{app}/allowed_endpoints.yaml
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to read from file first (in current directory)
    local_path = os.path.join(current_dir, "allowed_endpoints.yaml")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r") as f:
                return f.read()
        except Exception:
            pass
    
    # Try built-in endpoints based on app feature
    app_name = os.environ.get("ENCLAVE_APP", "weather-example")
    
    # List of paths to try (in order of preference)
    paths_to_try = []
    
    # 1. Enclave path: /apps/{app}/allowed_endpoints.yaml (when running from /)
    paths_to_try.append(f"/apps/{app_name}/allowed_endpoints.yaml")
    
    # 2. Relative from current file: apps/{app}/allowed_endpoints.yaml (host, from src/nautilus-server)
    paths_to_try.append(os.path.join(current_dir, f"apps/{app_name}/allowed_endpoints.yaml"))
    
    # 3. Absolute from repo root: src/nautilus-server/apps/{app}/allowed_endpoints.yaml
    parent_dir = os.path.dirname(current_dir)  # Should be 'src' when in src/nautilus-server
    repo_root = os.path.dirname(parent_dir) if parent_dir else current_dir
    paths_to_try.append(os.path.join(repo_root, f"src/nautilus-server/apps/{app_name}/allowed_endpoints.yaml"))
    
    # Try each path
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return f.read()
            except Exception:
                continue
    
    return None

