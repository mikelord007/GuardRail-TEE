# Copyright (c), Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Seal example app for nautilus server.
This includes two-phase bootstrap for secure parameter loading.
"""

import time
import requests
import threading
import yaml
import os
from dataclasses import dataclass
from typing import Optional, Dict, List
from flask import Flask, jsonify, request
from app_state import AppState
from common import (
    IntentMessage,
    IntentScope,
    ProcessDataRequest,
    ProcessedDataResponse,
    to_signed_response,
    EnclaveError,
)

# SEAL API key stored in a thread-safe way
SEAL_API_KEY: Optional[str] = None
SEAL_API_KEY_LOCK = threading.Lock()


@dataclass
class WeatherResponse:
    """Weather response data (same as weather example)."""
    location: str
    temperature: int


@dataclass
class WeatherRequest:
    """Weather request data."""
    location: str


# Load SEAL config
def load_seal_config():
    """Load SEAL configuration from YAML file."""
    # Try multiple possible paths
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "..", "src", "apps", "seal-example", "seal_config.yaml"),
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "apps", "seal-example", "seal_config.yaml"),
        "seal_config.yaml",
        "/seal_config.yaml",
    ]
    
    for config_path in possible_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return yaml.safe_load(f)
            except Exception:
                continue
    return None


SEAL_CONFIG = load_seal_config()


def process_data(
    state: AppState,
    request: ProcessDataRequest[WeatherRequest],
) -> ProcessedDataResponse[IntentMessage[WeatherResponse]]:
    """
    Process weather data request using SEAL API key.
    API key must be initialized via two-phase bootstrap first.
    """
    # Get SEAL API key
    with SEAL_API_KEY_LOCK:
        api_key = SEAL_API_KEY
    
    if not api_key:
        raise EnclaveError("API key not initialized. Please complete parameter load first.")
    
    url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={request.payload.location}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        json_data = response.json()
    except requests.RequestException as e:
        raise EnclaveError(f"Failed to get weather response: {e}")
    except ValueError as e:
        raise EnclaveError(f"Failed to parse weather response: {e}")
    
    location = json_data.get("location", {}).get("name", "Unknown")
    temperature = int(json_data.get("current", {}).get("temp_c", 0.0))
    last_updated_epoch = json_data.get("current", {}).get("last_updated_epoch", 0)
    last_updated_timestamp_ms = last_updated_epoch * 1000
    
    current_timestamp = int(time.time() * 1000)
    
    # 1 hour in milliseconds = 60 * 60 * 1000 = 3_600_000
    if last_updated_timestamp_ms + 3_600_000 < current_timestamp:
        raise EnclaveError("Weather API timestamp is too old")
    
    weather_response = WeatherResponse(
        location=location,
        temperature=temperature,
    )
    
    return to_signed_response(
        state.get_private_key(),
        weather_response,
        last_updated_timestamp_ms,
        IntentScope.ProcessData,
    )


def init_parameter_load(state: AppState, request_data: dict) -> dict:
    """
    Initialize parameter load (first phase of bootstrap).
    
    Note: This is a simplified version. The full implementation requires:
    - SEAL SDK Python bindings
    - Sui SDK Python bindings
    - Proper certificate generation and signing
    
    This placeholder maintains the API structure but needs full SEAL integration.
    """
    global SEAL_API_KEY
    
    with SEAL_API_KEY_LOCK:
        if SEAL_API_KEY is not None:
            raise EnclaveError("API key already set")
    
    # TODO: Implement full SEAL init_parameter_load logic:
    # 1. Generate session keypair
    # 2. Create certificate with enclave's ephemeral key
    # 3. Create PTB for seal_approve
    # 4. Create FetchKeyRequest with encryption keys
    # 5. Return encoded request
    
    # Placeholder implementation
    return {
        "encoded_request": "0000000000000000000000000000000000000000000000000000000000000000"
    }


def complete_parameter_load(state: AppState, request_data: dict) -> dict:
    """
    Complete parameter load (second phase of bootstrap).
    
    Note: This is a simplified version. The full implementation requires:
    - SEAL SDK Python bindings for decryption
    - Proper handling of encrypted objects and seal responses
    
    This placeholder maintains the API structure but needs full SEAL integration.
    """
    global SEAL_API_KEY
    
    with SEAL_API_KEY_LOCK:
        if SEAL_API_KEY is not None:
            raise EnclaveError("API key already set")
        
        # TODO: Implement full SEAL complete_parameter_load logic:
        # 1. Decrypt all encrypted objects using encryption secret key
        # 2. Extract API key from first decrypted secret
        # 3. Store API key in SEAL_API_KEY
        # 4. Return dummy secrets
        
        # Placeholder: set a dummy API key for testing
        # In production, this would come from decrypted SEAL objects
        SEAL_API_KEY = "placeholder_api_key"
    
    return {
        "dummy_secrets": []
    }


def spawn_host_init_server(state: AppState):
    """
    Spawn a separate Flask server on localhost:3001 for host-only bootstrap access.
    """
    host_app = Flask(__name__)
    
    @host_app.route("/ping", methods=["GET"])
    def ping():
        return jsonify({"message": "pong"})
    
    @host_app.route("/seal/init_parameter_load", methods=["POST"])
    def init_parameter_load_endpoint():
        try:
            json_data = request.get_json()
            if not json_data:
                return jsonify({"error": "Invalid JSON"}), 400
            result = init_parameter_load(state, json_data)
            return jsonify(result)
        except EnclaveError as e:
            return jsonify({"error": e.message}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @host_app.route("/seal/complete_parameter_load", methods=["POST"])
    def complete_parameter_load_endpoint():
        try:
            json_data = request.get_json()
            if not json_data:
                return jsonify({"error": "Invalid JSON"}), 400
            result = complete_parameter_load(state, json_data)
            return jsonify(result)
        except EnclaveError as e:
            return jsonify({"error": e.message}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    # Run in a separate thread
    def run_server():
        host_app.run(host="0.0.0.0", port=3001, threaded=True, use_reloader=False)
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()


def register_routes(app: Flask, state: AppState):
    """Register app-specific routes (none for seal example - uses separate host server)."""
    pass

