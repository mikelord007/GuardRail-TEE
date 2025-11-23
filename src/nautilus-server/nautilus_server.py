# Copyright (c), Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Main nautilus server application.
"""

import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
from app_state import AppState
from common import (
    get_attestation,
    health_check,
    ProcessDataRequest,
    EnclaveError,
)

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import app examples based on ENCLAVE_APP environment variable
ENCLAVE_APP = os.environ.get("ENCLAVE_APP", "weather-example")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins, methods, and headers (matching Rust CORS config)

# Initialize app state
# API_KEY can be stored with secret-manager. Follow configure_enclave.sh prompts.
# For seal-example, api_key is empty and loaded via two-phase bootstrap
api_key = os.environ.get("API_KEY", "bec305a5fb7b4918a94104231252211")
if ENCLAVE_APP == "seal-example":
    api_key = ""

state = AppState(api_key=api_key)

# Import the appropriate app module
if ENCLAVE_APP == "weather-example":
    from apps.weather_example import process_data, register_routes
elif ENCLAVE_APP == "twitter-example":
    from apps.twitter_example import process_data, register_routes
elif ENCLAVE_APP == "seal-example":
    from apps.seal_example import process_data, register_routes, spawn_host_init_server
    # Spawn host-only init server for seal example
    spawn_host_init_server(state)
else:
    raise ValueError(f"Unknown ENCLAVE_APP: {ENCLAVE_APP}")

# Register app-specific routes
register_routes(app, state)


@app.route("/", methods=["GET"])
def ping():
    """Ping endpoint."""
    return "Pong!"


@app.route("/get_attestation", methods=["GET"])
def get_attestation_endpoint():
    """Get attestation endpoint."""
    try:
        public_key_bytes = state.get_public_key_bytes()
        response = get_attestation(public_key_bytes)
        return jsonify({
            "attestation": response.attestation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health_check", methods=["GET"])
def health_check_endpoint():
    """Health check endpoint."""
    try:
        public_key_bytes = state.get_public_key_bytes()
        response = health_check(public_key_bytes)
        return jsonify({
            "pk": response.pk,
            "endpoints_status": response.endpoints_status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/process_data", methods=["POST"])
def process_data_endpoint():
    """Process data endpoint."""
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        # Parse request - convert payload dict to appropriate dataclass
        if ENCLAVE_APP == "weather-example":
            from apps.weather_example import WeatherRequest
            payload_obj = WeatherRequest(**json_data.get("payload", {}))
        elif ENCLAVE_APP == "twitter-example":
            from apps.twitter_example import UserRequest
            payload_obj = UserRequest(**json_data.get("payload", {}))
        elif ENCLAVE_APP == "seal-example":
            from apps.seal_example import WeatherRequest
            payload_obj = WeatherRequest(**json_data.get("payload", {}))
        else:
            return jsonify({"error": f"Unknown ENCLAVE_APP: {ENCLAVE_APP}"}), 400
        
        req = ProcessDataRequest(payload=payload_obj)
        
        # Call app-specific process_data
        result = process_data(state, req)
        
        # Convert result to JSON-serializable format
        # Handle dataclass data
        if hasattr(result.response.data, "__dict__"):
            data_dict = result.response.data.__dict__
            # Convert bytes to hex strings for JSON serialization
            data_dict = {k: v.hex() if isinstance(v, bytes) else v for k, v in data_dict.items()}
        else:
            data_dict = result.response.data
        
        return jsonify({
            "response": {
                "intent": result.response.intent.value,
                "timestamp_ms": result.response.timestamp_ms,
                "data": data_dict
            },
            "signature": result.signature
        })
    except EnclaveError as e:
        return jsonify({"error": e.message}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Run on all interfaces, port 3000
    # Debug mode enabled but with interactive debugger disabled (requires /dev/shm which isn't available in enclave)
    app.run(host="0.0.0.0", port=3000, threaded=True, debug=True, use_reloader=False, use_debugger=False)

