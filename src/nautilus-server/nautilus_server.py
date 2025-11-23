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
    EnclaveError,
)

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import guardrail app based on ENCLAVE_APP environment variable
ENCLAVE_APP = os.environ.get("ENCLAVE_APP", "guardrail")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins, methods, and headers (matching Rust CORS config)

# Initialize app state
api_key = os.environ.get("API_KEY", "")
state = AppState(api_key=api_key)

# Import guardrail app module
if ENCLAVE_APP == "guardrail":
    from guardrail import register_routes, run_evaluation
    register_routes(app, state)
else:
    raise ValueError(f"Unknown ENCLAVE_APP: {ENCLAVE_APP}. Expected 'guardrail'.")


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


@app.route("/run_evaluation", methods=["POST"])
def run_evaluation_endpoint():
    """Run model evaluation endpoint."""
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        model_id = json_data.get("model_id")
        if not model_id:
            return jsonify({"error": "model_id is required"}), 400
        
        result = run_evaluation(model_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Run on all interfaces, port 3000
    # Debug mode enabled but with interactive debugger disabled (requires /dev/shm which isn't available in enclave)
    app.run(host="0.0.0.0", port=3000, threaded=True, debug=True, use_reloader=False, use_debugger=False)

