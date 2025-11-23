# Copyright (c), Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
App state management for the nautilus server.
"""

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.backends import default_backend
import os


class AppState:
    """Application state containing ephemeral keypair and API key."""
    
    def __init__(self, api_key: str = ""):
        """
        Initialize app state with a new ephemeral keypair.
        
        Args:
            api_key: API key for external services (can be empty for seal-example)
        """
        # Generate new ephemeral Ed25519 keypair
        self.eph_private_key = ed25519.Ed25519PrivateKey.generate()
        self.eph_public_key = self.eph_private_key.public_key()
        self.api_key = api_key
    
    def get_public_key_bytes(self) -> bytes:
        """Get the public key as bytes."""
        return self.eph_public_key.public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw
        )
    
    def get_private_key(self) -> ed25519.Ed25519PrivateKey:
        """Get the private key."""
        return self.eph_private_key

