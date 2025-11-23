# Copyright (c), Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
BCS (Binary Canonical Serialization) implementation in Python.
Matches the behavior of the Rust bcs crate for exact compatibility.
"""

import struct
from typing import Any, List, Dict, Tuple, Union
from dataclasses import fields, is_dataclass


def encode_u8(value: int) -> bytes:
    """Encode a u8 value."""
    if not 0 <= value <= 255:
        raise ValueError(f"u8 value out of range: {value}")
    return struct.pack('<B', value)


def encode_u64(value: int) -> bytes:
    """Encode a u64 value as little-endian."""
    if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"u64 value out of range: {value}")
    return struct.pack('<Q', value)


def encode_uleb128(value: int) -> bytes:
    """Encode a value as ULEB128 (unsigned LEB128)."""
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def encode_string(value: str) -> bytes:
    """Encode a string as length (ULEB128) + UTF-8 bytes."""
    utf8_bytes = value.encode('utf-8')
    length_bytes = encode_uleb128(len(utf8_bytes))
    return length_bytes + utf8_bytes


def encode_bytes(value: bytes) -> bytes:
    """Encode raw bytes as length (ULEB128) + bytes."""
    length_bytes = encode_uleb128(len(value))
    return length_bytes + value


def encode_vector(elements: List[Any]) -> bytes:
    """Encode a vector as length (ULEB128) + encoded elements."""
    length_bytes = encode_uleb128(len(elements))
    element_bytes = b''.join(to_bytes(elem) for elem in elements)
    return length_bytes + element_bytes


def to_bytes(obj: Any) -> bytes:
    """
    Serialize an object to BCS bytes.
    Matches Rust bcs::to_bytes behavior.
    
    Supports:
    - int: encoded as u64 (little-endian)
    - u8: use encode_u8 explicitly
    - str: encoded as length (ULEB128) + UTF-8 bytes
    - bytes: encoded as length (ULEB128) + raw bytes
    - list: encoded as vector (length ULEB128 + elements)
    - dataclass: encoded as struct (fields in order)
    - objects with __dict__: encoded as struct (fields in order)
    - objects with __bcs__ method: calls custom method
    """
    if isinstance(obj, int):
        # Default to u64 for integers
        return encode_u64(obj)
    elif isinstance(obj, str):
        return encode_string(obj)
    elif isinstance(obj, bytes):
        return encode_bytes(obj)
    elif isinstance(obj, (list, tuple)):
        # Encode as vector
        return encode_vector(list(obj))
    elif hasattr(obj, '__bcs__'):
        # Custom serializable object
        encoder = BCSEncoder()
        obj.__bcs__(encoder)
        return encoder.to_bytes()
    elif is_dataclass(obj):
        # Dataclass - encode fields in order
        # Special handling for IntentMessage to encode intent as u8
        if hasattr(obj, 'intent') and hasattr(obj, 'timestamp_ms') and hasattr(obj, 'data'):
            # IntentMessage structure: intent (enum u8), timestamp_ms (u64), data (struct)
            encoder = BCSEncoder()
            # Encode intent as u8 (enum variant)
            intent_value = obj.intent.value if hasattr(obj.intent, 'value') else int(obj.intent)
            encoder.write_u8(intent_value)
            # Encode timestamp_ms as u64
            encoder.write_u64(obj.timestamp_ms)
            # Encode data
            encoder.buffer.extend(to_bytes(obj.data))
            return encoder.to_bytes()
        else:
            # Regular dataclass - encode fields in order
            encoder = BCSEncoder()
            for field in fields(obj):
                value = getattr(obj, field.name)
                encoder.buffer.extend(to_bytes(value))
            return encoder.to_bytes()
    elif hasattr(obj, '__dict__'):
        # Regular class - encode fields in order
        # Note: Python 3.7+ preserves dict insertion order
        encoder = BCSEncoder()
        for field_value in obj.__dict__.values():
            encoder.buffer.extend(to_bytes(field_value))
        return encoder.to_bytes()
    else:
        raise TypeError(f"Cannot serialize type: {type(obj)}")


class BCSEncoder:
    """BCS encoder for structured data."""
    
    def __init__(self):
        self.buffer = bytearray()
    
    def write_u8(self, value: int):
        """Write a u8 value."""
        self.buffer.extend(encode_u8(value))
    
    def write_u64(self, value: int):
        """Write a u64 value."""
        self.buffer.extend(encode_u64(value))
    
    def write_string(self, value: str):
        """Write a string."""
        self.buffer.extend(encode_string(value))
    
    def write_bytes(self, value: bytes):
        """Write raw bytes with length prefix."""
        self.buffer.extend(encode_bytes(value))
    
    def write_vector(self, elements: List[Any]):
        """Write a vector."""
        self.buffer.extend(encode_vector(elements))
    
    def to_bytes(self) -> bytes:
        """Get the encoded bytes."""
        return bytes(self.buffer)

