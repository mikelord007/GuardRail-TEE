# Copyright (c), Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Twitter example app for nautilus server.
"""

import re
import time
import requests
import binascii
from dataclasses import dataclass
from flask import Flask
from app_state import AppState
from common import (
    IntentMessage,
    IntentScope,
    ProcessDataRequest,
    ProcessedDataResponse,
    to_signed_response,
    EnclaveError,
)


@dataclass
class UserData:
    """User data response."""
    twitter_name: bytes  # Vec<u8> in Rust
    sui_address: bytes  # Vec<u8> in Rust


@dataclass
class UserRequest:
    """User request data."""
    user_url: str


def fetch_tweet_content(api_key: str, user_url: str) -> tuple[str, bytes]:
    """
    Fetch tweet content from Twitter API.
    Returns (twitter_name, sui_address_bytes).
    """
    client = requests.Session()
    
    if "/status/" in user_url:
        # Extract tweet ID from URL using regex
        re_tweet = re.compile(r"x\.com/\w+/status/(\d+)")
        match = re_tweet.search(user_url)
        if not match:
            raise EnclaveError("Invalid tweet URL")
        
        tweet_id = match.group(1)
        
        # Construct the Twitter API URL
        url = f"https://api.twitter.com/2/tweets/{tweet_id}?expansions=author_id&user.fields=username"
        
        # Make the request to Twitter API
        try:
            response = client.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10
            )
            response.raise_for_status()
            json_data = response.json()
        except requests.RequestException as e:
            raise EnclaveError(f"Failed to send request to Twitter API: {e}")
        except ValueError as e:
            raise EnclaveError(f"Failed to parse response from Twitter API: {e}")
        
        # Extract tweet text and author username
        tweet_text = json_data.get("data", {}).get("text")
        if not tweet_text:
            raise EnclaveError(f"Failed to extract tweet text: {json_data}")
        
        users = json_data.get("includes", {}).get("users", [])
        if not users:
            raise EnclaveError("Failed to extract username")
        twitter_name = users[0].get("username")
        if not twitter_name:
            raise EnclaveError("Failed to extract username")
        
        # Find the position of "#SUI" and extract address before it
        sui_tag_pos = tweet_text.find("#SUI")
        if sui_tag_pos == -1:
            raise EnclaveError("No #SUI tag found in tweet")
        
        text_before_tag = tweet_text[:sui_tag_pos]
        sui_address_re = re.compile(r"0x[0-9a-fA-F]{64}")
        match = sui_address_re.search(text_before_tag)
        if not match:
            raise EnclaveError("No valid Sui address found before #SUI in tweet")
        
        sui_address_hex = match.group(0)
        try:
            sui_address = binascii.unhexlify(sui_address_hex[2:])  # Remove 0x prefix
        except Exception:
            raise EnclaveError("Invalid Sui address")
        
        return twitter_name, sui_address
        
    else:
        # Handle profile URL
        re_profile = re.compile(r"x\.com/(\w+)(?:/)?$")
        match = re_profile.search(user_url)
        if not match:
            raise EnclaveError("Invalid profile URL")
        
        username = match.group(1)
        
        # Fetch user profile
        url = f"https://api.twitter.com/2/users/by/username/{username}?user.fields=description"
        
        try:
            response = client.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10
            )
            response.raise_for_status()
            json_data = response.json()
        except requests.RequestException as e:
            raise EnclaveError(f"Failed to send request to Twitter API: {e}")
        except ValueError as e:
            raise EnclaveError(f"Failed to parse response from Twitter API: {e}")
        
        # Extract user description
        description = json_data.get("data", {}).get("description")
        if not description:
            raise EnclaveError("Failed to extract user description")
        
        sui_tag_pos = description.find("#SUI")
        if sui_tag_pos == -1:
            raise EnclaveError("No #SUI tag found in profile description")
        
        text_before_tag = description[:sui_tag_pos]
        sui_address_re = re.compile(r"0x[0-9a-fA-F]{64}")
        match = sui_address_re.search(text_before_tag)
        if not match:
            raise EnclaveError("No valid Sui address found before #SUI in profile description")
        
        sui_address_hex = match.group(0)
        try:
            # Remove 0x prefix and decode hex
            sui_address = binascii.unhexlify(sui_address_hex[2:])
        except Exception:
            raise EnclaveError("Invalid Sui address")
        
        return username, sui_address


def process_data(
    state: AppState,
    request: ProcessDataRequest[UserRequest],
) -> ProcessedDataResponse[IntentMessage[UserData]]:
    """
    Process user data request.
    Fetches Twitter data and returns signed response.
    """
    user_url = request.payload.user_url
    current_timestamp = int(time.time() * 1000)
    
    # Fetch tweet content
    twitter_name, sui_address = fetch_tweet_content(state.api_key, user_url)
    
    user_data = UserData(
        twitter_name=twitter_name.encode('utf-8'),
        sui_address=sui_address,
    )
    
    return to_signed_response(
        state.get_private_key(),
        user_data,
        current_timestamp,
        IntentScope.ProcessData,
    )


def register_routes(app: Flask, state: AppState):
    """Register app-specific routes (none for twitter example)."""
    pass

