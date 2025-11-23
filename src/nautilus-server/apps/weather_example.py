# Copyright (c), Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weather example app for nautilus server.
"""

import time
import requests
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
class WeatherResponse:
    """Weather response data."""
    location: str
    temperature: int


@dataclass
class WeatherRequest:
    """Weather request data."""
    location: str


def process_data(
    state: AppState,
    request: ProcessDataRequest[WeatherRequest],
) -> ProcessedDataResponse[IntentMessage[WeatherResponse]]:
    """
    Process weather data request.
    Fetches weather data from API and returns signed response.
    """
    url = f"https://api.weatherapi.com/v1/current.json?key={state.api_key}&q={request.payload.location}"
    
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


def register_routes(app: Flask, state: AppState):
    """Register app-specific routes (none for weather example)."""
    pass

