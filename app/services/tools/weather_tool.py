from __future__ import annotations

import logging
from typing import Any, Dict

import openmeteo_requests
import requests
from langchain_core.tools import StructuredTool

from app.core.config import Settings


def get_current_weather_from_api(
    latitude: float,
    longitude: float,
    *,
    temperature_unit: str = "celsius",
    wind_speed_unit: str = "kmh",
    timezone: str = "auto",
) -> Dict[str, Any]:
    """Fetch current weather from Open-Meteo for given coordinates."""
    base_url = Settings().openmeteo_base_url
    logger = logging.getLogger("vera.weather_tool")

    session = requests.Session()
    client = openmeteo_requests.Client(session=session)

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m", "wind_speed_10m"],
        "temperature_unit": temperature_unit,
        "wind_speed_unit": wind_speed_unit,
        "timezone": timezone,
    }

    responses = client.weather_api(base_url, params=params)
    if not responses:
        result = {"status": "error", "message": "No response from Open-Meteo"}
        logger.info("tool_end tool=%s output=%s", "get_current_weather", result)
        return result

    response = responses[0]
    current = response.Current()

    result = {
        "status": "ok",
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
        "timezone": response.Timezone(),
        "temperature_2m": current.Variables(0).Value(),
        "wind_speed_10m": current.Variables(1).Value(),
    }
    logger.info("tool_end tool=%s output=%s", "get_current_weather", result)
    return result


def get_current_weather_tool(
) -> StructuredTool:
    return StructuredTool.from_function(
        func=get_current_weather_from_api,
        name="get_current_weather",
        description=(
            "Get current weather for a latitude/longitude. "
            "Use when the user asks about weather."
        ),
    )
