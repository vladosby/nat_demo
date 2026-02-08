import logging
from datetime import date

import requests
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WeatherToolFunctionConfig(FunctionBaseConfig, name="weather_tool"):
    """
    WeatherTool Config File
    """


class WeatherToolInput(BaseModel):
    city_name: str = Field(
        description="City name to search by (e.g., 'Warsaw', 'Berlin', 'Paris')"
    )


@register_function(config_type=WeatherToolFunctionConfig)
async def weather_tool_function(config: WeatherToolFunctionConfig, builder: Builder):
    """Register tool for search weather by city name."""

    async def _wrapper(city_name: str) -> str:
        result = get_today_weather(city_name)
        return result  # Already returns JSON string

    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=WeatherToolInput,
        description="Get information about weather for a specific city for today."
    )


def get_today_weather(city: str) -> dict:
    """
    Retrieve today's weather for a given city using Open-Meteo (no API key).

    Returns:
        {
            "city": str,
            "date": str,
            "temperature_max": float,
            "temperature_min": float,
            "weather_code": int,
            "wind_speed_max": float
        }
    """
    # 1. Geocode city name → lat/lon
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(geo_url, params={"name": city, "count": 1})
    geo_resp.raise_for_status()

    geo_data = geo_resp.json()
    if not geo_data.get("results"):
        raise ValueError(f"City not found: {city}")

    location = geo_data["results"][0]
    lat = location["latitude"]
    lon = location["longitude"]

    # 2. Fetch today's weather
    today = date.today().isoformat()
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_resp = requests.get(
        weather_url,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "weathercode",
                "windspeed_10m_max",
            ],
            "timezone": "auto",
            "start_date": today,
            "end_date": today,
        },
    )
    weather_resp.raise_for_status()
    weather_data = weather_resp.json()

    daily = weather_data["daily"]

    return {
        "city": location["name"],
        "date": today,
        "temperature_max": daily["temperature_2m_max"][0],
        "temperature_min": daily["temperature_2m_min"][0],
        "weather_code": daily["weathercode"][0],
        "wind_speed_max": daily["windspeed_10m_max"][0],
    }
