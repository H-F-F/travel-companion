from __future__ import annotations

from datetime import date

import httpx


class WeatherClient:
    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    async def fetch_daily_weather(self, city: str, target_date: date) -> dict:
        async with httpx.AsyncClient(timeout=8.0) as client:
            geo_resp = await client.get(self.GEO_URL, params={"name": city, "count": 1, "language": "zh", "format": "json"})
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()

            results = geo_data.get("results") or []
            if not results:
                raise ValueError(f"City not found: {city}")

            loc = results[0]
            lat = loc["latitude"]
            lon = loc["longitude"]
            resolved_name = loc.get("name", city)

            day = target_date.isoformat()
            forecast_resp = await client.get(
                self.FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": day,
                    "end_date": day,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                    "timezone": "auto",
                },
            )
            forecast_resp.raise_for_status()
            forecast_data = forecast_resp.json().get("daily", {})

            return {
                "city": resolved_name,
                "max_temp_c": float((forecast_data.get("temperature_2m_max") or [26])[0]),
                "min_temp_c": float((forecast_data.get("temperature_2m_min") or [18])[0]),
                "precipitation_mm": float((forecast_data.get("precipitation_sum") or [0])[0]),
                "wind_speed_mps": round(float((forecast_data.get("wind_speed_10m_max") or [3])[0]) / 3.6, 1),
            }
