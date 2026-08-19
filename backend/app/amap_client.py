from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _load_local_env() -> None:
    root = Path(__file__).resolve().parents[2]
    for env_path in (root / ".env", root / "backend" / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


class AMapClient:
    BASE = "https://restapi.amap.com"

    def __init__(self, api_key: str | None = None) -> None:
        _load_local_env()
        self.api_key = api_key or os.getenv("AMAP_API_KEY", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _get_json(self, path: str, params: dict, timeout: float = 6.0) -> dict:
        if not self.enabled:
            raise RuntimeError("AMAP_API_KEY not set")
        query = dict(params)
        query["key"] = self.api_key
        url = f"{self.BASE}{path}?{urlencode(query)}"
        req = Request(url, headers={"User-Agent": "ai-travel-companion/0.1"})
        with urlopen(req, timeout=timeout) as resp:  # nosec B310
            data = json.loads(resp.read().decode("utf-8"))
        if str(data.get("status")) != "1":
            raise RuntimeError(data.get("info") or "AMap API error")
        return data

    def reverse_geocode(self, lat: float, lon: float) -> dict:
        data = self._get_json(
            "/v3/geocode/regeo",
            {
                "location": f"{lon:.6f},{lat:.6f}",
                "extensions": "base",
                "radius": 1000,
                "poitype": "",
                "roadlevel": 0,
            },
        )
        regeo = data.get("regeocode", {})
        addr = regeo.get("addressComponent", {})
        city_raw = addr.get("city")
        city = city_raw[0] if isinstance(city_raw, list) and city_raw else city_raw
        return {
            "province": addr.get("province") or "",
            "city": city or addr.get("district") or "",
            "county": addr.get("district") or "",
            "adcode": addr.get("adcode") or "",
            "formatted_address": regeo.get("formatted_address") or "",
        }

    def geocode(self, address: str, city: str = "") -> dict | None:
        params = {"address": address}
        if city:
            params["city"] = city
        data = self._get_json("/v3/geocode/geo", params)
        geocodes = data.get("geocodes") or []
        if not geocodes:
            return None
        best = geocodes[0]
        location = best.get("location") or ""
        if "," not in location:
            return None
        lon_s, lat_s = location.split(",", 1)
        city_raw = best.get("city")
        city_name = city_raw[0] if isinstance(city_raw, list) and city_raw else city_raw
        return {
            "province": best.get("province") or "",
            "city": city_name or "",
            "county": best.get("district") or "",
            "adcode": best.get("adcode") or "",
            "formatted_address": best.get("formatted_address") or address,
            "lat": float(lat_s),
            "lon": float(lon_s),
        }

    def search_address(self, query: str, city: str = "", limit: int = 10) -> list[dict]:
        # Prefer input tips for UX, then fallback to place text.
        items: list[dict] = []
        try:
            tips = self._get_json(
                "/v3/assistant/inputtips",
                {
                    "keywords": query,
                    "city": city,
                    "citylimit": "false",
                    "datatype": "all",
                },
            ).get("tips", [])
            for tip in tips:
                location = tip.get("location") or ""
                if not location or "," not in location:
                    continue
                lon_s, lat_s = location.split(",", 1)
                district = tip.get("district") or ""
                items.append(
                    {
                        "name": tip.get("name") or tip.get("address") or "",
                        "display_name": f"{district}{tip.get('name') or ''}{tip.get('address') or ''}".strip(),
                        "province": "",
                        "city": district,
                        "county": district,
                        "lat": float(lat_s),
                        "lon": float(lon_s),
                        "adcode": tip.get("adcode") or "",
                    }
                )
                if len(items) >= limit:
                    return items[:limit]
        except Exception:
            pass

        try:
            pois = self._get_json(
                "/v3/place/text",
                {
                    "keywords": query,
                    "city": city,
                    "offset": min(20, max(1, limit)),
                    "page": 1,
                    "extensions": "base",
                },
            ).get("pois", [])
            for poi in pois:
                location = poi.get("location") or ""
                if not location or "," not in location:
                    continue
                lon_s, lat_s = location.split(",", 1)
                items.append(
                    {
                        "name": poi.get("name") or "",
                        "display_name": f"{poi.get('pname') or ''}{poi.get('cityname') or ''}{poi.get('adname') or ''}{poi.get('address') or ''}",
                        "province": poi.get("pname") or "",
                        "city": poi.get("cityname") or "",
                        "county": poi.get("adname") or "",
                        "lat": float(lat_s),
                        "lon": float(lon_s),
                        "adcode": poi.get("adcode") or "",
                    }
                )
                if len(items) >= limit:
                    break
        except Exception:
            pass

        return items[:limit]

    def weather(self, adcode_or_city: str, forecast: bool = False) -> dict:
        data = self._get_json(
            "/v3/weather/weatherInfo",
            {
                "city": adcode_or_city,
                "extensions": "all" if forecast else "base",
            },
        )
        lives = data.get("lives") or []
        if lives:
            return {"mode": "live", **lives[0]}
        forecasts = data.get("forecasts") or []
        if forecasts:
            return {"mode": "forecast", **forecasts[0]}
        return {"mode": "none"}

    def nearby_pois(
        self,
        lat: float,
        lon: float,
        keywords: str,
        types: str = "",
        radius: int = 6000,
        offset: int = 20,
        page: int = 1,
    ) -> list[dict]:
        params = {
            "location": f"{lon:.6f},{lat:.6f}",
            "keywords": keywords,
            "radius": radius,
            "sortrule": "distance",
            "offset": offset,
            "page": max(1, page),
            "extensions": "all",
        }
        if types:
            params["types"] = types
        data = self._get_json(
            "/v3/place/around",
            params,
            timeout=8.0,
        )
        return data.get("pois", []) or []

    def nearest_rail_station(self, lat: float, lon: float, city: str = "", radius: int = 50000) -> dict | None:
        queries = [
            ("高铁站", "150500"),
            ("火车站", "150500"),
        ]
        candidates: list[dict] = []

        for keyword, type_hint in queries:
            try:
                candidates.extend(
                    self.nearby_pois(
                        lat,
                        lon,
                        keywords=keyword,
                        types=type_hint,
                        radius=radius,
                        offset=8,
                        page=1,
                    )
                )
            except Exception:
                continue

        if not candidates and city:
            for keyword in (f"{city} 高铁站", f"{city} 火车站"):
                try:
                    candidates.extend(self.text_pois(keyword, city=city, types="150500", offset=8, page=1))
                except Exception:
                    continue

        best: dict | None = None
        best_distance: int | None = None
        for row in candidates:
            location = str(row.get("location") or "").strip()
            if "," not in location:
                continue
            raw_distance = row.get("distance")
            try:
                distance = int(float(raw_distance)) if raw_distance not in (None, "") else None
            except Exception:
                distance = None
            if best is None or (distance is not None and (best_distance is None or distance < best_distance)):
                best = row
                best_distance = distance

        return best

    def text_pois(self, keywords: str, city: str = "", types: str = "", offset: int = 20, page: int = 1) -> list[dict]:
        params = {
            "keywords": keywords,
            "offset": min(25, max(1, offset)),
            "page": max(1, page),
            "extensions": "all",
        }
        if city:
            params["city"] = city
        if types:
            params["types"] = types
        data = self._get_json("/v3/place/text", params, timeout=8.0)
        return data.get("pois", []) or []

    def distance_matrix(self, origin_lat: float, origin_lon: float, destinations: list[tuple[float, float]]) -> list[dict]:
        if not destinations:
            return []
        dest_str = "|".join([f"{lon:.6f},{lat:.6f}" for lat, lon in destinations])
        data = self._get_json(
            "/v3/distance",
            {
                "origins": f"{origin_lon:.6f},{origin_lat:.6f}",
                "destination": dest_str,
                "type": 1,
            },
            timeout=8.0,
        )
        return data.get("results", []) or []
