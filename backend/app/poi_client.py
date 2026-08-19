from __future__ import annotations

import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .recommendation import Poi

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def _http_get_json(url: str, params: dict, timeout: float = 3.0) -> dict | list:
    full_url = f"{url}?{urlencode(params)}"
    req = Request(full_url, headers={"User-Agent": "ai-travel-companion/0.1"})
    with urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, payload: str, timeout: float = 3.0) -> dict:
    req = Request(
        url,
        data=payload.encode("utf-8"),
        headers={
            "User-Agent": "ai-travel-companion/0.1",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        },
    )
    with urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode("utf-8"))


def _geo_city(city: str) -> tuple[float, float] | None:
    data = _http_get_json(
        NOMINATIM_URL,
        {
            "q": city,
            "format": "json",
            "limit": 1,
            "addressdetails": 0,
        },
        timeout=2.0,
    )
    if not data:
        return None
    first = data[0]
    return float(first["lat"]), float(first["lon"])


def _build_overpass_query(lat: float, lon: float, radius_m: int = 8000) -> str:
    return f"""
[out:json][timeout:20];
(
  node["tourism"~"attraction|museum|gallery|viewpoint"](around:{radius_m},{lat},{lon});
  way["tourism"~"attraction|museum|gallery|viewpoint"](around:{radius_m},{lat},{lon});
  relation["tourism"~"attraction|museum|gallery|viewpoint"](around:{radius_m},{lat},{lon});

  node["amenity"~"restaurant|cafe|fast_food"](around:{radius_m},{lat},{lon});
  way["amenity"~"restaurant|cafe|fast_food"](around:{radius_m},{lat},{lon});
  relation["amenity"~"restaurant|cafe|fast_food"](around:{radius_m},{lat},{lon});
);
out tags center 120;
""".strip()


def _poi_name(tags: dict) -> str | None:
    return tags.get("name:zh") or tags.get("name")


def _cost_for_attraction(tourism: str) -> int:
    if tourism in {"museum", "gallery"}:
        return 2
    if tourism == "viewpoint":
        return 1
    return 2


def _cost_for_food(amenity: str) -> int:
    if amenity == "fast_food":
        return 1
    return 2


def _category_for_attraction(tourism: str) -> str:
    return {
        "museum": "museum",
        "gallery": "gallery",
        "viewpoint": "landmark",
        "attraction": "attraction",
    }.get(tourism, "attraction")


def _is_indoor_attraction(tourism: str) -> bool:
    return tourism in {"museum", "gallery"}


def _extract_lat_lon(el: dict) -> tuple[float | None, float | None]:
    lat = el.get("lat")
    lon = el.get("lon")
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    center = el.get("center") or {}
    if center.get("lat") is not None and center.get("lon") is not None:
        return float(center["lat"]), float(center["lon"])
    return None, None


def _fetch_nominatim_fallback(city: str, lat: float | None = None, lon: float | None = None) -> tuple[list[Poi], list[Poi]]:
    attractions: list[Poi] = []
    foods: list[Poi] = []
    seen_attr: set[str] = set()
    seen_food: set[str] = set()

    near = f" near {lat},{lon}" if lat is not None and lon is not None else ""
    queries = [
        (f"tourist attraction in {city}{near}", "attraction"),
        (f"popular restaurant in {city}{near}", "restaurant"),
    ]

    for q, kind in queries:
        try:
            data = _http_get_json(
                NOMINATIM_URL,
                {
                    "q": q,
                    "format": "json",
                    "limit": 10,
                    "addressdetails": 0,
                },
                timeout=2.0,
            )
        except Exception:
            continue

        for row in data:
            name = row.get("name") or row.get("display_name", "").split(",")[0].strip()
            if not name:
                continue

            poi_lat = float(row.get("lat")) if row.get("lat") else None
            poi_lon = float(row.get("lon")) if row.get("lon") else None
            key = name.lower()

            if kind == "restaurant":
                if key in seen_food:
                    continue
                seen_food.add(key)
                foods.append(Poi(name=name, category="restaurant", indoor=True, cost_level=2, tags=("local", "restaurant"), lat=poi_lat, lon=poi_lon))
            else:
                if key in seen_attr:
                    continue
                seen_attr.add(key)
                attractions.append(Poi(name=name, category="attraction", indoor=False, cost_level=2, tags=("culture", "sunny_friendly"), lat=poi_lat, lon=poi_lon))

            if len(attractions) >= 16 and len(foods) >= 16:
                return attractions, foods

    return attractions, foods


def fetch_nearby_pois(lat: float, lon: float, city_hint: str = "") -> tuple[list[Poi], list[Poi]]:
    try:
        start = time.monotonic()
        budget_sec = 8.0

        query = _build_overpass_query(lat, lon)
        elements: list[dict] = []
        for endpoint in OVERPASS_URLS:
            if time.monotonic() - start > budget_sec:
                break
            try:
                data = _http_post_json(endpoint, urlencode({"data": query}), timeout=3.0)
                elements = data.get("elements", [])
                if elements:
                    break
            except Exception:
                continue

        if not elements:
            return _fetch_nominatim_fallback(city_hint or "current location", lat, lon)

        attractions: list[Poi] = []
        foods: list[Poi] = []
        seen_attr: set[str] = set()
        seen_food: set[str] = set()

        for el in elements:
            tags = el.get("tags") or {}
            name = _poi_name(tags)
            if not name:
                continue

            poi_lat, poi_lon = _extract_lat_lon(el)
            tourism = tags.get("tourism")
            amenity = tags.get("amenity")

            if tourism in {"attraction", "museum", "gallery", "viewpoint"}:
                key = name.strip().lower()
                if key in seen_attr:
                    continue
                seen_attr.add(key)

                tag_pool = [tourism, "culture", "local"]
                if tourism in {"attraction", "viewpoint"}:
                    tag_pool.append("sunny_friendly")
                if tourism in {"museum", "gallery"}:
                    tag_pool.append("rainy_friendly")

                attractions.append(
                    Poi(
                        name=name,
                        category=_category_for_attraction(tourism),
                        indoor=_is_indoor_attraction(tourism),
                        cost_level=_cost_for_attraction(tourism),
                        tags=tuple(tag_pool),
                        lat=poi_lat,
                        lon=poi_lon,
                    )
                )

            if amenity in {"restaurant", "cafe", "fast_food"}:
                key = name.strip().lower()
                if key in seen_food:
                    continue
                seen_food.add(key)

                cuisine = (tags.get("cuisine") or "").lower()
                tag_pool = ["local", amenity]
                if "seafood" in cuisine:
                    tag_pool.append("seafood")
                if "coffee" in cuisine:
                    tag_pool.append("dessert")
                if "chinese" in cuisine or "sichuan" in cuisine or "hotpot" in cuisine:
                    tag_pool.append("spicy")

                foods.append(
                    Poi(
                        name=name,
                        category=amenity,
                        indoor=True,
                        cost_level=_cost_for_food(amenity),
                        tags=tuple(tag_pool),
                        lat=poi_lat,
                        lon=poi_lon,
                    )
                )

            if len(attractions) >= 20 and len(foods) >= 20:
                break

        if attractions or foods:
            return attractions, foods
        return _fetch_nominatim_fallback(city_hint or "current location", lat, lon)
    except Exception:
        return _fetch_nominatim_fallback(city_hint or "current location", lat, lon)


def fetch_city_pois(city: str) -> tuple[list[Poi], list[Poi]]:
    try:
        geo = _geo_city(city)
        if not geo:
            return [], []
        return fetch_nearby_pois(geo[0], geo[1], city)
    except Exception:
        return [], []
