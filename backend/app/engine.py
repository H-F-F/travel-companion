from __future__ import annotations

import json
import math
import os
import re
from datetime import date, datetime, time, timedelta
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .amap_client import AMapClient
from .poi_client import fetch_city_pois, fetch_nearby_pois
from .recommendation import (
    Poi,
    cost_level_text,
    is_chain_restaurant,
    score_attraction,
    score_food,
    score_hotel,
    weather_label,
    weather_label_zh,
    weather_tips,
)
from .train_12306_client import MCP12306Client

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - stdlib fallback
    ZoneInfo = None  # type: ignore[assignment]

AMAP = AMapClient()
TRAIN12306 = MCP12306Client()
try:
    LOCAL_TZ = ZoneInfo("Asia/Shanghai") if ZoneInfo else None
except Exception:  # pragma: no cover - OS tzdata fallback
    LOCAL_TZ = None

ATTRACTION_THEME_LABELS = {
    "culture": "人文景点",
    "nature": "自然景观",
    "landmark": "地标景点",
    "indoor": "室内场馆",
}
ATTRACTION_THEME_ORDER = ["culture", "nature", "landmark", "indoor"]
ATTRACTION_TARGET_PER_THEME = 5
RECOMMENDATION_FULL_RADIUS_KM = 5.0
ATTRACTION_FETCH_TARGET = 50
FOOD_FETCH_TARGET = 40
ATTRACTION_DISPLAY_LIMIT = 120
FOOD_DISPLAY_LIMIT = 100
POI_DISPLAY_CATEGORY_LABELS = {
    "culture": "人文景观",
    "nature": "自然景观",
    "urban_leisure": "城市休闲",
    "other": "其他景点",
}
POI_DISPLAY_CATEGORY_ORDER = ["culture", "nature", "urban_leisure", "other"]
AMAP_TYPECODE_DISPLAY_CATEGORY = {
    "110100": ("urban_leisure", "城市休闲"),
    "110101": ("nature", "自然景观"),
    "110102": ("culture", "人文景观"),
    "110103": ("culture", "人文景观"),
    "110104": ("culture", "人文景观"),
    "110105": ("nature", "自然景观"),
    "110106": ("nature", "自然景观"),
    "110107": ("urban_leisure", "城市休闲"),
    "110108": ("nature", "自然景观"),
    "110109": ("nature", "自然景观"),
    "110110": ("urban_leisure", "城市休闲"),
}
TRAIN_SEAT_LABELS = {
    "business": "商务座",
    "first_class": "一等座",
    "second_class": "二等座",
    "soft_sleeper": "软卧",
    "hard_sleeper": "硬卧",
    "hard_seat": "硬座",
    "no_seat": "无座",
    "premium_soft_sleeper": "高级软卧",
    "dynamic_sleeping": "动卧",
}
FOOD_COMPANY_NAME_TOKENS = (
    "有限责任公司",
    "股份有限公司",
    "有限公司",
    "集团",
    "实业",
    "商贸",
    "贸易",
    "供应链",
    "食品厂",
    "加工厂",
    "工厂",
    "产业园",
    "工业园",
)
FOOD_TYPE_HINT_TOKENS = (
    "餐饮服务",
    "中餐厅",
    "外国餐厅",
    "快餐厅",
    "咖啡厅",
    "茶艺馆",
    "冷饮店",
    "糕饼店",
    "小吃",
    "餐厅",
    "饭店",
)
FOOD_TYPE_BLOCKLIST_TOKENS = (
    "公司企业",
    "商务住宅",
    "购物服务",
    "科教文化服务",
    "金融保险服务",
    "政府机构及社会团体",
    "交通设施服务",
    "公共设施",
    "汽车服务",
    "汽车销售",
    "汽车维修",
    "医疗保健服务",
)
TRANSPORT_HUB_TYPE_LABELS = {
    "rail": "火车 / 高铁",
    "airport": "机场",
    "bus": "客运枢纽",
    "other": "交通枢纽",
}
TRANSPORT_HUB_SORT_ORDER = {
    "rail": 0,
    "airport": 1,
    "bus": 2,
    "other": 3,
}
TRANSPORT_HUB_RAIL_TOKENS = ("火车站", "高铁站", "铁路", "动车", "城际", "虹桥")
TRANSPORT_HUB_AIRPORT_TOKENS = ("机场", "航站楼")
TRANSPORT_HUB_BUS_TOKENS = ("客运站", "汽车站", "客运中心")


def _prefer_amap_nearest_rail() -> bool:
    return str(os.getenv("AMAP_NEAREST_RAIL_FIRST", "")).strip().lower() in {"1", "true", "yes", "on"}


def _http_get_json(url: str, params: dict, timeout_sec: float = 2.5) -> dict:
    full_url = f"{url}?{urlencode(params)}"
    req = Request(full_url, headers={"User-Agent": "ai-travel-companion/0.1"})
    with urlopen(req, timeout=timeout_sec) as resp:  # nosec B310
        return json.loads(resp.read().decode("utf-8"))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _travel_minutes_by_km(km: float) -> int:
    # city mixed traffic rough estimate
    return max(8, int((km / 22.0) * 60))


def _local_now() -> datetime:
    return datetime.now(LOCAL_TZ) if LOCAL_TZ else datetime.now()


def _parse_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value).strip())
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _parse_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    match = re.search(r"\d+", str(value).strip())
    if not match:
        return None
    try:
        return int(match.group(0))
    except Exception:
        return None


def _parse_clock_minutes(value: object) -> int:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{1,2}:\d{2}", text):
        return 24 * 60 + 1
    hour_s, minute_s = text.split(":", 1)
    return int(hour_s) * 60 + int(minute_s)


def _parse_duration_minutes(value: object) -> int:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{1,2}:\d{1,2}", text):
        return 24 * 60 + 1
    hour_s, minute_s = text.split(":", 1)
    return int(hour_s) * 60 + int(minute_s)


def _seat_availability_score(value: object) -> int:
    text = str(value or "").strip()
    if not text or text in {"--", "-", "无", "0"}:
        return 0
    if "无" in text and "候补" not in text:
        return 0
    if "候补" in text:
        return 1
    if text == "有":
        return 4
    count = _parse_int(text)
    if count is not None:
        return 3 if count > 0 else 0
    return 2


def _normalize_seat_items(seats: object) -> list[dict]:
    if not isinstance(seats, dict):
        return []
    items = []
    for key, raw_value in seats.items():
        value = str(raw_value or "").strip()
        if not value:
            continue
        label = TRAIN_SEAT_LABELS.get(str(key), str(key).replace("_", " "))
        items.append(
            {
                "key": str(key),
                "label": label,
                "value": value,
                "availability_score": _seat_availability_score(value),
            }
        )
    items.sort(key=lambda item: (item["availability_score"], item["label"]), reverse=True)
    return items


def _seat_summary_text(seat_items: list[dict], limit: int = 4) -> str:
    if not seat_items:
        return "暂无席位信息"
    return " / ".join([f"{item['label']} {item['value']}" for item in seat_items[:limit]])


def _normalize_transport_tickets(trains: object) -> list[dict]:
    if not isinstance(trains, list):
        return []
    items = []
    for train in trains:
        if not isinstance(train, dict):
            continue
        seat_items = _normalize_seat_items(train.get("seats") or {})
        available_count = sum(1 for item in seat_items if item["availability_score"] > 0)
        availability_score = max([item["availability_score"] for item in seat_items], default=0)
        items.append(
            {
                "train_no": str(train.get("train_no") or "").strip(),
                "from_station": str(train.get("from_station") or "").strip(),
                "to_station": str(train.get("to_station") or "").strip(),
                "from_station_code": str(train.get("from_station_code") or "").strip(),
                "to_station_code": str(train.get("to_station_code") or "").strip(),
                "start_time": str(train.get("start_time") or "").strip(),
                "arrive_time": str(train.get("arrive_time") or "").strip(),
                "duration": str(train.get("duration") or "").strip(),
                "seat_items": [{"label": item["label"], "value": item["value"]} for item in seat_items],
                "seat_summary": _seat_summary_text(seat_items),
                "best_seat": (
                    f"{seat_items[0]['label']} {seat_items[0]['value']}"
                    if seat_items
                    else "暂无席位信息"
                ),
                "availability_score": availability_score,
                "available_seat_count": available_count,
            }
        )
    items.sort(
        key=lambda item: (
            -item["availability_score"],
            -item["available_seat_count"],
            _parse_duration_minutes(item["duration"]),
            _parse_clock_minutes(item["start_time"]),
        )
    )
    return items[:8]


def _normalize_transport_transfers(transfers: object) -> list[dict]:
    if not isinstance(transfers, list):
        return []
    items = []
    for transfer in transfers:
        if not isinstance(transfer, dict):
            continue
        segments = []
        segment_scores = []
        for segment in transfer.get("segments") or []:
            if not isinstance(segment, dict):
                continue
            seat_items = _normalize_seat_items(segment.get("seats") or {})
            segment_scores.append(max([item["availability_score"] for item in seat_items], default=0))
            segments.append(
                {
                    "train_no": str(segment.get("train_no") or "").strip(),
                    "from_station": str(segment.get("from_station") or "").strip(),
                    "to_station": str(segment.get("to_station") or "").strip(),
                    "start_time": str(segment.get("start_time") or "").strip(),
                    "arrive_time": str(segment.get("arrive_time") or "").strip(),
                    "duration": str(segment.get("duration") or "").strip(),
                    "seat_items": [{"label": item["label"], "value": item["value"]} for item in seat_items],
                    "seat_summary": _seat_summary_text(seat_items),
                }
            )
        items.append(
            {
                "middle_station": str(transfer.get("middle_station") or "").strip(),
                "total_duration": str(transfer.get("total_duration") or "").strip(),
                "wait_time": str(transfer.get("wait_time") or "").strip(),
                "segments": segments,
                "availability_score": max(segment_scores, default=0),
            }
        )
    items.sort(
        key=lambda item: (
            -item["availability_score"],
            _parse_duration_minutes(item["total_duration"]),
            _parse_duration_minutes(item["wait_time"]),
        )
    )
    return items[:5]


def _selected_transport_arrival(selected_transport: object) -> dict:
    if not isinstance(selected_transport, dict):
        return {}
    kind = str(selected_transport.get("kind") or "").strip().lower()
    raw = selected_transport.get("raw") or {}
    if not isinstance(raw, dict):
        raw = {}
    arrive_time = str(selected_transport.get("arrive_time") or raw.get("arrive_time") or "").strip()
    start_time = str(selected_transport.get("start_time") or raw.get("start_time") or "").strip()
    label = str(selected_transport.get("headline") or selected_transport.get("label") or "").strip()
    if kind == "transfer" and not arrive_time:
        segments = raw.get("segments") or []
        if isinstance(segments, list) and segments:
            last = segments[-1] if isinstance(segments[-1], dict) else {}
            first = segments[0] if isinstance(segments[0], dict) else {}
            arrive_time = str(last.get("arrive_time") or "").strip()
            start_time = str(first.get("start_time") or "").strip()
    arrival_minutes = _parse_clock_minutes(arrive_time)
    if arrival_minutes > 24 * 60:
        return {}
    return {
        "kind": kind or "direct",
        "arrive_time": arrive_time,
        "start_time": start_time,
        "arrival_minutes": arrival_minutes,
        "label": label,
    }


def _guess_city_from_station_name(name: str, query: str = "") -> str:
    station_name = str(name or "").strip()
    query_text = str(query or "").strip().replace("市", "")
    query_text = re.sub(r"(火车站|高铁站|机场|站)$", "", query_text)
    if query_text and all(ord(ch) < 128 for ch in query_text):
        query_text = ""
    if query_text and query_text in station_name:
        return query_text

    cleaned = re.sub(r"(火车站|高铁站|机场|站)$", "", station_name)
    if cleaned.endswith("虹桥") and len(cleaned) > 2:
        cleaned = cleaned[:-2]
    if cleaned.endswith(("东", "西", "南", "北")) and len(cleaned) > 2:
        cleaned = cleaned[:-1]
    return cleaned.strip()


def _classify_transport_hub(name: str, type_text: str = "", type_code: str = "") -> tuple[str, str]:
    text = f"{name} {type_text}".strip()
    normalized_code = _normalize_amap_typecode(type_code)
    if normalized_code.startswith("1505") or any(token in text for token in TRANSPORT_HUB_RAIL_TOKENS):
        return "rail", TRANSPORT_HUB_TYPE_LABELS["rail"]
    if normalized_code.startswith("1507") or any(token in text for token in TRANSPORT_HUB_AIRPORT_TOKENS):
        return "airport", TRANSPORT_HUB_TYPE_LABELS["airport"]
    if normalized_code.startswith("1504") or any(token in text for token in TRANSPORT_HUB_BUS_TOKENS):
        return "bus", TRANSPORT_HUB_TYPE_LABELS["bus"]
    if "交通枢纽" in text:
        return "other", TRANSPORT_HUB_TYPE_LABELS["other"]
    return "", ""


def _is_valid_destination_station_name(name: str, provider: str = "", hub_type: str = "") -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    if hub_type and hub_type != "rail":
        return False
    forbidden_tokens = (
        "酒店",
        "宾馆",
        "民宿",
        "客栈",
        "停车场",
        "停车楼",
        "地下停车",
        "购物",
        "商场",
        "公寓",
        "广场",
        "服务区",
        "游客中心",
        "接送点",
        "售票处",
    )
    if any(token in text for token in forbidden_tokens):
        return False
    if provider == "12306-mcp":
        return True
    rail_name_patterns = ("火车站", "高铁站", "铁路", "站")
    if not any(token in text for token in rail_name_patterns):
        return False
    if text.endswith(("停车场", "停车楼", "大酒店", "酒店", "宾馆")):
        return False
    return True


def _normalize_station_match_key(name: str) -> str:
    text = str(name or "").strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("火车站", "站").replace("高铁站", "站")
    text = re.sub(r"[（）()\-\·,，/]+", "", text)
    return text.casefold()


def _is_standard_rail_station_name(name: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    blocked_tokens = (
        "进站口",
        "出站口",
        "检票口",
        "候车室",
        "售票处",
        "公交站",
        "公交站台",
        "汽车站",
        "地铁站",
        "停车场",
        "停车楼",
        "落客平台",
        "上客区",
        "下客区",
        "地下通道",
        "换乘",
    )
    if any(token in text for token in blocked_tokens):
        return False
    if not text.endswith("站") and "火车站" not in text and "高铁站" not in text:
        return False
    return _is_valid_destination_station_name(text, provider="12306-mcp", hub_type="rail")


def _normalize_city_label(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"(特别行政区|自治州|地区|盟|省|市|区|县)$", "", value)
    value = re.sub(r"\s+", "", value)
    return value


def _normalize_station_core_name(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"(火车站|高铁站)$", "站", value)
    value = re.sub(r"\s+", "", value)
    return value


def _transport_hub_city_rank(item: dict, resolved_city: str, normalized_query: str = "") -> tuple[int, int, int]:
    city_key = _normalize_city_label(resolved_city or normalized_query)
    name = _normalize_station_core_name(str(item.get("name") or ""))
    city_guess = _normalize_city_label(str(item.get("city_guess") or ""))
    if not city_key:
        return (9, 9, 9)

    name_without_suffix = re.sub(r"站$", "", name)
    if name_without_suffix == city_key:
        return (0, 0, 0)
    if name in {f"{city_key}站", f"{city_key}北站", f"{city_key}南站", f"{city_key}西站", f"{city_key}东站"}:
        return (0, 0, 1)
    if name_without_suffix.startswith(city_key):
        return (0, 1, len(name_without_suffix))
    if city_guess == city_key:
        return (0, 2, len(name_without_suffix))
    if city_key in name_without_suffix:
        return (1, 0, len(name_without_suffix))
    return (9, 9, len(name_without_suffix))


def search_train_stations(query: str, limit: int = 10) -> dict:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("query is required")

    result = TRAIN12306.search_stations(normalized_query, limit=max(1, min(20, limit)))
    stations = result.get("stations") if isinstance(result, dict) else []
    seen = set()
    items = []
    for station in stations or []:
        if not isinstance(station, dict):
            continue
        name = str(station.get("name") or "").strip()
        code = str(station.get("code") or "").strip()
        if not name or not code:
            continue
        key = (name.lower(), code.upper())
        if key in seen:
            continue
        seen.add(key)
        pinyin = str(station.get("pinyin") or "").strip()
        py_short = str(station.get("py_short") or "").strip()
        items.append(
            {
                "name": name,
                "code": code,
                "pinyin": pinyin,
                "py_short": py_short,
                "num": str(station.get("num") or "").strip(),
                "city_guess": _guess_city_from_station_name(name, normalized_query),
                "display_name": " · ".join([part for part in [name, code, py_short or pinyin] if part]),
            }
        )
        if len(items) >= limit:
            break

    return {
        "query": normalized_query,
        "count": len(items),
        "provider": "12306-mcp",
        "items": items,
    }


def search_transport_hubs(query: str = "", city: str = "", lat: float | None = None, lon: float | None = None, limit: int = 8) -> dict:
    normalized_query = str(query or "").strip()
    resolved_city = str(city or "").strip()
    center_lat = lat
    center_lon = lon

    if (center_lat is None or center_lon is None) and (normalized_query or resolved_city):
        parsed = geocode_address(normalized_query or resolved_city, resolved_city)
        if parsed:
            center_lat = parsed.get("lat")
            center_lon = parsed.get("lon")
            parsed_city = str(parsed.get("city") or "").replace("市", "").strip()
            if parsed_city:
                resolved_city = parsed_city

    if not resolved_city and center_lat is not None and center_lon is not None:
        resolved_city = resolve_city_from_coords(center_lat, center_lon) or _guess_city_by_coords(center_lat, center_lon) or ""

    response = {
        "query": normalized_query,
        "city": resolved_city,
        "lat": center_lat,
        "lon": center_lon,
        "provider": "none",
        "count": 0,
        "recommended": {},
        "items": [],
    }

    if not normalized_query and not resolved_city and (center_lat is None or center_lon is None):
        return response

    seen: set[str] = set()
    items: list[dict] = []
    providers: list[str] = []
    station_master: dict[str, dict] = {}

    station_queries: list[str] = []
    if resolved_city:
        station_queries.append(resolved_city)
    if normalized_query and normalized_query not in station_queries:
        station_queries.append(normalized_query)

    if TRAIN12306.enabled and station_queries:
        providers.append("12306-mcp")
        for station_query in station_queries:
            try:
                station_result = TRAIN12306.search_stations(station_query, limit=max(8, min(20, limit * 2)))
                for station in station_result.get("stations") or []:
                    if not isinstance(station, dict):
                        continue
                    station_name = str(station.get("name") or "").strip()
                    station_code = str(station.get("code") or "").strip()
                    if not station_name or not station_code:
                        continue
                    station_master.setdefault(
                        _normalize_station_match_key(station_name),
                        {
                            "name": station_name,
                            "code": station_code,
                            "pinyin": str(station.get("pinyin") or "").strip(),
                            "py_short": str(station.get("py_short") or "").strip(),
                        },
                    )
            except Exception:
                continue

    def add_item(
        *,
        name: str,
        provider: str,
        hub_type: str,
        hub_label: str,
        code: str = "",
        pinyin: str = "",
        py_short: str = "",
        amap_type: str = "",
        amap_typecode: str = "",
        poi_lat: float | None = None,
        poi_lon: float | None = None,
        priority_rank: int = 10,
    ) -> None:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            return
        if not _is_valid_destination_station_name(normalized_name, provider=provider, hub_type=hub_type):
            return
        dedupe_key = f"{normalized_name.casefold()}|{round(poi_lat or 0.0, 4)}|{round(poi_lon or 0.0, 4)}|{hub_type}"
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        distance_km = (
            round(_haversine_km(center_lat, center_lon, poi_lat, poi_lon), 2)
            if center_lat is not None and center_lon is not None and poi_lat is not None and poi_lon is not None
            else None
        )
        items.append(
            {
                "name": normalized_name,
                "code": code,
                "city_guess": _guess_city_from_station_name(normalized_name, normalized_query or resolved_city),
                "pinyin": pinyin,
                "py_short": py_short,
                "provider": provider,
                "hub_type": hub_type,
                "hub_type_label": hub_label,
                "priority_rank": priority_rank,
                "distance_km": distance_km,
                "lat": poi_lat,
                "lon": poi_lon,
                "amap_type": amap_type,
                "amap_typecode": amap_typecode,
            }
        )

    if AMAP.enabled:
        providers.append("amap")
        if _prefer_amap_nearest_rail() and center_lat is not None and center_lon is not None:
            try:
                nearest_row = AMAP.nearest_rail_station(center_lat, center_lon, city=resolved_city, radius=50000)
            except Exception:
                nearest_row = None
            if isinstance(nearest_row, dict):
                nearest_name = str(nearest_row.get("name") or "").strip()
                nearest_type = str(nearest_row.get("type") or "").strip()
                nearest_typecode = str(nearest_row.get("typecode") or nearest_row.get("typeCode") or "").strip()
                nearest_hub_type, nearest_hub_label = _classify_transport_hub(nearest_name, nearest_type, nearest_typecode)
                nearest_location = str(nearest_row.get("location") or "").strip()
                nearest_lat = None
                nearest_lon = None
                if "," in nearest_location:
                    nearest_lon_s, nearest_lat_s = nearest_location.split(",", 1)
                    nearest_lat = float(nearest_lat_s)
                    nearest_lon = float(nearest_lon_s)
                mapped_station = station_master.get(_normalize_station_match_key(nearest_name))
                nearest_provider = "12306-mcp" if mapped_station else "amap-nearest"
                if nearest_hub_type == "rail" and _is_standard_rail_station_name(nearest_name):
                    add_item(
                        name=str((mapped_station or {}).get("name") or nearest_name).strip(),
                        provider=nearest_provider,
                        hub_type=nearest_hub_type,
                        hub_label=nearest_hub_label,
                        code=str((mapped_station or {}).get("code") or "").strip(),
                        pinyin=str((mapped_station or {}).get("pinyin") or "").strip(),
                        py_short=str((mapped_station or {}).get("py_short") or "").strip(),
                        amap_type=nearest_type,
                        amap_typecode=_normalize_amap_typecode(nearest_typecode),
                        poi_lat=nearest_lat,
                        poi_lon=nearest_lon,
                        priority_rank=0,
                    )
        amap_rows: list[dict] = []
        search_seed = normalized_query or resolved_city
        text_queries: list[str] = []
        if search_seed:
            if "站" in search_seed:
                text_queries.append(search_seed)
            else:
                text_queries.extend([f"{search_seed} 火车站", f"{search_seed} 高铁站"])
        else:
            text_queries.extend(["火车站", "高铁站"])

        for keyword in dict.fromkeys([item.strip() for item in text_queries if item.strip()]):
            try:
                amap_rows.extend(AMAP.text_pois(keyword, city=resolved_city, offset=min(25, max(limit * 2, 10)), page=1))
            except Exception:
                continue

        if center_lat is not None and center_lon is not None:
            nearby_queries = (
                ("火车站", "150500"),
                ("高铁站", "150500"),
            )
            for keyword, type_hint in nearby_queries:
                try:
                    amap_rows.extend(
                        AMAP.nearby_pois(
                            center_lat,
                            center_lon,
                            keywords=keyword,
                            types=type_hint,
                            radius=50000,
                            offset=min(25, max(limit * 2, 10)),
                            page=1,
                        )
                    )
                except Exception:
                    continue

        for row in amap_rows:
            name = str(row.get("name") or "").strip()
            location = str(row.get("location") or "").strip()
            type_text = str(row.get("type") or "").strip()
            type_code = str(row.get("typecode") or row.get("typeCode") or "").strip()
            hub_type, hub_label = _classify_transport_hub(name, type_text, type_code)
            if hub_type != "rail":
                continue
            if not _is_standard_rail_station_name(name):
                continue
            mapped_station = station_master.get(_normalize_station_match_key(name))
            if TRAIN12306.enabled and not mapped_station:
                continue
            poi_lat = None
            poi_lon = None
            if "," in location:
                lon_s, lat_s = location.split(",", 1)
                poi_lat = float(lat_s)
                poi_lon = float(lon_s)
            add_item(
                name=str((mapped_station or {}).get("name") or name).strip(),
                provider="amap",
                hub_type=hub_type,
                hub_label=hub_label,
                code=str((mapped_station or {}).get("code") or "").strip(),
                pinyin=str((mapped_station or {}).get("pinyin") or "").strip(),
                py_short=str((mapped_station or {}).get("py_short") or "").strip(),
                amap_type=type_text,
                amap_typecode=_normalize_amap_typecode(type_code),
                poi_lat=poi_lat,
                poi_lon=poi_lon,
            )

    for station in station_master.values():
        add_item(
            name=str(station.get("name") or "").strip(),
            provider="12306-mcp",
            hub_type="rail",
            hub_label=TRANSPORT_HUB_TYPE_LABELS["rail"],
            code=str(station.get("code") or "").strip(),
            pinyin=str(station.get("pinyin") or "").strip(),
            py_short=str(station.get("py_short") or "").strip(),
        )

    items.sort(
        key=lambda item: (
            int(item.get("priority_rank") or 10),
            _transport_hub_city_rank(item, resolved_city, normalized_query),
            TRANSPORT_HUB_SORT_ORDER.get(str(item.get("hub_type") or ""), 99),
            0 if str(item.get("provider") or "") == "12306-mcp" else 1,
            9999 if item.get("distance_km") is None else float(item["distance_km"]),
            str(item.get("name") or ""),
        )
    )
    limited_items = items[: max(1, min(20, limit))]
    return {
        **response,
        "provider": "+".join(dict.fromkeys(providers)) if providers else "none",
        "count": len(limited_items),
        "recommended": limited_items[0] if limited_items else {},
        "items": limited_items,
    }


def _build_transport_info(
    booking_date: str,
    origin_city: str,
    origin_station: str,
    origin_station_code: str,
    destination_city: str,
    destination_station: str,
    destination_station_code: str,
    destination_query: str,
    destination_hub: dict | None,
    destination_hubs: list[dict] | None,
    train_date: date,
) -> dict:
    from_query = origin_station or origin_city
    to_query = destination_station or destination_query or destination_city
    to_query_code = str(destination_station_code or (destination_hub or {}).get("code") or "").strip().upper()
    transport = {
        "booking_date": booking_date,
        "origin_city": origin_city,
        "origin_station": origin_station,
        "origin_station_code": origin_station_code,
        "destination_city": destination_city,
        "destination_station": destination_station,
        "destination_query": to_query,
        "destination_station_code": to_query_code,
        "destination_hub": destination_hub or {},
        "destination_hubs": destination_hubs or [],
        "train_date": train_date.isoformat(),
        "from_query": from_query,
        "to_query": to_query,
        "provider": "not-requested",
        "status": "not_requested",
        "enabled": TRAIN12306.enabled,
        "note": "未触发 12306 查询；当前仅生成目的地内路线。",
        "tickets": [],
        "transfers": [],
        "warnings": [],
        "errors": {},
    }
    if not (booking_date or origin_city or origin_station):
        return {
            **transport,
            "note": "未填写预定日期或出发城市；当前仅生成目的地内路线。",
        }
    if not origin_city:
        return {
            **transport,
            "note": "未填写出发城市；当前直接按目的地地址生成本地路线，未调用 12306。",
        }
    if not from_query:
        return {
            **transport,
            "provider": "12306-missing-origin",
            "status": "missing_origin",
            "note": "已填写预定日期，但还没有出发城市或车站，暂时无法发起 12306 查询。",
        }
    if not TRAIN12306.enabled:
        return {
            **transport,
            "provider": "12306-disabled",
            "status": "disabled",
            "note": "已记录出发信息，但当前既没有可用的 12306 MCP 地址，也没有检测到本地 12306 MCP 包；请配置 `MCP_12306_URL` 或安装本地 12306 MCP 后重试。",
        }

    try:
        raw = TRAIN12306.fetch_trip_options(
            from_query,
            to_query,
            train_date.isoformat(),
            from_station_code=origin_station_code,
            to_station_code=to_query_code,
        )
    except Exception as exc:
        return {
            **transport,
            "provider": "12306-mcp",
            "status": "error",
            "note": f"12306 查询失败：{exc}",
            "errors": {"transport": str(exc)},
        }

    tickets = _normalize_transport_tickets((raw.get("tickets") or {}).get("trains") or [])
    transfers = _normalize_transport_transfers((raw.get("transfers") or {}).get("transfers") or [])
    errors = raw.get("errors") if isinstance(raw, dict) else {}
    direct_count = _parse_int((raw.get("tickets") or {}).get("count")) if isinstance(raw, dict) else None
    transfer_count = _parse_int((raw.get("transfers") or {}).get("count")) if isinstance(raw, dict) else None
    direct_count = direct_count if direct_count is not None else len(tickets)
    transfer_count = transfer_count if transfer_count is not None else len(transfers)
    warnings = []
    if errors.get("tickets"):
        warnings.append(f"直达查询失败：{errors['tickets']}")
    if errors.get("transfers"):
        warnings.append(f"中转换乘查询失败：{errors['transfers']}")

    if tickets or transfers:
        note_parts = [f"已接入 12306：{from_query} -> {to_query} 在 {train_date.isoformat()}"]
        if origin_station_code:
            note_parts.append(f"出发站代码 {origin_station_code}")
        if to_query_code:
            note_parts.append(f"到达站代码 {to_query_code}")
        if destination_hub and to_query != destination_city:
            note_parts.append(f"已按目的地附近枢纽 {to_query} 自动检索")
        if tickets:
            note_parts.append(f"查到 {direct_count} 趟直达")
        if transfers:
            note_parts.append(f"查到 {transfer_count} 条中转方案")
        top_ticket = tickets[0] if tickets else None
        if top_ticket:
            note_parts.append(
                f"推荐直达 {top_ticket['train_no']}，{top_ticket['start_time']} 出发，{top_ticket['duration']} 抵达"
            )
        note = "，".join(note_parts) + "。"
        status = "ready" if not warnings else "partial"
    elif warnings:
        note = "12306 已接入，但本次没有拿到完整票务结果，可能是服务限流、目标日期暂不可查或站点组合较少。"
        status = "error" if len(warnings) >= 2 else "partial"
    else:
        note = "12306 已接入，但未查到可用直达或中转结果，可能是当天无票、未开售或站点组合较少。"
        status = "empty"

    return {
        **transport,
        "provider": "12306-mcp",
        "status": status,
        "note": note,
        "tickets": tickets,
        "transfers": transfers,
        "warnings": warnings,
        "errors": errors,
    }


def _cost_level_from_avg_cost(avg_cost: float | None) -> int:
    if avg_cost is None:
        return 2
    if avg_cost <= 50:
        return 1
    if avg_cost >= 160:
        return 3
    return 2


def _normalize_amap_typecode(value: object) -> str:
    digits = re.sub(r"\D", "", str(value or ""))
    return digits[:6]


def _is_company_like_food_name(name: object) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    return any(token in text for token in FOOD_COMPANY_NAME_TOKENS)


def _is_valid_food_name(name: object) -> bool:
    text = str(name or "").strip()
    return bool(text) and not _is_company_like_food_name(text)


def _is_valid_food_row(row: dict) -> bool:
    name = str(row.get("name") or "").strip()
    if not _is_valid_food_name(name):
        return False
    type_text = str(row.get("type") or "").strip()
    type_code = _normalize_amap_typecode(row.get("typecode") or row.get("typeCode"))
    if type_code and not type_code.startswith("05"):
        return False
    if type_text and any(token in type_text for token in FOOD_TYPE_BLOCKLIST_TOKENS):
        if not any(token in type_text for token in FOOD_TYPE_HINT_TOKENS):
            return False
    return True


def _meal_period(current_time: time) -> str:
    windows = [
        ("breakfast", time(7, 0), time(9, 30)),
        ("lunch", time(11, 0), time(13, 30)),
        ("dinner", time(17, 0), time(19, 30)),
    ]
    for label, start, end in windows:
        if start <= current_time <= end:
            return label
    return ""


def _meal_period_zh(label: str) -> str:
    return {
        "breakfast": "早餐",
        "lunch": "午餐",
        "dinner": "晚餐",
    }.get(label, "")


def _time_label(value: time) -> str:
    return value.strftime("%H:%M")


def _time_from_minutes(minutes: int) -> time:
    normalized = max(0, min(23 * 60 + 59, int(minutes)))
    return time(normalized // 60, normalized % 60)


def _build_itinerary_slots(travel_date: date, earliest_clock_minutes: int | None = None) -> list[dict]:
    base_slots = [
        (time(9, 0), "attraction", ""),
        (time(11, 30), "attraction", ""),
        (time(12, 30), "food", "lunch"),
        (time(15, 0), "attraction", ""),
        (time(18, 30), "food", "dinner"),
        (time(20, 0), "attraction", ""),
    ]
    buffer_minutes = 45
    if earliest_clock_minutes is not None:
        start_after = max(0, int(earliest_clock_minutes) + buffer_minutes)
        slots = []
        for slot_time, kind, meal_key in base_slots:
            if _parse_clock_minutes(_time_label(slot_time)) <= start_after:
                continue
            slots.append(
                {
                    "time": _time_label(slot_time),
                    "kind": kind,
                    "meal_priority": kind == "food" and bool(meal_key),
                    "meal_key": meal_key,
                    "meal_label": _meal_period_zh(meal_key),
                }
            )
        if slots:
            return slots
        if start_after < 22 * 60 + 30:
            fallback_time = _time_from_minutes(min(22 * 60 + 30, start_after))
            meal_key = _meal_period(fallback_time)
            return [
                {
                    "time": _time_label(fallback_time),
                    "kind": "food" if meal_key else "attraction",
                    "meal_priority": bool(meal_key),
                    "meal_key": meal_key,
                    "meal_label": _meal_period_zh(meal_key),
                }
            ]
        return []
    now = _local_now()
    if travel_date != now.date():
        return [
            {
                "time": _time_label(slot_time),
                "kind": kind,
                "meal_priority": kind == "food" and bool(meal_key),
                "meal_key": meal_key,
                "meal_label": _meal_period_zh(meal_key),
            }
            for slot_time, kind, meal_key in base_slots
        ]

    now_time = now.time().replace(second=0, microsecond=0)
    meal_key = _meal_period(now_time)
    slots = [
        {
            "time": _time_label(now_time),
            "kind": "food" if meal_key else "attraction",
            "meal_priority": bool(meal_key),
            "meal_key": meal_key,
            "meal_label": _meal_period_zh(meal_key),
        }
    ]
    for slot_time, kind, meal_key in base_slots:
        if slot_time <= now_time:
            continue
        if kind == "food" and slots and slots[-1]["kind"] == "food":
            continue
        slots.append(
            {
                "time": _time_label(slot_time),
                "kind": kind,
                "meal_priority": kind == "food" and bool(meal_key),
                "meal_key": meal_key,
                "meal_label": _meal_period_zh(meal_key),
            }
        )
    return slots


def _day_title(day_date: date, day_index: int) -> str:
    return f"第{day_index}天 · {day_date.isoformat()} · {_week_label_zh(day_date.isoformat())}"


def _guess_city_by_coords(lat: float, lon: float) -> str:
    city_centers = [
        ("上海", 31.2304, 121.4737),
        ("北京", 39.9042, 116.4074),
        ("广州", 23.1291, 113.2644),
        ("深圳", 22.5431, 114.0579),
        ("成都", 30.5728, 104.0668),
        ("杭州", 30.2741, 120.1551),
        ("重庆", 29.5630, 106.5516),
        ("武汉", 30.5928, 114.3055),
        ("西安", 34.3416, 108.9398),
        ("南京", 32.0603, 118.7969),
    ]
    best_name = ""
    best_dist = 10**9
    for name, c_lat, c_lon in city_centers:
        d = _haversine_km(lat, lon, c_lat, c_lon)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name if best_dist <= 180 else ""


def _guess_admin_by_coords(lat: float, lon: float) -> dict:
    city_meta = [
        ("上海", "上海市", "上海市"),
        ("北京", "北京市", "北京市"),
        ("广州", "广东省", "广州市"),
        ("深圳", "广东省", "深圳市"),
        ("成都", "四川省", "成都市"),
        ("杭州", "浙江省", "杭州市"),
        ("重庆", "重庆市", "重庆市"),
        ("武汉", "湖北省", "武汉市"),
        ("西安", "陕西省", "西安市"),
        ("南京", "江苏省", "南京市"),
    ]
    guess_city = _guess_city_by_coords(lat, lon)
    for city, prov, c in city_meta:
        if city == guess_city:
            return {"province": prov, "city": c, "county": ""}
    return {"province": "", "city": "", "county": ""}


def _weather_condition_zh(weather_code: int | None, precip_mm: float, max_temp_c: float, min_temp_c: float) -> str:
    if weather_code is None:
        if precip_mm >= 1 and min_temp_c <= 0:
            return "雨夹雪"
        if precip_mm >= 1:
            return "雨"
        return "晴"
    mapping = {
        0: "晴",
        1: "少云",
        2: "多云",
        3: "阴",
        45: "雾",
        48: "雾凇",
        51: "毛毛雨",
        53: "毛毛雨",
        55: "毛毛雨",
        56: "冻毛毛雨",
        57: "冻毛毛雨",
        61: "小雨",
        63: "中雨",
        65: "大雨",
        66: "冻雨",
        67: "冻雨",
        71: "小雪",
        73: "中雪",
        75: "大雪",
        77: "雪粒",
        80: "阵雨",
        81: "阵雨",
        82: "强阵雨",
        85: "阵雪",
        86: "强阵雪",
        95: "雷雨",
        96: "雷雨冰雹",
        99: "强雷雨冰雹",
    }
    condition = mapping.get(weather_code)
    if condition:
        return condition
    if precip_mm >= 1 and min_temp_c <= 0:
        return "雨夹雪"
    if precip_mm >= 1:
        return "雨"
    if max_temp_c >= 32:
        return "晴热"
    return "晴"


def _weather_tone(max_temp_c: float | None) -> str:
    try:
        temp = float(max_temp_c) if max_temp_c is not None else 0.0
    except Exception:
        temp = 0.0
    if temp < 15:
        return "cool"
    if temp > 25:
        return "warm"
    return "mild"


def _clothing_tip(max_temp_c: float | None, min_temp_c: float | None, condition_zh: str = "", precip_mm: float | None = None) -> str:
    try:
        high = float(max_temp_c) if max_temp_c is not None else 26.0
    except Exception:
        high = 26.0
    try:
        low = float(min_temp_c) if min_temp_c is not None else 18.0
    except Exception:
        low = 18.0
    rain_like = "雨" in str(condition_zh or "") or float(precip_mm or 0) >= 1.0
    if high < 10:
        base = "建议穿厚外套"
    elif high < 15:
        base = "建议穿薄外套"
    elif high < 22 or low < 14:
        base = "建议穿长袖或薄衫"
    elif high > 29:
        base = "建议穿短袖并注意防晒"
    else:
        base = "建议穿轻薄上衣"
    return f"{base}，并备好雨具" if rain_like else base


def _windpower_to_mps(windpower_raw: str) -> float:
    levels = re.findall(r"\d+(?:\.\d+)?", str(windpower_raw or ""))
    if not levels:
        return 0.0
    # Rough conversion: Beaufort scale to m/s approximation.
    return round(max(0.0, float(levels[0]) * 1.5), 1)


def _week_label_zh(date_str: str = "", week_raw: str = "") -> str:
    week_map = {
        "0": "周日",
        "1": "周一",
        "2": "周二",
        "3": "周三",
        "4": "周四",
        "5": "周五",
        "6": "周六",
        "7": "周日",
    }
    if str(week_raw) in week_map:
        return week_map[str(week_raw)]
    try:
        idx = date.fromisoformat(date_str).weekday()
        return ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][idx]
    except Exception:
        return ""


def _day_label_zh(index: int, date_str: str = "", week_raw: str = "") -> str:
    if index == 0:
        return "今天"
    if index == 1:
        return "明天"
    if index == 2:
        return "后天"
    return _week_label_zh(date_str, week_raw) or f"D+{index}"


def _map_url(name: str, city: str, lat: float | None, lon: float | None) -> str:
    if lat is not None and lon is not None:
        return f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=16/{lat:.6f}/{lon:.6f}"
    keyword = quote(f"{city} {name}".strip())
    return f"https://uri.amap.com/search?keyword={keyword}&city={quote(city)}"


def _nav_url(from_lat: float | None, from_lon: float | None, to_lat: float | None, to_lon: float | None, name: str, city: str) -> str:
    if from_lat is not None and from_lon is not None and to_lat is not None and to_lon is not None:
        return (
            "https://uri.amap.com/navigation?"
            f"from={from_lon:.6f},{from_lat:.6f},当前位置&to={to_lon:.6f},{to_lat:.6f},{quote(name)}"
            "&mode=car&src=ai-travel-companion&coordinate=gaode&callnative=0"
        )
    return _map_url(name, city, to_lat, to_lon)


def _city_fallback_pois(city: str) -> tuple[list[Poi], list[Poi]]:
    c = city.lower()
    if "beijing" in c or "北京" in city:
        return (
            [
                Poi("故宫博物院", "museum", True, 2, ("culture", "history", "local"), 39.9163, 116.3972),
                Poi("颐和园", "landmark", False, 2, ("culture", "view", "local"), 39.9996, 116.2755),
                Poi("天坛公园", "park", False, 1, ("nature", "walking", "local"), 39.8822, 116.4065),
                Poi("景山公园", "park", False, 1, ("nature", "view", "walking"), 39.9320, 116.4039),
                Poi("国家博物馆", "museum", True, 2, ("culture", "history", "indoor"), 39.9050, 116.4013),
                Poi("奥林匹克塔", "landmark", False, 2, ("landmark", "view", "photo"), 39.9929, 116.3975),
            ],
            [
                Poi("北京烤鸭", "restaurant", True, 3, ("local", "savory"), 39.9042, 116.4074),
                Poi("炸酱面馆", "noodle", True, 1, ("local", "savory"), 39.9211, 116.4120),
                Poi("老北京小吃", "snack", True, 1, ("local", "street_food"), 39.9087, 116.3975),
                Poi("铜锅涮肉", "restaurant", True, 2, ("local", "savory"), 39.9155, 116.3889),
                Poi("豆汁焦圈铺", "snack", True, 1, ("local", "street_food"), 39.8874, 116.4067),
                Poi("京味点心铺", "dessert", True, 1, ("dessert", "local"), 39.9022, 116.4178),
            ],
        )
    if "shanghai" in c or "上海" in city:
        return (
            [
                Poi("外滩", "landmark", False, 1, ("view", "walking", "landmark"), 31.2400, 121.4900),
                Poi("豫园", "attraction", False, 2, ("culture", "local"), 31.2272, 121.4923),
                Poi("上海博物馆", "museum", True, 2, ("culture", "history", "local"), 31.2304, 121.4737),
                Poi("东方明珠", "landmark", True, 3, ("landmark", "view", "photo"), 31.2397, 121.4998),
                Poi("世纪公园", "park", False, 1, ("nature", "walking", "local"), 31.2215, 121.5443),
                Poi("上海植物园", "garden", False, 1, ("nature", "walking", "sunny_friendly"), 31.1446, 121.4462),
                Poi("徐汇滨江", "landmark", False, 1, ("landmark", "walking", "view"), 31.1837, 121.4547),
                Poi("共青森林公园", "park", False, 1, ("nature", "walking", "forest"), 31.3290, 121.5385),
                Poi("滴水湖", "park", False, 1, ("nature", "view", "walking"), 30.8987, 121.9295),
                Poi("辰山植物园", "garden", False, 1, ("nature", "walking", "sunny_friendly"), 31.0723, 121.1807),
            ],
            [
                Poi("生煎馆", "snack", True, 1, ("local", "street_food"), 31.2318, 121.4717),
                Poi("本帮菜馆", "restaurant", True, 2, ("local", "savory"), 31.2250, 121.4870),
                Poi("蟹粉面馆", "noodle", True, 2, ("local", "seafood"), 31.2200, 121.4800),
                Poi("葱油拌面馆", "noodle", True, 1, ("local", "savory"), 31.2301, 121.4822),
                Poi("小笼馆", "restaurant", True, 2, ("local", "savory"), 31.2332, 121.4744),
                Poi("海派甜品铺", "dessert", True, 2, ("dessert", "local"), 31.2284, 121.4892),
            ],
        )
    if "guangzhou" in c or "广州" in city:
        return (
            [
                Poi("陈家祠", "museum", True, 2, ("culture", "local", "indoor"), 23.1259, 113.2446),
                Poi("沙面", "district", False, 1, ("culture", "walking", "view"), 23.1085, 113.2387),
                Poi("广州塔", "landmark", True, 3, ("landmark", "view", "photo"), 23.1065, 113.3248),
                Poi("白云山", "park", False, 1, ("nature", "walking", "view"), 23.1849, 113.2958),
                Poi("海珠湿地公园", "park", False, 1, ("nature", "walking", "wetland"), 23.0662, 113.3643),
                Poi("广东省博物馆", "museum", True, 2, ("culture", "history", "indoor"), 23.1180, 113.3246),
            ],
            [
                Poi("早茶点心", "restaurant", True, 2, ("local", "street_food"), 23.1250, 113.2644),
                Poi("烧腊店", "restaurant", True, 1, ("local", "savory"), 23.1180, 113.2700),
                Poi("艇仔粥", "snack", True, 1, ("local", "healthy"), 23.1200, 113.2600),
                Poi("牛杂小吃", "snack", True, 1, ("local", "street_food"), 23.1277, 113.2656),
                Poi("云吞面馆", "noodle", True, 1, ("local", "savory"), 23.1302, 113.2611),
                Poi("双皮奶甜品", "dessert", True, 1, ("dessert", "local"), 23.1238, 113.2579),
            ],
        )
    if "chengdu" in c or "成都" in city:
        return (
            [
                Poi("宽窄巷子", "district", False, 1, ("culture", "local"), 30.6670, 104.0508),
                Poi("武侯祠", "museum", True, 2, ("history", "local", "indoor"), 30.6460, 104.0432),
                Poi("杜甫草堂", "museum", True, 2, ("culture", "local", "indoor"), 30.6622, 104.0284),
                Poi("青城山", "park", False, 1, ("nature", "walking", "view"), 30.9046, 103.5670),
                Poi("兴隆湖", "park", False, 1, ("nature", "walking", "view"), 30.4860, 104.1218),
                Poi("天府熊猫塔", "landmark", True, 2, ("landmark", "view", "photo"), 30.6631, 104.0945),
            ],
            [
                Poi("成都火锅", "restaurant", True, 2, ("local", "spicy"), 30.6590, 104.0660),
                Poi("串串香", "snack", True, 1, ("local", "spicy"), 30.6660, 104.0620),
                Poi("担担面", "noodle", True, 1, ("local", "spicy"), 30.6720, 104.0720),
                Poi("冒菜馆", "restaurant", True, 1, ("local", "spicy"), 30.6610, 104.0710),
                Poi("糖油果子摊", "snack", True, 1, ("local", "street_food"), 30.6655, 104.0599),
                Poi("盖碗茶铺", "dessert", True, 1, ("local", "dessert"), 30.6682, 104.0541),
            ],
        )
    return ([], [])


def fetch_weather(city: str, travel_date: date) -> dict:
    try:
        geo_data = _http_get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name": city, "count": 1, "language": "zh", "format": "json"},
        )
        results = geo_data.get("results") or []
        if not results:
            raise ValueError("city not found")
        loc = results[0]
        return fetch_weather_by_coords(float(loc["latitude"]), float(loc["longitude"]), travel_date, loc.get("name", city))
    except Exception:
        return {
            "city": city,
            "max_temp_c": 26.0,
            "min_temp_c": 18.0,
            "precipitation_mm": 0.0,
            "wind_speed_mps": 3.0,
            "source": "fallback",
        }


def fetch_weather_by_coords(lat: float, lon: float, travel_date: date, city_hint: str = "") -> dict:
    if AMAP.enabled:
        try:
            admin = AMAP.reverse_geocode(lat, lon)
            adcode = admin.get("adcode") or admin.get("city") or city_hint
            live = AMAP.weather(str(adcode), forecast=False)
            weather_text = str(live.get("weather") or "晴")
            temp = float(live.get("temperature") or 26)
            wind_mps = _windpower_to_mps(str(live.get("windpower") or "3"))
            return {
                "city": (live.get("city") or admin.get("city") or city_hint or "未知城市").replace("市", ""),
                "max_temp_c": temp,
                "min_temp_c": temp,
                "precipitation_mm": 0.0,
                "wind_speed_mps": wind_mps,
                "weather_code": None,
                "weather_condition_zh": weather_text,
                "source": "amap-weather",
            }
        except Exception:
            pass
    try:
        day = travel_date.isoformat()
        forecast_data = _http_get_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": day,
                "end_date": day,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weather_code",
                "timezone": "auto",
            },
        )
        daily = forecast_data.get("daily", {})
        weather_code_raw = (daily.get("weather_code") or [None])[0]
        weather_code = int(weather_code_raw) if weather_code_raw is not None else None
        max_temp = float((daily.get("temperature_2m_max") or [26])[0])
        min_temp = float((daily.get("temperature_2m_min") or [18])[0])
        precip = float((daily.get("precipitation_sum") or [0])[0])
        return {
            "city": city_hint or "未知城市",
            "max_temp_c": max_temp,
            "min_temp_c": min_temp,
            "precipitation_mm": precip,
            "wind_speed_mps": round(float((daily.get("wind_speed_10m_max") or [10])[0]) / 3.6, 1),
            "weather_code": weather_code,
            "weather_condition_zh": _weather_condition_zh(weather_code, precip, max_temp, min_temp),
            "source": "open-meteo",
        }
    except Exception:
        return {
            "city": city_hint or "未知城市",
            "max_temp_c": 26.0,
            "min_temp_c": 18.0,
            "precipitation_mm": 0.0,
            "wind_speed_mps": 3.0,
            "weather_code": None,
            "weather_condition_zh": "晴",
            "source": "fallback",
        }


def _fetch_amap_daily_forecast(adcode_or_city: str, days: int) -> list[dict]:
    if not AMAP.enabled or not adcode_or_city:
        return []
    try:
        forecast = AMAP.weather(str(adcode_or_city), forecast=True)
        casts = forecast.get("casts") or []
        rows: list[dict] = []
        for idx, cast in enumerate(casts[: max(1, min(3, days))]):
            max_temp = float(cast.get("daytemp") or 26)
            min_temp = float(cast.get("nighttemp") or 18)
            day_weather = str(cast.get("dayweather") or "")
            night_weather = str(cast.get("nightweather") or "")
            condition = day_weather if day_weather == night_weather or not night_weather else f"{day_weather}转{night_weather}"
            wind_candidates = [cast.get("daypower") or "", cast.get("nightpower") or ""]
            wind_mps = max([_windpower_to_mps(v) for v in wind_candidates] or [0.0])
            rows.append(
                {
                    "date": cast.get("date") or "",
                    "day_label": _day_label_zh(idx, str(cast.get("date") or ""), str(cast.get("week") or "")),
                    "week_label": _week_label_zh(str(cast.get("date") or ""), str(cast.get("week") or "")),
                    "weather_condition_zh": condition or "晴",
                    "max_temp_c": max_temp,
                    "min_temp_c": min_temp,
                    "precipitation_mm": None,
                    "wind_speed_mps": wind_mps,
                    "source": "amap-forecast",
                }
            )
        return rows
    except Exception:
        return []


def _fetch_open_meteo_daily_forecast(lat: float, lon: float, days: int) -> list[dict]:
    start_day = date.today()
    end_day = start_day + timedelta(days=max(0, days - 1))
    try:
        forecast_data = _http_get_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_day.isoformat(),
                "end_date": end_day.isoformat(),
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                "timezone": "auto",
            },
            timeout_sec=8.0,
        )
        daily = forecast_data.get("daily", {})
        dates = daily.get("time") or []
        rows: list[dict] = []
        for idx, date_str in enumerate(dates[:days]):
            max_temp = float((daily.get("temperature_2m_max") or [26])[idx])
            min_temp = float((daily.get("temperature_2m_min") or [18])[idx])
            precip = float((daily.get("precipitation_sum") or [0])[idx])
            wind_kmh = float((daily.get("wind_speed_10m_max") or [10])[idx])
            code_raw = (daily.get("weather_code") or [None])[idx]
            weather_code = int(code_raw) if code_raw is not None else None
            rows.append(
                {
                    "date": date_str,
                    "day_label": _day_label_zh(idx, date_str),
                    "week_label": _week_label_zh(date_str),
                    "weather_condition_zh": _weather_condition_zh(weather_code, precip, max_temp, min_temp),
                    "max_temp_c": max_temp,
                    "min_temp_c": min_temp,
                    "precipitation_mm": precip,
                    "wind_speed_mps": round(wind_kmh / 3.6, 1),
                    "source": "open-meteo",
                }
            )
        return rows
    except Exception:
        return []


def _resolve_weather_location(city: str = "", address: str = "", lat: float | None = None, lon: float | None = None) -> dict:
    query = address.strip() or city.strip()
    if lat is not None and lon is not None:
        admin = resolve_admin_from_coords(lat, lon)
        resolved_city = str(admin.get("city") or city or query or "").replace("市", "")
        return {
            "city": resolved_city,
            "lat": lat,
            "lon": lon,
            "adcode": admin.get("adcode") or "",
            "formatted_address": query or resolved_city,
            "province": admin.get("province") or "",
            "county": admin.get("county") or "",
        }

    if query:
        parsed = geocode_address(query, city)
        if parsed:
            return {
                "city": str(parsed.get("city") or city or query or "").replace("市", ""),
                "lat": parsed.get("lat"),
                "lon": parsed.get("lon"),
                "adcode": parsed.get("adcode") or "",
                "formatted_address": parsed.get("formatted_address") or query,
                "province": parsed.get("province") or "",
                "county": parsed.get("county") or "",
            }

    if city:
        return {
            "city": city.replace("市", ""),
            "lat": None,
            "lon": None,
            "adcode": "",
            "formatted_address": query or city,
            "province": "",
            "county": "",
        }
    return {}


def fetch_weather_forecast(city: str = "", address: str = "", days: int = 3, lat: float | None = None, lon: float | None = None) -> dict:
    target_days = 3
    location = _resolve_weather_location(city=city, address=address, lat=lat, lon=lon)
    if not location:
        raise ValueError("city or address is required")

    resolved_city = str(location.get("city") or city or address or "未知城市").replace("市", "")
    resolved_lat = location.get("lat")
    resolved_lon = location.get("lon")

    if resolved_lat is not None and resolved_lon is not None:
        summary = fetch_weather_by_coords(float(resolved_lat), float(resolved_lon), date.today(), resolved_city)
    else:
        summary = fetch_weather(resolved_city, date.today())

    forecast_rows: list[dict] = []
    forecast_source = ""

    if target_days == 3:
        forecast_rows = _fetch_amap_daily_forecast(str(location.get("adcode") or resolved_city), target_days)
        if forecast_rows:
            forecast_source = "amap-forecast"

    if not forecast_rows and resolved_lat is not None and resolved_lon is not None:
        forecast_rows = _fetch_open_meteo_daily_forecast(float(resolved_lat), float(resolved_lon), target_days)
        if forecast_rows:
            forecast_source = "open-meteo"

    if not forecast_rows:
        first_row = {
            "date": date.today().isoformat(),
            "day_label": "今天",
            "week_label": _week_label_zh(date.today().isoformat()),
            "weather_condition_zh": summary.get("weather_condition_zh") or "晴",
            "max_temp_c": float(summary.get("max_temp_c") or 26),
            "min_temp_c": float(summary.get("min_temp_c") or 18),
            "precipitation_mm": summary.get("precipitation_mm"),
            "wind_speed_mps": float(summary.get("wind_speed_mps") or 0),
            "source": summary.get("source") or "fallback",
        }
        forecast_rows = [first_row]
        while len(forecast_rows) < target_days:
            clone = dict(first_row)
            future_day = date.today() + timedelta(days=len(forecast_rows))
            clone["date"] = future_day.isoformat()
            clone["day_label"] = _day_label_zh(len(forecast_rows), clone["date"])
            clone["week_label"] = _week_label_zh(clone["date"])
            forecast_rows.append(clone)
        forecast_source = summary.get("source") or "fallback"

    label = weather_label(float(summary.get("max_temp_c") or 26), float(summary.get("precipitation_mm") or 0))
    return {
        "location": {
            "city": resolved_city,
            "lat": resolved_lat,
            "lon": resolved_lon,
            "adcode": location.get("adcode") or "",
            "formatted_address": location.get("formatted_address") or resolved_city,
            "province": location.get("province") or "",
            "county": location.get("county") or "",
        },
        "days": target_days,
        "available_day_options": [3],
        "summary": {
            "city": summary.get("city") or resolved_city,
            "date": date.today().isoformat(),
            "max_temp_c": float(summary.get("max_temp_c") or 26),
            "min_temp_c": float(summary.get("min_temp_c") or 18),
            "precipitation_mm": float(summary.get("precipitation_mm") or 0),
            "wind_speed_mps": float(summary.get("wind_speed_mps") or 0),
            "weather_label": label,
            "weather_label_zh": summary.get("weather_condition_zh") or weather_label_zh(label),
            "weather_condition_zh": summary.get("weather_condition_zh") or weather_label_zh(label),
            "tips": weather_tips(label),
            "source": summary.get("source") or "fallback",
        },
        "forecast_source": forecast_source or "fallback",
        "daily": forecast_rows[:target_days],
    }


def _build_trip_weather_rows(city: str, start_date: date, days: int, lat: float | None = None, lon: float | None = None) -> list[dict]:
    target_days = max(1, min(int(days or 1), 14))
    rows: list[dict] = []

    if lat is not None and lon is not None:
        try:
            end_day = start_date + timedelta(days=max(0, target_days - 1))
            forecast_data = _http_get_json(
                "https://api.open-meteo.com/v1/forecast",
                {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date.isoformat(),
                    "end_date": end_day.isoformat(),
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                    "timezone": "auto",
                },
                timeout_sec=8.0,
            )
            daily = forecast_data.get("daily", {})
            dates = daily.get("time") or []
            for idx, date_str in enumerate(dates[:target_days]):
                max_temp = float((daily.get("temperature_2m_max") or [26])[idx])
                min_temp = float((daily.get("temperature_2m_min") or [18])[idx])
                precip = float((daily.get("precipitation_sum") or [0])[idx])
                wind_kmh = float((daily.get("wind_speed_10m_max") or [10])[idx])
                code_raw = (daily.get("weather_code") or [None])[idx]
                weather_code = int(code_raw) if code_raw is not None else None
                day_date = date.fromisoformat(date_str)
                label = weather_label(max_temp, precip)
                condition = _weather_condition_zh(weather_code, precip, max_temp, min_temp)
                rows.append(
                    {
                        "day_index": idx + 1,
                        "day_label": _day_title(day_date, idx + 1),
                        "date": date_str,
                        "week_label": _week_label_zh(date_str),
                        "weather_label": label,
                        "weather_label_zh": condition or weather_label_zh(label),
                        "weather_condition_zh": condition or weather_label_zh(label),
                        "max_temp_c": max_temp,
                        "min_temp_c": min_temp,
                        "precipitation_mm": precip,
                        "wind_speed_mps": round(wind_kmh / 3.6, 1),
                        "temperature_tone": _weather_tone(max_temp),
                        "clothing_tip": _clothing_tip(max_temp, min_temp, condition, precip),
                        "source": "open-meteo-trip",
                    }
                )
        except Exception:
            rows = []

    if not rows:
        for idx in range(target_days):
            day_date = start_date + timedelta(days=idx)
            weather_row = (
                fetch_weather_by_coords(lat, lon, day_date, city)
                if lat is not None and lon is not None
                else fetch_weather(city, day_date)
            )
            max_temp = float(weather_row.get("max_temp_c") or 26)
            min_temp = float(weather_row.get("min_temp_c") or 18)
            precip = float(weather_row.get("precipitation_mm") or 0)
            label = weather_label(max_temp, precip)
            condition = weather_row.get("weather_condition_zh") or weather_label_zh(label)
            rows.append(
                {
                    "day_index": idx + 1,
                    "day_label": _day_title(day_date, idx + 1),
                    "date": day_date.isoformat(),
                    "week_label": _week_label_zh(day_date.isoformat()),
                    "weather_label": label,
                    "weather_label_zh": condition,
                    "weather_condition_zh": condition,
                    "max_temp_c": max_temp,
                    "min_temp_c": min_temp,
                    "precipitation_mm": precip,
                    "wind_speed_mps": float(weather_row.get("wind_speed_mps") or 0),
                    "temperature_tone": _weather_tone(max_temp),
                    "clothing_tip": _clothing_tip(max_temp, min_temp, str(condition), precip),
                    "source": weather_row.get("source") or "fallback",
                }
            )

    while rows and len(rows) < target_days:
        idx = len(rows)
        day_date = start_date + timedelta(days=idx)
        clone = dict(rows[-1])
        clone["day_index"] = idx + 1
        clone["day_label"] = _day_title(day_date, idx + 1)
        clone["date"] = day_date.isoformat()
        clone["week_label"] = _week_label_zh(clone["date"])
        rows.append(clone)

    return rows[:target_days]


def resolve_city_from_coords(lat: float, lon: float) -> str:
    if AMAP.enabled:
        try:
            a = AMAP.reverse_geocode(lat, lon)
            city = a.get("city") or a.get("county") or ""
            if city:
                return str(city).replace("市", "")
        except Exception:
            pass
    guessed = _guess_city_by_coords(lat, lon)
    if guessed:
        return guessed
    try:
        data = _http_get_json(
            "https://nominatim.openstreetmap.org/reverse",
            {
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 10,
                "addressdetails": 1,
                "accept-language": "zh-CN",
            },
            timeout_sec=5.0,
        )
        address = data.get("address", {})
        city = (
            address.get("city")
            or address.get("town")
            or address.get("county")
            or address.get("state_district")
            or address.get("state")
        )
        if city:
            return str(city)
    except Exception:
        pass
    try:
        data = _http_get_json(
            "https://api.bigdatacloud.net/data/reverse-geocode-client",
            {
                "latitude": lat,
                "longitude": lon,
                "localityLanguage": "zh",
            },
            timeout_sec=5.0,
        )
        city = data.get("city") or data.get("locality") or data.get("principalSubdivision")
        if city:
            return str(city)
    except Exception:
        pass
    return _guess_city_by_coords(lat, lon)


def resolve_admin_from_coords(lat: float, lon: float) -> dict:
    guessed = _guess_admin_by_coords(lat, lon)
    if AMAP.enabled:
        try:
            a = AMAP.reverse_geocode(lat, lon)
            return {
                "province": a.get("province", "") or guessed["province"],
                "city": (a.get("city", "") or guessed["city"]).replace("市", "") if (a.get("city") or guessed["city"]) else "",
                "county": a.get("county", "") or guessed["county"],
                "adcode": a.get("adcode", ""),
            }
        except Exception:
            pass
    try:
        data = _http_get_json(
            "https://nominatim.openstreetmap.org/reverse",
            {
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 10,
                "addressdetails": 1,
                "accept-language": "zh-CN",
            },
            timeout_sec=5.0,
        )
        address = data.get("address", {})
        province = address.get("state") or guessed["province"]
        city = (
            address.get("city")
            or address.get("town")
            or address.get("county")
            or address.get("state_district")
            or guessed["city"]
        )
        county = address.get("county") or ""
        return {"province": str(province or ""), "city": str(city or ""), "county": str(county or ""), "adcode": ""}
    except Exception:
        return {**guessed, "adcode": ""}


def geocode_address(address: str, city: str = "") -> dict | None:
    query = address.strip()
    city_hint = city.strip()
    if not query:
        return None

    if AMAP.enabled:
        try:
            item = AMAP.geocode(query, city_hint)
            if item:
                if item.get("city"):
                    item["city"] = str(item["city"]).replace("市", "")
                return item
        except Exception:
            pass

        try:
            items = AMAP.search_address(query, city=city_hint, limit=1)
            if items:
                item = dict(items[0])
                item["formatted_address"] = item.get("display_name") or query
                item["adcode"] = item.get("adcode") or ""
                if item.get("city"):
                    item["city"] = str(item["city"]).replace("市", "")
                return item
        except Exception:
            pass

    try:
        data = _http_get_json(
            "https://nominatim.openstreetmap.org/search",
            {
                "q": query,
                "format": "json",
                "limit": 1,
                "addressdetails": 1,
                "accept-language": "zh-CN",
                "countrycodes": "cn",
            },
            timeout_sec=6.0,
        )
        rows = data if isinstance(data, list) else []
        if rows:
            row = rows[0]
            address_data = row.get("address") or {}
            city_name = address_data.get("city") or address_data.get("town") or address_data.get("county") or address_data.get("state_district") or ""
            return {
                "name": row.get("name") or query,
                "display_name": row.get("display_name") or query,
                "formatted_address": row.get("display_name") or query,
                "province": address_data.get("state") or "",
                "city": str(city_name).replace("市", "") if city_name else "",
                "county": address_data.get("county") or address_data.get("suburb") or "",
                "lat": float(row.get("lat")),
                "lon": float(row.get("lon")),
                "adcode": "",
            }
    except Exception:
        pass

    try:
        data = _http_get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            {
                "name": query,
                "count": 1,
                "language": "zh",
                "format": "json",
            },
            timeout_sec=6.0,
        )
        results = data.get("results") or []
        if results:
            row = results[0]
            city_name = row.get("name") or query
            county = row.get("admin3") or row.get("admin2") or ""
            return {
                "name": city_name,
                "display_name": row.get("name") or query,
                "formatted_address": row.get("name") or query,
                "province": row.get("admin1") or "",
                "city": str(city_name).replace("市", ""),
                "county": county,
                "lat": float(row.get("latitude")),
                "lon": float(row.get("longitude")),
                "adcode": "",
            }
    except Exception:
        pass

    items = _fallback_search_addresses(query, limit=1, allow_default=False)
    if items:
        item = dict(items[0])
        item["formatted_address"] = item.get("display_name") or query
        item["adcode"] = item.get("adcode") or ""
        if item.get("city"):
            item["city"] = str(item["city"]).replace("市", "")
        return item
    return None


def search_addresses(query: str, limit: int = 10) -> list[dict]:
    q = query.strip()
    if not q:
        return []
    if AMAP.enabled:
        try:
            items = AMAP.search_address(q, limit=limit)
            if items:
                # normalize city name (remove trailing 市 for UI field)
                for it in items:
                    if it.get("city"):
                        it["city"] = str(it["city"]).replace("市", "")
                return items[:limit]
        except Exception:
            pass
    try:
        data = _http_get_json(
            "https://nominatim.openstreetmap.org/search",
            {
                "q": q,
                "format": "json",
                "limit": limit,
                "addressdetails": 1,
                "accept-language": "zh-CN",
                "countrycodes": "cn",
            },
            timeout_sec=6.0,
        )
    except Exception:
        return []

    results: list[dict] = []
    for row in data if isinstance(data, list) else []:
        address = row.get("address") or {}
        province = address.get("state") or ""
        city = address.get("city") or address.get("town") or address.get("county") or address.get("state_district") or ""
        county = address.get("county") or address.get("suburb") or ""
        display_name = row.get("display_name") or ""
        name = row.get("name") or (display_name.split(",")[0].strip() if display_name else "")
        lat = float(row.get("lat")) if row.get("lat") else None
        lon = float(row.get("lon")) if row.get("lon") else None
        if lat is None or lon is None:
            continue
        results.append(
            {
                "name": name,
                "display_name": display_name,
                "province": province,
                "city": city,
                "county": county,
                "lat": lat,
                "lon": lon,
            }
        )
    if results:
        return results
    return []


def search_pois(
    query: str,
    kind: str = "attraction",
    city: str = "",
    lat: float | None = None,
    lon: float | None = None,
    limit: int = 10,
) -> dict:
    q = query.strip()
    normalized_kind = "food" if str(kind).strip().lower() == "food" else "attraction"
    response = {
        "kind": normalized_kind,
        "query": q,
        "grouped": normalized_kind == "attraction",
        "classification_source": "fallback",
        "items": [],
        "groups": [],
    }
    if not q and lat is None and lon is None and not city:
        return response
    items: list[dict] = []

    if AMAP.enabled:
        rows: list[dict] = []
        type_hint = "050000" if normalized_kind == "food" else "110000"
        if lat is not None and lon is not None:
            radius = 15000 if normalized_kind == "food" else 30000
            try:
                rows = AMAP.nearby_pois(
                    lat,
                    lon,
                    keywords=q,
                    types=type_hint,
                    radius=radius,
                    offset=min(25, max(1, limit)),
                    page=1,
                )
            except Exception:
                rows = []
        if not rows:
            try:
                fallback_query = q or ("景点" if normalized_kind == "attraction" else "美食")
                rows = AMAP.text_pois(fallback_query, city=city, types=type_hint, offset=min(25, max(1, limit)), page=1)
            except Exception:
                rows = []

        seen: set[str] = set()
        resolved_city = city
        if not resolved_city and lat is not None and lon is not None:
            resolved_city = resolve_city_from_coords(lat, lon)

        for row in rows:
            name = str(row.get("name") or "").strip()
            location = str(row.get("location") or "").strip()
            if not name or "," not in location:
                continue
            if normalized_kind == "food" and not _is_valid_food_row(row):
                continue
            lon_s, lat_s = location.split(",", 1)
            key = f"{name.casefold()}|{lat_s}|{lon_s}"
            if key in seen:
                continue
            seen.add(key)
            poi_lat = float(lat_s)
            poi_lon = float(lon_s)
            distance_km = round(_haversine_km(lat, lon, poi_lat, poi_lon), 2) if lat is not None and lon is not None else None
            map_url = _map_url(name, resolved_city, poi_lat, poi_lon)
            nav_url = _nav_url(lat, lon, poi_lat, poi_lon, name, resolved_city)
            if normalized_kind == "food":
                type_text = str(row.get("type") or "")
                normalized_text = f"{name} {type_text} {row.get('tag') or ''}"
                biz_ext = row.get("biz_ext") or {}
                if not isinstance(biz_ext, dict):
                    biz_ext = {}
                avg_cost = _parse_float(biz_ext.get("cost") or row.get("cost"))
                rating = _parse_float(biz_ext.get("rating") or row.get("rating"))
                reviews = _parse_int(biz_ext.get("review_count") or row.get("review_count") or row.get("comment_num"))
                tags = ["restaurant"]
                if any(token in normalized_text for token in ("咖啡", "甜品", "蛋糕", "奶茶")):
                    tags.append("dessert")
                if any(token in normalized_text for token in ("火锅", "川菜", "湘菜", "冒菜", "串串", "麻辣", "烤鱼")):
                    tags.append("spicy")
                if any(token in normalized_text for token in ("海鲜", "鱼港", "海产")):
                    tags.append("seafood")
                items.append(
                    {
                        "name": name,
                        "category": "restaurant",
                        "theme_label": "美食搜索",
                        "distance_km": distance_km,
                        "rating": rating,
                        "reviews": reviews,
                        "avg_cost": avg_cost,
                        "est_cost": cost_level_text(_cost_level_from_avg_cost(avg_cost)),
                        "reason": "高德搜索结果",
                        "map_url": map_url,
                        "nav_url": nav_url,
                        "tags": list(dict.fromkeys(tags)),
                        "amap_type": type_text,
                    }
                )
            else:
                type_text = str(row.get("type") or "")
                type_code = _normalize_amap_typecode(row.get("typecode") or row.get("typeCode"))
                category, indoor, tags = _classify_attraction_source(name, type_text, type_text, type_code)
                theme_tags = _attraction_theme_tags(category, indoor, tags)
                display_category_key, display_category_label = _amap_display_category(type_code, name, type_text)
                biz_ext = row.get("biz_ext") or {}
                if not isinstance(biz_ext, dict):
                    biz_ext = {}
                rating = _parse_float(biz_ext.get("rating") or row.get("rating"))
                reviews = _parse_int(biz_ext.get("review_count") or row.get("review_count") or row.get("comment_num"))
                items.append(
                    {
                        "name": name,
                        "category": category,
                        "theme_key": theme_tags[0],
                        "theme_label": ATTRACTION_THEME_LABELS.get(theme_tags[0], ""),
                        "theme_tags": theme_tags,
                        "display_category_key": display_category_key,
                        "display_category_label": display_category_label,
                        "amap_typecode": type_code,
                        "amap_type": type_text,
                        "distance_km": distance_km,
                        "rating": rating,
                        "reviews": reviews,
                        "reason": "高德搜索结果",
                        "est_cost": "中",
                        "map_url": map_url,
                        "nav_url": nav_url,
                    }
                )
            if len(items) >= limit:
                break
        response["classification_source"] = "amap-typecode" if normalized_kind == "attraction" else "amap-search"
        response["items"] = items[:limit]
        response["groups"] = _group_search_items(items[:limit], normalized_kind)
        return response

    if normalized_kind == "food":
        pool = fetch_city_pois(city)[1] if city else []
        items = [
            {
                "name": poi.name,
                "category": poi.category,
                "theme_label": "本地搜索",
                "distance_km": round(_haversine_km(lat, lon, poi.lat, poi.lon), 2) if lat is not None and lon is not None and poi.lat is not None and poi.lon is not None else None,
                "rating": poi.rating,
                "reviews": poi.reviews,
                "avg_cost": poi.avg_cost,
                "est_cost": cost_level_text(poi.cost_level),
                "reason": "本地回退搜索结果",
                "map_url": _map_url(poi.name, city, poi.lat, poi.lon),
                "nav_url": _nav_url(lat, lon, poi.lat, poi.lon, poi.name, city),
            }
            for poi in pool
            if _is_valid_food_name(poi.name) and (not q or q.casefold() in poi.name.casefold())
        ][:limit]
        response["items"] = items
        return response

    pool = fetch_city_pois(city)[0] if city else []
    fallback_items: list[dict] = []
    for poi in pool:
        if q and q.casefold() not in poi.name.casefold():
            continue
        theme_tags = _attraction_theme_tags(poi.category, poi.indoor, poi.tags)
        display_category_key, display_category_label = _display_category_from_theme(poi.category, poi.indoor, poi.tags)
        fallback_items.append(
            {
                "name": poi.name,
                "category": poi.category,
                "theme_key": theme_tags[0],
                "theme_label": ATTRACTION_THEME_LABELS.get(theme_tags[0], ""),
                "theme_tags": theme_tags,
                "display_category_key": display_category_key,
                "display_category_label": display_category_label,
                "distance_km": round(_haversine_km(lat, lon, poi.lat, poi.lon), 2) if lat is not None and lon is not None and poi.lat is not None and poi.lon is not None else None,
                "reason": "本地回退搜索结果",
                "est_cost": cost_level_text(poi.cost_level),
                "map_url": _map_url(poi.name, city, poi.lat, poi.lon),
                "nav_url": _nav_url(lat, lon, poi.lat, poi.lon, poi.name, city),
            }
        )
    response["classification_source"] = "heuristic-theme"
    response["items"] = fallback_items[:limit]
    response["groups"] = _group_search_items(fallback_items[:limit], normalized_kind)
    return response


def _fallback_search_addresses(query: str, limit: int = 10, allow_default: bool = True) -> list[dict]:
    fallback = [
        {"province": "上海市", "city": "上海市", "county": "浦东新区", "lat": 31.2304, "lon": 121.4737},
        {"province": "北京市", "city": "北京市", "county": "朝阳区", "lat": 39.9042, "lon": 116.4074},
        {"province": "广东省", "city": "广州市", "county": "天河区", "lat": 23.1291, "lon": 113.2644},
        {"province": "广东省", "city": "深圳市", "county": "南山区", "lat": 22.5431, "lon": 114.0579},
        {"province": "浙江省", "city": "杭州市", "county": "西湖区", "lat": 30.2741, "lon": 120.1551},
        {"province": "江苏省", "city": "南京市", "county": "鼓楼区", "lat": 32.0603, "lon": 118.7969},
        {"province": "四川省", "city": "成都市", "county": "武侯区", "lat": 30.5728, "lon": 104.0668},
        {"province": "湖北省", "city": "武汉市", "county": "武昌区", "lat": 30.5928, "lon": 114.3055},
        {"province": "陕西省", "city": "西安市", "county": "雁塔区", "lat": 34.3416, "lon": 108.9398},
        {"province": "重庆市", "city": "重庆市", "county": "渝中区", "lat": 29.5630, "lon": 106.5516},
        {"province": "山东省", "city": "青岛市", "county": "市南区", "lat": 36.0671, "lon": 120.3826},
        {"province": "福建省", "city": "厦门市", "county": "思明区", "lat": 24.4798, "lon": 118.0894},
        {"province": "湖南省", "city": "长沙市", "county": "岳麓区", "lat": 28.2282, "lon": 112.9388},
        {"province": "辽宁省", "city": "大连市", "county": "中山区", "lat": 38.9140, "lon": 121.6147},
    ]
    q = query.strip().lower()
    matched = []
    for item in fallback:
        text = f"{item['province']}{item['city']}{item['county']}".lower()
        if q in text:
            matched.append(
                {
                    "name": item["county"],
                    "display_name": f"{item['province']}{item['city']}{item['county']}",
                    "province": item["province"],
                    "city": item["city"],
                    "county": item["county"],
                    "lat": item["lat"],
                    "lon": item["lon"],
                }
            )
    if matched:
        return matched[:limit]
    if not allow_default:
        return []
    return [
        {
            "name": item["county"],
            "display_name": f"{item['province']}{item['city']}{item['county']}",
            "province": item["province"],
            "city": item["city"],
            "county": item["county"],
            "lat": item["lat"],
            "lon": item["lon"],
        }
        for item in fallback[:limit]
    ]


def _sort_by_distance(items: list[dict], start_lat: float, start_lon: float) -> list[dict]:
    remaining = items[:]
    ordered: list[dict] = []
    cur_lat, cur_lon = start_lat, start_lon
    while remaining:
        best_idx = 0
        best_d = 10**9
        for i, item in enumerate(remaining):
            lat = item.get("lat")
            lon = item.get("lon")
            if lat is None or lon is None:
                d = 9999.0
            else:
                d = _haversine_km(cur_lat, cur_lon, lat, lon)
            if d < best_d:
                best_d = d
                best_idx = i
        picked = remaining.pop(best_idx)
        if picked.get("lat") is not None and picked.get("lon") is not None:
            cur_lat, cur_lon = picked["lat"], picked["lon"]
        ordered.append(picked)
    return ordered


def _normalize_name_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _normalize_attraction_style_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    alias_map = {
        "人文": "culture",
        "人文景点": "culture",
        "culture": "culture",
        "自然": "nature",
        "自然景观": "nature",
        "nature": "nature",
        "地标": "landmark",
        "城市地标": "landmark",
        "landmark": "landmark",
        "室内": "indoor",
        "室内场馆": "indoor",
        "indoor": "indoor",
    }
    for raw in values:
        text = str(raw or "").strip().lower()
        if not text:
            continue
        key = alias_map.get(text)
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_nature_name(name: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return False
    if _contains_any(text, ("文化广场", "人民广场", "市民广场", "时代广场", "商业广场", "购物广场")):
        return False
    strong_tokens = (
        "湿地",
        "森林",
        "植物园",
        "山谷",
        "峡谷",
        "瀑布",
        "草原",
        "花海",
        "绿道",
        "溪谷",
        "河谷",
        "海滩",
        "沙滩",
        "自然保护区",
        "地质公园",
        "国家公园",
        "山脉",
        "湖泊",
        "江景",
        "河景",
    )
    if _contains_any(text, strong_tokens):
        return True
    return text.endswith(("山", "湖", "江", "河", "海", "湾", "溪", "谷", "峡", "岛"))


def _display_category_from_theme(category: str, indoor: bool, tags: tuple[str, ...] | list[str] | None = None) -> tuple[str, str]:
    theme_tags = _attraction_theme_tags(category, indoor, tags)
    if "nature" in theme_tags:
        return ("nature", POI_DISPLAY_CATEGORY_LABELS["nature"])
    if "culture" in theme_tags or indoor:
        return ("culture", POI_DISPLAY_CATEGORY_LABELS["culture"])
    if "landmark" in theme_tags:
        return ("urban_leisure", POI_DISPLAY_CATEGORY_LABELS["urban_leisure"])
    return ("other", POI_DISPLAY_CATEGORY_LABELS["other"])


def _amap_display_category(type_code: str, name: str = "", type_text: str = "") -> tuple[str, str]:
    normalized_code = _normalize_amap_typecode(type_code)
    if normalized_code in AMAP_TYPECODE_DISPLAY_CATEGORY:
        return AMAP_TYPECODE_DISPLAY_CATEGORY[normalized_code]

    combined = f"{name} {type_text}".strip()
    if _contains_any(combined, ("纪念馆", "寺", "庙", "教堂", "故居", "遗址", "古建", "博物馆", "美术馆")):
        return ("culture", POI_DISPLAY_CATEGORY_LABELS["culture"])
    if _contains_any(combined, ("风景", "山", "湖", "湿地", "森林", "植物园", "动物园", "水族馆", "海滩", "瀑布")):
        return ("nature", POI_DISPLAY_CATEGORY_LABELS["nature"])
    if _contains_any(combined, ("公园", "广场", "乐园", "游乐", "步行街")):
        return ("urban_leisure", POI_DISPLAY_CATEGORY_LABELS["urban_leisure"])
    return ("other", POI_DISPLAY_CATEGORY_LABELS["other"])


def _group_search_items(items: list[dict], kind: str) -> list[dict]:
    if kind != "attraction" or not items:
        return []

    grouped: dict[str, dict] = {}
    for item in items:
        key = str(item.get("display_category_key") or "other")
        label = str(item.get("display_category_label") or POI_DISPLAY_CATEGORY_LABELS.get(key, POI_DISPLAY_CATEGORY_LABELS["other"]))
        bucket = grouped.setdefault(key, {"key": key, "label": label, "count": 0, "items": []})
        bucket["items"].append(item)
        bucket["count"] += 1

    ordered_groups: list[dict] = []
    for key in POI_DISPLAY_CATEGORY_ORDER:
        bucket = grouped.get(key)
        if bucket:
            ordered_groups.append(bucket)
    for key, bucket in grouped.items():
        if key not in POI_DISPLAY_CATEGORY_ORDER:
            ordered_groups.append(bucket)
    return ordered_groups


def _classify_attraction_source(name: str, category: str = "", type_text: str = "", type_code: str = "") -> tuple[str, bool, tuple[str, ...]]:
    combined = f"{name} {category} {type_text}".lower()
    name_text = str(name or "").strip()
    indoor_keywords = ("博物馆", "美术馆", "纪念馆", "展览馆", "科技馆", "海洋馆", "艺术馆", "图书馆", "天文馆", "剧院")
    culture_keywords = ("古镇", "老街", "古街", "故居", "遗址", "书院", "城墙", "寺", "庙", "祠", "宫", "文庙", "文化", "历史", "纪念")
    landmark_keywords = ("观景台", "塔", "桥", "门", "楼", "地标", "摩天轮", "外滩")
    square_keywords = ("广场",)
    display_category_key, _ = _amap_display_category(type_code, name_text, type_text)

    is_indoor = _contains_any(name_text, indoor_keywords) or _contains_any(combined, indoor_keywords)
    is_culture = _contains_any(name_text, culture_keywords) or _contains_any(combined, ("博物馆", "纪念馆", "展览馆", "美术馆", "艺术馆", "古迹", "古建筑"))
    square_like = _contains_any(name_text, square_keywords)
    nature_like = _is_nature_name(name_text) or (_contains_any(combined, ("湿地", "森林", "植物园", "自然风景", "风景名胜")) or ("公园" in combined and not square_like))
    landmark_like = _contains_any(name_text, landmark_keywords) or _contains_any(combined, ("观景台", "塔", "桥", "地标", "夜景")) or square_like

    if display_category_key == "culture":
        is_culture = True
    if display_category_key == "nature":
        nature_like = True
        landmark_like = False
    if display_category_key == "urban_leisure" and not is_culture and not is_indoor:
        landmark_like = True

    if is_indoor:
        base_category = "museum" if _contains_any(name_text + type_text, ("博物馆", "纪念馆", "科技馆", "海洋馆")) else "gallery"
        tags = ["culture", "indoor", "rainy_friendly", "local"]
        return base_category, True, tuple(dict.fromkeys(tags))

    if nature_like and not is_culture:
        base_category = "garden" if _contains_any(name_text + type_text, ("植物园", "花园")) else "park"
        tags = ["nature", "view", "walking", "sunny_friendly", "local"]
        return base_category, False, tuple(dict.fromkeys(tags))

    if landmark_like and not is_culture:
        tags = ["view", "photo", "local", "sunny_friendly"]
        return "landmark", False, tuple(dict.fromkeys(tags))

    tags = ["culture", "history", "local"]
    if is_culture:
        tags.append("sunny_friendly")
    return "district" if _contains_any(name_text + type_text, ("古镇", "老街", "街区", "广场")) else "attraction", False, tuple(dict.fromkeys(tags))


def _attraction_theme_tags(category: str, indoor: bool, tags: tuple[str, ...] | list[str] | None = None) -> list[str]:
    normalized_category = str(category or "").strip().lower()
    tag_set = {str(tag or "").strip().lower() for tag in (tags or []) if str(tag or "").strip()}
    theme_tags: list[str] = []

    if normalized_category in {"park", "garden"} or "nature" in tag_set:
        theme_tags.append("nature")
    if normalized_category in {"landmark", "viewpoint"} or (
        tag_set.intersection({"view", "photo", "night"}) and normalized_category not in {"park", "garden"} and "nature" not in tag_set
    ):
        theme_tags.append("landmark")
    if normalized_category in {"museum", "gallery", "district"} or tag_set.intersection({"culture", "history", "art"}):
        theme_tags.append("culture")
    if normalized_category == "attraction":
        if "nature" in tag_set:
            theme_tags.append("nature")
        elif tag_set.intersection({"view", "photo"}):
            theme_tags.append("landmark")
        else:
            theme_tags.append("culture")
    if indoor or normalized_category in {"museum", "gallery"}:
        theme_tags.append("indoor")

    deduped: list[str] = []
    seen: set[str] = set()
    for key in theme_tags:
        if key not in ATTRACTION_THEME_LABELS or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    if deduped:
        return deduped
    return ["culture" if indoor else "landmark"]


def _theme_match_count(item: dict, attraction_styles: list[str]) -> int:
    if not attraction_styles:
        return 0
    item_tags = {str(tag or "").strip() for tag in (item.get("theme_tags") or []) if str(tag or "").strip()}
    if item.get("theme_key"):
        item_tags.add(str(item["theme_key"]))
    return sum(1 for style in attraction_styles if style in item_tags)


def _sort_attractions_by_styles(items: list[dict], attraction_styles: list[str]) -> list[dict]:
    if not attraction_styles:
        return sorted(items, key=lambda x: (float(x.get("score") or 0), -(x.get("distance_km") or 9999)), reverse=True)
    return sorted(
        items,
        key=lambda x: (_theme_match_count(x, attraction_styles), float(x.get("score") or 0), -(x.get("distance_km") or 9999)),
        reverse=True,
    )


def _distance_bucket(distance_km: object) -> str:
    try:
        distance = float(distance_km)
    except Exception:
        return "unknown"
    if distance <= 1.0:
        return "within_1km"
    if distance <= 3.0:
        return "within_3km"
    if distance <= 8.0:
        return "within_8km"
    if distance <= 20.0:
        return "within_20km"
    return "beyond_20km"


def _spread_items_by_distance(items: list[dict], limit: int) -> list[dict]:
    if not items or limit <= 0:
        return []

    bucket_order = ("within_1km", "within_3km", "within_8km", "within_20km", "beyond_20km", "unknown")
    buckets: dict[str, list[dict]] = {bucket: [] for bucket in bucket_order}
    fallback: list[dict] = []

    for item in items:
        bucket = _distance_bucket(item.get("distance_km"))
        if bucket in buckets:
            buckets[bucket].append(item)
        else:
            fallback.append(item)

    selected: list[dict] = []
    selected_keys: set[str] = set()

    def push(entry: dict) -> None:
        key = f"{str(entry.get('name') or '').casefold()}|{entry.get('lat')}|{entry.get('lon')}"
        if key in selected_keys or len(selected) >= limit:
            return
        selected_keys.add(key)
        selected.append(entry)

    while len(selected) < limit:
        progressed = False
        for bucket in bucket_order:
            bucket_items = buckets[bucket]
            if not bucket_items:
                continue
            push(bucket_items.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    if len(selected) < limit:
        for item in items:
            push(item)
            if len(selected) >= limit:
                break

    if len(selected) < limit:
        for item in fallback:
            push(item)
            if len(selected) >= limit:
                break

    return selected[:limit]


def _count_items_within_distance(items: list[dict], max_distance_km: float) -> int:
    count = 0
    for item in items:
        try:
            distance = float(item.get("distance_km"))
        except Exception:
            continue
        if distance <= max_distance_km:
            count += 1
    return count


def _theme_counts(items: list[dict]) -> dict[str, int]:
    counts = {theme: 0 for theme in ATTRACTION_THEME_ORDER}
    for item in items:
        tags = {str(tag or "").strip() for tag in (item.get("theme_tags") or []) if str(tag or "").strip()}
        if item.get("theme_key"):
            tags.add(str(item["theme_key"]))
        for theme in ATTRACTION_THEME_ORDER:
            if theme in tags:
                counts[theme] += 1
    return counts


def _balance_attraction_items(
    items: list[dict],
    min_per_theme: int = ATTRACTION_TARGET_PER_THEME,
    limit: int = ATTRACTION_DISPLAY_LIMIT,
    priority_themes: list[str] | None = None,
) -> list[dict]:
    if not items:
        return []
    available = _theme_counts(items)
    target = {theme: min(min_per_theme, available.get(theme, 0)) for theme in ATTRACTION_THEME_ORDER}
    selected: list[dict] = []
    selected_keys: set[str] = set()
    achieved = {theme: 0 for theme in ATTRACTION_THEME_ORDER}
    theme_order = []
    for theme in (priority_themes or []) + ATTRACTION_THEME_ORDER:
        if theme in ATTRACTION_THEME_ORDER and theme not in theme_order:
            theme_order.append(theme)

    def item_tags(entry: dict) -> set[str]:
        tags = {str(tag or "").strip() for tag in (entry.get("theme_tags") or []) if str(tag or "").strip()}
        if entry.get("theme_key"):
            tags.add(str(entry["theme_key"]))
        return tags

    for theme in theme_order:
        if target[theme] <= 0:
            continue
        for item in items:
            key = str(item.get("name") or "")
            if key in selected_keys:
                continue
            tags = item_tags(item)
            if theme not in tags:
                continue
            selected.append(item)
            selected_keys.add(key)
            for tag in tags:
                if tag in achieved:
                    achieved[tag] += 1
            if achieved[theme] >= target[theme] or len(selected) >= limit:
                break
        if len(selected) >= limit:
            return selected[:limit]

    for item in items:
        key = str(item.get("name") or "")
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _filter_ranked_items(items: list[dict], selected_names: list[str], excluded_names: list[str]) -> list[dict]:
    selected_keys = {name.casefold() for name in selected_names}
    excluded_keys = {name.casefold() for name in excluded_names}
    filtered: list[dict] = []
    for item in items:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        key = name.casefold()
        if key in excluded_keys:
            continue
        if selected_keys and key not in selected_keys:
            continue
        filtered.append(item)
    return filtered


def _prioritize_ranked_items(items: list[dict], selected_names: list[str], excluded_names: list[str]) -> list[dict]:
    selected_keys = {name.casefold() for name in selected_names}
    excluded_keys = {name.casefold() for name in excluded_names}
    selected_items: list[dict] = []
    fallback_items: list[dict] = []
    seen_keys: set[str] = set()

    for item in items:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen_keys or key in excluded_keys:
            continue
        seen_keys.add(key)
        enriched = {**item, "user_selected": key in selected_keys}
        if enriched["user_selected"]:
            selected_items.append(enriched)
        else:
            fallback_items.append(enriched)

    return selected_items + fallback_items


def _estimate_plan_window(
    attraction_count: int,
    food_count: int,
    hotel_count: int = 0,
    has_transport_buffer: bool = False,
) -> tuple[int, float]:
    activity_count = attraction_count + food_count + hotel_count
    transfer_hours = max(0, activity_count - 1) * 0.45
    content_hours = attraction_count * 2.2 + food_count * 1.1 + hotel_count * 0.5
    if has_transport_buffer:
        transfer_hours += 3.5
    total_hours = round(content_hours + transfer_hours, 1)
    estimated_days = max(1, math.ceil(total_hours / 8.0)) if total_hours > 0 else 1
    return estimated_days, total_hours


def _merge_poi_lists(primary: list[Poi], secondary: list[Poi], limit: int = 24) -> list[Poi]:
    merged: list[Poi] = []
    seen: set[str] = set()
    for source in (primary, secondary):
        for poi in source:
            key = f"{poi.name.strip().lower()}|{round(poi.lat or 0.0, 4)}|{round(poi.lon or 0.0, 4)}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(poi)
            if len(merged) >= limit:
                return merged
    return merged


def _fetch_amap_pois(lat: float, lon: float) -> tuple[list[Poi], list[Poi]]:
    if not AMAP.enabled:
        return ([], [])
    try:
        attr_raw: list[dict] = []
        food_raw: list[dict] = []
        attr_keywords = ("旅游景点", "景点", "博物馆", "公园", "观景台")
        shortage_keywords = {
            "culture": ("纪念馆", "古镇", "老街"),
            "nature": ("湿地公园", "森林公园", "植物园"),
            "landmark": ("塔", "桥", "地标"),
            "indoor": ("美术馆", "科技馆", "海洋馆"),
        }
        food_keywords = ("美食", "餐厅", "地方菜", "川菜", "火锅", "小吃", "面馆", "咖啡")
        radii = (1000, 3000, 5000, 8000, 15000, 30000, 50000)

        for radius in radii:
            page_limit = 3 if radius <= 5000 else (2 if radius <= 15000 else 1)
            for keyword in attr_keywords:
                for page in range(1, page_limit + 1):
                    try:
                        attr_raw.extend(AMAP.nearby_pois(lat, lon, keywords=keyword, radius=radius, offset=25, page=page))
                    except Exception:
                        continue
            for keyword in food_keywords:
                for page in range(1, min(2, page_limit) + 1):
                    try:
                        food_raw.extend(AMAP.nearby_pois(lat, lon, keywords=keyword, radius=radius, offset=25, page=page))
                    except Exception:
                        continue

            temp_items: list[dict] = []
            temp_seen: set[str] = set()
            for row in attr_raw:
                name = row.get("name") or ""
                loc = row.get("location") or ""
                if not name or "," not in loc:
                    continue
                lon_s, lat_s = loc.split(",", 1)
                key = f"{name.strip().lower()}|{round(float(lat_s), 4)}|{round(float(lon_s), 4)}"
                if key in temp_seen:
                    continue
                temp_seen.add(key)
                category, indoor, tags = _classify_attraction_source(
                    name,
                    str(row.get("type") or ""),
                    str(row.get("type") or ""),
                    _normalize_amap_typecode(row.get("typecode") or row.get("typeCode")),
                )
                temp_items.append(
                    {
                        "theme_key": _attraction_theme_tags(category, indoor, tags)[0],
                        "theme_tags": _attraction_theme_tags(category, indoor, tags),
                    }
                )
            shortages = [theme for theme, count in _theme_counts(temp_items).items() if count < ATTRACTION_TARGET_PER_THEME]
            if shortages and radius < 50000:
                for theme in shortages:
                    for keyword in shortage_keywords.get(theme, ()):
                        try:
                            attr_raw.extend(AMAP.nearby_pois(lat, lon, keywords=keyword, radius=radius, offset=25, page=1))
                        except Exception:
                            continue

            if len(temp_items) >= ATTRACTION_FETCH_TARGET and len(food_raw) >= FOOD_FETCH_TARGET and radius >= 5000:
                break

        attractions: list[Poi] = []
        foods: list[Poi] = []
        seen_a: set[str] = set()
        seen_f: set[str] = set()

        for row in attr_raw:
            name = row.get("name") or ""
            loc = row.get("location") or ""
            if not name or "," not in loc:
                continue
            key = name.strip().lower()
            if key in seen_a:
                continue
            seen_a.add(key)
            lon_s, lat_s = loc.split(",", 1)
            type_text = str(row.get("type") or "")
            category, indoor, tags = _classify_attraction_source(
                name,
                type_text,
                type_text,
                _normalize_amap_typecode(row.get("typecode") or row.get("typeCode")),
            )
            attractions.append(
                Poi(
                    name=name,
                    category=category,
                    indoor=indoor,
                    cost_level=2,
                    tags=tags,
                    lat=float(lat_s),
                    lon=float(lon_s),
                )
            )

        for row in food_raw:
            name = row.get("name") or ""
            loc = row.get("location") or ""
            if not name or "," not in loc:
                continue
            if not _is_valid_food_row(row):
                continue
            key = name.strip().lower()
            if key in seen_f:
                continue
            seen_f.add(key)
            lon_s, lat_s = loc.split(",", 1)
            type_text = str(row.get("type") or "")
            normalized_text = f"{name} {type_text} {row.get('tag') or ''}"
            biz_ext = row.get("biz_ext") or {}
            if not isinstance(biz_ext, dict):
                biz_ext = {}
            avg_cost = _parse_float(biz_ext.get("cost") or row.get("cost"))
            rating = _parse_float(biz_ext.get("rating") or row.get("rating"))
            reviews = _parse_int(
                biz_ext.get("review_count")
                or biz_ext.get("comment_num")
                or row.get("review_count")
                or row.get("comment_num")
                or row.get("review_num")
            )
            tags = ["restaurant"]
            if any(token in normalized_text for token in ("咖啡", "甜品", "蛋糕", "奶茶")):
                tags.append("dessert")
            if any(token in normalized_text for token in ("火锅", "川菜", "湘菜", "冒菜", "串串", "麻辣", "烤鱼")):
                tags.append("spicy")
            if any(token in normalized_text for token in ("海鲜", "鱼港", "海产")):
                tags.append("seafood")
            if any(token in normalized_text for token in ("轻食", "沙拉", "素食", "养生", "粥")):
                tags.append("healthy")
            if any(token in normalized_text for token in ("面", "粉", "小吃", "烧烤", "锅贴", "饺子", "包子")):
                tags.append("savory")
            if any(token in normalized_text for token in ("快餐", "汉堡", "炸鸡")):
                tags.append("quick")
            if any(token in normalized_text for token in ("小吃", "夜市")):
                tags.append("street_food")
            if any(token in normalized_text for token in ("地方菜", "特色", "老字号", "本帮", "私房", "农家", "土菜")):
                tags.append("local")
            foods.append(
                Poi(
                    name=name,
                    category="restaurant",
                    indoor=True,
                    cost_level=_cost_level_from_avg_cost(avg_cost),
                    tags=tuple(dict.fromkeys(tags)),
                    lat=float(lat_s),
                    lon=float(lon_s),
                    rating=rating,
                    reviews=reviews,
                    avg_cost=avg_cost,
                    chain=is_chain_restaurant(name),
                )
            )
        return attractions, foods
    except Exception:
        return ([], [])


def _fetch_amap_hotels(lat: float, lon: float) -> list[Poi]:
    if not AMAP.enabled:
        return []
    try:
        hotel_raw: list[dict] = []
        for keyword in ("酒店", "宾馆", "民宿", "客栈"):
            try:
                hotel_raw.extend(AMAP.nearby_pois(lat, lon, keywords=keyword, radius=10000, offset=20))
            except Exception:
                continue

        hotels: list[Poi] = []
        seen: set[str] = set()
        for row in hotel_raw:
            name = row.get("name") or ""
            loc = row.get("location") or ""
            if not name or "," not in loc:
                continue
            lon_s, lat_s = loc.split(",", 1)
            key = f"{name.strip().lower()}|{round(float(lat_s), 4)}|{round(float(lon_s), 4)}"
            if key in seen:
                continue
            seen.add(key)
            biz_ext = row.get("biz_ext") or {}
            if not isinstance(biz_ext, dict):
                biz_ext = {}
            avg_cost = _parse_float(biz_ext.get("cost") or row.get("cost"))
            rating = _parse_float(biz_ext.get("rating") or row.get("rating"))
            reviews = _parse_int(
                biz_ext.get("review_count")
                or biz_ext.get("comment_num")
                or row.get("review_count")
                or row.get("comment_num")
                or row.get("review_num")
            )
            hotels.append(
                Poi(
                    name=name,
                    category="hotel",
                    indoor=True,
                    cost_level=_cost_level_from_avg_cost(avg_cost),
                    tags=("hotel",),
                    lat=float(lat_s),
                    lon=float(lon_s),
                    rating=rating,
                    reviews=reviews,
                    avg_cost=avg_cost,
                )
            )
        return hotels
    except Exception:
        return []


def _route_leg_metric(from_lat: float | None, from_lon: float | None, to_lat: float | None, to_lon: float | None) -> tuple[float | None, int | None]:
    if from_lat is None or from_lon is None or to_lat is None or to_lon is None:
        return (None, None)
    if AMAP.enabled:
        try:
            results = AMAP.distance_matrix(from_lat, from_lon, [(to_lat, to_lon)])
            if results:
                dist_m = float(results[0].get("distance") or 0)
                dur_s = float(results[0].get("duration") or 0)
                if dist_m > 0:
                    return (round(dist_m / 1000.0, 2), max(1, int(round(dur_s / 60.0))))
        except Exception:
            pass
    d = round(_haversine_km(from_lat, from_lon, to_lat, to_lon), 2)
    return (d, _travel_minutes_by_km(d))


def _pop_nearest_item(items: list[dict], from_lat: float | None, from_lon: float | None) -> dict | None:
    if not items:
        return None

    def item_key(index: int) -> tuple[int, float, float, str]:
        item = items[index]
        if from_lat is not None and from_lon is not None and item.get("lat") is not None and item.get("lon") is not None:
            distance = _haversine_km(from_lat, from_lon, float(item["lat"]), float(item["lon"]))
        else:
            raw_distance = item.get("distance_km")
            distance = float(raw_distance) if raw_distance is not None else 9999.0
        return (0 if item.get("user_selected") else 1, distance, -float(item.get("score") or 0), str(item.get("name") or ""))

    best_index = min(range(len(items)), key=item_key)
    return items.pop(best_index)


def _food_meal_penalty(item: dict, meal_key: str = "") -> int:
    tags = {str(tag or "").strip().lower() for tag in (item.get("tags") or []) if str(tag or "").strip()}
    if not meal_key:
        return 0
    if meal_key in tags:
        return 0
    if meal_key == "lunch":
        if {"quick", "healthy", "street_food"} & tags:
            return 1
        if {"dessert", "afternoon"} & tags:
            return 3
        return 2
    if meal_key == "dinner":
        if {"savory", "seafood", "spicy", "local"} & tags:
            return 1
        if {"dessert", "afternoon", "quick"} & tags:
            return 3
        return 2
    if meal_key == "breakfast":
        if {"quick", "street_food", "healthy"} & tags:
            return 1
        if {"seafood", "spicy", "dessert"} & tags:
            return 3
        return 2
    return 0


def _pop_food_item(items: list[dict], from_lat: float | None, from_lon: float | None, meal_key: str = "") -> dict | None:
    if not items:
        return None

    def item_key(index: int) -> tuple[int, int, float, float, str]:
        item = items[index]
        if from_lat is not None and from_lon is not None and item.get("lat") is not None and item.get("lon") is not None:
            distance = _haversine_km(from_lat, from_lon, float(item["lat"]), float(item["lon"]))
        else:
            raw_distance = item.get("distance_km")
            distance = float(raw_distance) if raw_distance is not None else 9999.0
        return (
            _food_meal_penalty(item, meal_key),
            0 if item.get("user_selected") else 1,
            distance,
            -float(item.get("score") or 0),
            str(item.get("name") or ""),
        )

    best_index = min(range(len(items)), key=item_key)
    return items.pop(best_index)


def build_plan(payload: dict, include_itinerary: bool = True) -> dict:
    city = str(payload.get("city", "")).strip()
    travel_date_raw = payload.get("travel_date")
    if not travel_date_raw:
        raise ValueError("travel_date is required")
    travel_date = date.fromisoformat(str(travel_date_raw))

    days = int(payload.get("days", 1))
    if days < 1 or days > 14:
        raise ValueError("days must be between 1 and 14")

    planning_mode = str(payload.get("planning_mode", "auto")).strip().lower()
    if planning_mode not in {"auto", "candidate"}:
        planning_mode = "auto"

    booking_date_raw = str(payload.get("booking_date", "") or "").strip()
    booking_date = ""
    if booking_date_raw:
        booking_date = date.fromisoformat(booking_date_raw).isoformat()

    origin_city = str(payload.get("origin_city", "") or "").strip()
    origin_station = str(payload.get("origin_station", "") or "").strip()
    origin_station_code = str(payload.get("origin_station_code", "") or "").strip().upper()

    budget_level = str(payload.get("budget_level", "standard"))
    if budget_level not in {"economy", "standard", "premium"}:
        budget_level = "standard"

    attraction_styles = _normalize_attraction_style_list(payload.get("attraction_styles"))
    food_preferences = payload.get("food_preferences", [])
    if not isinstance(food_preferences, list):
        food_preferences = []

    include_hotel = bool(payload.get("include_hotel", False))
    hotel_price_max_raw = payload.get("hotel_price_max", 500)
    try:
        hotel_price_max = max(100, min(5000, int(hotel_price_max_raw)))
    except Exception:
        hotel_price_max = 500
    selected_hotel_name = str(payload.get("selected_hotel_name", "") or "").strip()
    selected_transport = payload.get("selected_transport") or {}
    arrival_selection = _selected_transport_arrival(selected_transport)

    selected_attractions = _normalize_name_list(payload.get("selected_attractions"))
    excluded_attractions = _normalize_name_list(payload.get("excluded_attractions"))
    selected_foods = _normalize_name_list(payload.get("selected_foods"))
    excluded_foods = _normalize_name_list(payload.get("excluded_foods"))

    live_poi = bool(payload.get("live_poi", True))
    cur_lat = payload.get("location_lat")
    cur_lon = payload.get("location_lon")
    try:
        cur_lat = float(cur_lat) if cur_lat is not None else None
        cur_lon = float(cur_lon) if cur_lon is not None else None
    except Exception:
        cur_lat = None
        cur_lon = None

    if city and (cur_lat is None or cur_lon is None):
        parsed_location = geocode_address(city)
        if parsed_location:
            cur_lat = parsed_location.get("lat")
            cur_lon = parsed_location.get("lon")
            parsed_city = str(parsed_location.get("city") or "").replace("市", "")
            if parsed_city:
                city = parsed_city

    if not city and cur_lat is None:
        raise ValueError("city or location is required")

    if not city and cur_lat is not None and cur_lon is not None:
        city = resolve_city_from_coords(cur_lat, cur_lon) or _guess_city_by_coords(cur_lat, cur_lon) or "未知城市"

    if cur_lat is not None and cur_lon is not None:
        weather_raw = fetch_weather_by_coords(cur_lat, cur_lon, travel_date, city)
    else:
        weather_raw = fetch_weather(city, travel_date)

    label = weather_label(weather_raw["max_temp_c"], weather_raw["precipitation_mm"])

    live_attractions: list[Poi] = []
    live_foods: list[Poi] = []
    search_attractions: list[Poi] = []
    search_foods: list[Poi] = []

    if live_poi and cur_lat is not None and cur_lon is not None:
        live_attractions, live_foods = _fetch_amap_pois(cur_lat, cur_lon)
        if len(live_attractions) < 6 or len(live_foods) < 6:
            search_attractions, search_foods = fetch_nearby_pois(cur_lat, cur_lon, city)
        if city and (len(live_attractions) + len(search_attractions) < 6 or len(live_foods) + len(search_foods) < 6):
            city_search_attractions, city_search_foods = fetch_city_pois(city)
            search_attractions = _merge_poi_lists(search_attractions, city_search_attractions, limit=24)
            search_foods = _merge_poi_lists(search_foods, city_search_foods, limit=24)
    elif live_poi:
        search_attractions, search_foods = fetch_city_pois(city)
    else:
        search_attractions, search_foods = _city_fallback_pois(city)

    attraction_pool = _merge_poi_lists(live_attractions, search_attractions, limit=120)
    food_pool = [poi for poi in _merge_poi_lists(live_foods, search_foods, limit=100) if _is_valid_food_name(poi.name)]
    hotel_pool = _fetch_amap_hotels(cur_lat, cur_lon) if include_hotel and cur_lat is not None and cur_lon is not None else []

    scored_attractions = []
    for poi in attraction_pool:
        score, reason = score_attraction(poi, label, budget_level)
        theme_tags = _attraction_theme_tags(poi.category, poi.indoor, poi.tags)
        primary_theme = theme_tags[0]
        distance_km = (
            round(_haversine_km(cur_lat, cur_lon, poi.lat, poi.lon), 2)
            if cur_lat is not None and cur_lon is not None and poi.lat is not None and poi.lon is not None
            else None
        )
        scored_attractions.append(
            {
                "name": poi.name,
                "category": poi.category,
                "score": score,
                "reason": reason,
                "indoor": poi.indoor,
                "est_cost": cost_level_text(poi.cost_level),
                "theme_key": primary_theme,
                "theme_label": ATTRACTION_THEME_LABELS.get(primary_theme, ""),
                "theme_tags": theme_tags,
                "lat": poi.lat,
                "lon": poi.lon,
                "distance_km": distance_km,
                "rating": poi.rating,
                "reviews": poi.reviews,
                "avg_cost": poi.avg_cost,
                "map_url": _map_url(poi.name, city, poi.lat, poi.lon),
                "nav_url": _nav_url(cur_lat, cur_lon, poi.lat, poi.lon, poi.name, city),
            }
        )
    scored_attractions = _sort_attractions_by_styles(scored_attractions, attraction_styles)
    balanced_attractions = _spread_items_by_distance(
        _balance_attraction_items(scored_attractions, priority_themes=attraction_styles),
        limit=ATTRACTION_DISPLAY_LIMIT,
    )

    scored_foods = []
    for poi in food_pool:
        score, reason = score_food(poi, budget_level, food_preferences)
        distance_km = (
            round(_haversine_km(cur_lat, cur_lon, poi.lat, poi.lon), 2)
            if cur_lat is not None and cur_lon is not None and poi.lat is not None and poi.lon is not None
            else None
        )
        scored_foods.append(
            {
                "name": poi.name,
                "category": poi.category,
                "score": score,
                "reason": reason,
                "indoor": poi.indoor,
                "est_cost": cost_level_text(poi.cost_level),
                "lat": poi.lat,
                "lon": poi.lon,
                "distance_km": distance_km,
                "rating": poi.rating,
                "reviews": poi.reviews,
                "avg_cost": poi.avg_cost,
                "chain": poi.chain,
                "tags": list(poi.tags),
                "map_url": _map_url(poi.name, city, poi.lat, poi.lon),
                "nav_url": _nav_url(cur_lat, cur_lon, poi.lat, poi.lon, poi.name, city),
            }
        )
    scored_foods.sort(key=lambda x: (x["score"], -(x["distance_km"] or 9999)), reverse=True)
    display_foods = _spread_items_by_distance(scored_foods, limit=FOOD_DISPLAY_LIMIT)

    scored_hotels = []
    for poi in hotel_pool:
        score, reason = score_hotel(poi, hotel_price_max)
        distance_km = (
            round(_haversine_km(cur_lat, cur_lon, poi.lat, poi.lon), 2)
            if cur_lat is not None and cur_lon is not None and poi.lat is not None and poi.lon is not None
            else None
        )
        scored_hotels.append(
            {
                "name": poi.name,
                "category": poi.category,
                "score": score,
                "reason": reason,
                "indoor": poi.indoor,
                "est_cost": cost_level_text(poi.cost_level),
                "lat": poi.lat,
                "lon": poi.lon,
                "distance_km": distance_km,
                "rating": poi.rating,
                "reviews": poi.reviews,
                "avg_cost": poi.avg_cost,
                "map_url": _map_url(poi.name, city, poi.lat, poi.lon),
                "nav_url": _nav_url(cur_lat, cur_lon, poi.lat, poi.lon, poi.name, city),
            }
        )
    scored_hotels.sort(key=lambda x: (x["score"], -(x["distance_km"] or 9999)), reverse=True)

    nearby_attraction_count = _count_items_within_distance(balanced_attractions, RECOMMENDATION_FULL_RADIUS_KM)
    nearby_food_count = _count_items_within_distance(display_foods, RECOMMENDATION_FULL_RADIUS_KM)
    candidate_attraction_limit = min(len(balanced_attractions), max(30, days * 12, nearby_attraction_count))
    candidate_food_limit = min(len(display_foods), max(20, days * 8, nearby_food_count))
    candidate_attractions = balanced_attractions[:candidate_attraction_limit]
    candidate_foods = display_foods[:candidate_food_limit]

    selection_active = bool(selected_attractions or excluded_attractions or selected_foods or excluded_foods)
    matched_selected_attractions = _filter_ranked_items(scored_attractions, selected_attractions, [])
    matched_selected_foods = _filter_ranked_items(display_foods, selected_foods, [])
    prioritized_attractions = _prioritize_ranked_items(scored_attractions, selected_attractions, excluded_attractions)
    prioritized_foods = _prioritize_ranked_items(display_foods, selected_foods, excluded_foods)

    top_attractions = prioritized_attractions[: max(10, days * 6, len(matched_selected_attractions) + 4)]
    top_foods = prioritized_foods[: max(10, days * 5, len(matched_selected_foods) + 2)]
    selected_hotel = None
    if include_hotel and scored_hotels and selected_hotel_name:
        selected_hotel = next(
            (hotel for hotel in scored_hotels if str(hotel.get("name") or "").strip().casefold() == selected_hotel_name.casefold()),
            None,
        )
    if selected_hotel and selected_hotel.get("lat") is not None and selected_hotel.get("lon") is not None:
        top_attractions = _sort_by_distance(top_attractions, float(selected_hotel["lat"]), float(selected_hotel["lon"]))
        top_foods = _sort_by_distance(top_foods, float(selected_hotel["lat"]), float(selected_hotel["lon"]))
    hotel_candidates = scored_hotels[:]
    if selected_hotel:
        hotel_candidates = [selected_hotel] + [
            hotel for hotel in hotel_candidates
            if str(hotel.get("name") or "").strip().casefold() != str(selected_hotel.get("name") or "").strip().casefold()
        ]
    top_hotels = hotel_candidates[:10]

    budget_hint = {
        "economy": "优先低成本项目，建议公共交通+人均80元以内餐食。",
        "standard": "推荐中等预算组合，建议人均80-200元并搭配地铁/打车。",
        "premium": "可优先高评分体验，建议预约餐厅并预留弹性时间。",
    }[budget_level]

    estimate_attraction_count = len(matched_selected_attractions) if selected_attractions else min(len(top_attractions), max(3, days * 3))
    estimate_food_count = len(matched_selected_foods) if selected_foods else min(len(top_foods), max(2, days * 2))
    estimated_days, estimated_hours = _estimate_plan_window(
        attraction_count=estimate_attraction_count,
        food_count=estimate_food_count,
        hotel_count=1 if include_hotel and selected_hotel else 0,
        has_transport_buffer=bool(origin_city or origin_station),
    )
    itinerary: list[dict] = []
    daily_itinerary: list[dict] = []
    if include_itinerary:
        remaining_attractions = [{"kind": "attraction", **x} for x in top_attractions]
        remaining_foods = [{"kind": "food", **x} for x in top_foods]

        def _append_day_item(
            day_items: list[dict],
            *,
            day_index: int,
            day_date: date,
            item_time: str,
            activity_type: str,
            name: object,
            reason: object,
            lat: float | None = None,
            lon: float | None = None,
            theme_label: str = "",
            avg_cost: object = None,
            est_cost: str = "",
            score: object = None,
            rating: object = None,
            distance_km: float | None = None,
            travel_minutes: int | None = None,
            map_url: str = "",
            nav_url: str = "",
            extra: dict | None = None,
        ) -> dict:
            route_item = {
                "day_index": day_index,
                "day_label": _day_title(day_date, day_index),
                "date": day_date.isoformat(),
                "time": item_time,
                "activity_type": activity_type,
                "name": name,
                "reason": reason,
                "lat": lat,
                "lon": lon,
                "theme_label": theme_label,
                "avg_cost": avg_cost,
                "est_cost": est_cost,
                "score": score,
                "rating": rating,
                "distance_km": distance_km,
                "travel_minutes": travel_minutes,
                "map_url": map_url,
                "nav_url": nav_url,
            }
            if extra:
                route_item.update(extra)
            itinerary.append(route_item)
            day_items.append(route_item)
            return route_item

        base_start_lat = cur_lat
        base_start_lon = cur_lon
        for day_index in range(1, days + 1):
            day_date = travel_date + timedelta(days=day_index - 1)
            day_items: list[dict] = []
            hotel_has_coords = bool(selected_hotel and selected_hotel.get("lat") is not None and selected_hotel.get("lon") is not None)
            start_from_hotel = bool(selected_hotel and hotel_has_coords and (day_index > 1 or not arrival_selection))
            prev_lat = float(selected_hotel["lat"]) if start_from_hotel else base_start_lat
            prev_lon = float(selected_hotel["lon"]) if start_from_hotel else base_start_lon
            earliest_clock_minutes = None
            inserted_checkin = False

            if day_index == 1 and arrival_selection:
                arrival_label = str(arrival_selection.get("label") or "").strip()
                arrival_reason = "首日路线已严格对齐你添加的车次到达时间，并从到达后开始展开。"
                if arrival_label:
                    arrival_reason = f"{arrival_reason} 当前车次：{arrival_label}。"
                if arrival_selection.get("start_time") and arrival_selection.get("arrive_time"):
                    arrival_reason = (
                        f"{arrival_reason} 出发 {arrival_selection.get('start_time')}，"
                        f"到达 {arrival_selection.get('arrive_time')}。"
                    )
                arrival_lat = base_start_lat
                arrival_lon = base_start_lon
                if arrival_lat is None or arrival_lon is None:
                    arrival_lat = float(selected_hotel["lat"]) if hotel_has_coords else None
                    arrival_lon = float(selected_hotel["lon"]) if hotel_has_coords else None
                _append_day_item(
                    day_items,
                    day_index=day_index,
                    day_date=day_date,
                    item_time=str(arrival_selection.get("arrive_time") or ""),
                    activity_type="arrival",
                    name=f"抵达{city}",
                    reason=arrival_reason,
                    lat=arrival_lat,
                    lon=arrival_lon,
                    extra={
                        "transport_kind": arrival_selection.get("kind") or "",
                        "transport_label": arrival_label,
                        "transport_start_time": arrival_selection.get("start_time") or "",
                        "transport_arrive_time": arrival_selection.get("arrive_time") or "",
                    },
                )
                if arrival_lat is not None and arrival_lon is not None:
                    prev_lat, prev_lon = arrival_lat, arrival_lon
                earliest_clock_minutes = int(arrival_selection.get("arrival_minutes") or 0)

                if selected_hotel:
                    hotel_distance, hotel_minutes = _route_leg_metric(
                        prev_lat,
                        prev_lon,
                        selected_hotel.get("lat"),
                        selected_hotel.get("lon"),
                    )
                    checkin_minutes = min(22 * 60 + 30, max(0, earliest_clock_minutes) + 45)
                    hotel_reason = str(selected_hotel.get("reason") or "已按你确认的酒店作为住宿中心。").strip()
                    hotel_reason = f"到达后先前往酒店办理入住，后续节点会围绕这家酒店展开；{hotel_reason}"
                    if hotel_distance is not None and hotel_minutes is not None:
                        hotel_reason = f"{hotel_reason}；距到达点约{hotel_distance}km，预计{hotel_minutes}分钟"
                    _append_day_item(
                        day_items,
                        day_index=day_index,
                        day_date=day_date,
                        item_time=_time_label(_time_from_minutes(checkin_minutes)),
                        activity_type="hotel",
                        name=selected_hotel.get("name"),
                        reason=hotel_reason,
                        lat=selected_hotel.get("lat"),
                        lon=selected_hotel.get("lon"),
                        avg_cost=selected_hotel.get("avg_cost"),
                        est_cost=selected_hotel.get("est_cost", ""),
                        score=selected_hotel.get("score"),
                        rating=selected_hotel.get("rating"),
                        distance_km=hotel_distance,
                        travel_minutes=hotel_minutes,
                        map_url=selected_hotel.get("map_url", ""),
                        nav_url=selected_hotel.get("nav_url", ""),
                        extra={"hotel_selected": True, "hotel_stage": "checkin"},
                    )
                    if hotel_has_coords:
                        prev_lat = float(selected_hotel["lat"])
                        prev_lon = float(selected_hotel["lon"])
                        base_start_lat = prev_lat
                        base_start_lon = prev_lon
                    earliest_clock_minutes = checkin_minutes
                    inserted_checkin = True

            time_slots = (
                _build_itinerary_slots(day_date, earliest_clock_minutes=earliest_clock_minutes)
                if day_index == 1 and earliest_clock_minutes is not None
                else _build_itinerary_slots(day_date)
            )

            for slot in time_slots:
                if not remaining_attractions and not remaining_foods:
                    break
                preferred_kind = str(slot.get("kind") or "attraction")
                if preferred_kind == "food":
                    item = _pop_food_item(remaining_foods, prev_lat, prev_lon, str(slot.get("meal_key") or ""))
                else:
                    item = _pop_nearest_item(remaining_attractions, prev_lat, prev_lon)
                if item is None:
                    continue
                distance_from_prev, travel_min = _route_leg_metric(prev_lat, prev_lon, item.get("lat"), item.get("lon"))
                if item.get("lat") is not None and item.get("lon") is not None:
                    prev_lat, prev_lon = item["lat"], item["lon"]
                reason = item["reason"]
                if selection_active and not item.get("user_selected"):
                    if item["kind"] == "food":
                        reason = f"你没有明确选择这家餐饮，系统从未排除的候选里按饭点、距离和评分自动补充；{reason}"
                    else:
                        reason = f"你没有明确选择这个景点，系统从未排除的候选里按路线顺序、距离和评分自动补充；{reason}"
                if slot.get("meal_priority") and item["kind"] == "food":
                    reason = f"当前处于{slot.get('meal_label') or '用餐'}时段，优先安排用餐；{reason}"
                if distance_from_prev is not None and travel_min is not None:
                    reason = f"{reason}；距上一站约{distance_from_prev}km，预计{travel_min}分钟"
                _append_day_item(
                    day_items,
                    day_index=day_index,
                    day_date=day_date,
                    item_time=slot.get("time") or "",
                    activity_type=item["kind"],
                    name=item["name"],
                    reason=reason,
                    lat=item.get("lat"),
                    lon=item.get("lon"),
                    theme_label=item.get("theme_label", ""),
                    avg_cost=item.get("avg_cost"),
                    est_cost=item.get("est_cost", ""),
                    score=item.get("score"),
                    rating=item.get("rating"),
                    distance_km=distance_from_prev,
                    travel_minutes=travel_min,
                    map_url=item.get("map_url", ""),
                    nav_url=item.get("nav_url", ""),
                )
            has_route_content = any(item.get("activity_type") not in {"arrival", "hotel"} for item in day_items)
            should_append_hotel = bool(include_hotel and selected_hotel and (day_index > 1 or has_route_content or not inserted_checkin))
            if should_append_hotel:
                hotel_distance, hotel_minutes = _route_leg_metric(prev_lat, prev_lon, selected_hotel.get("lat"), selected_hotel.get("lon"))
                hotel_reason = str(selected_hotel.get("reason") or "作为过夜住宿推荐").strip()
                if selected_hotel_name:
                    hotel_reason = f"你已确认入住这家酒店，系统会把它作为当天收束点；{hotel_reason}"
                if inserted_checkin and has_route_content:
                    hotel_reason = f"晚间返回已选酒店休息；{hotel_reason}"
                if hotel_distance is not None and hotel_minutes is not None:
                    hotel_reason = f"{hotel_reason}；距上一站约{hotel_distance}km，预计{hotel_minutes}分钟"
                _append_day_item(
                    day_items,
                    day_index=day_index,
                    day_date=day_date,
                    item_time="21:00",
                    activity_type="hotel",
                    name=selected_hotel.get("name"),
                    reason=hotel_reason,
                    lat=selected_hotel.get("lat"),
                    lon=selected_hotel.get("lon"),
                    avg_cost=selected_hotel.get("avg_cost"),
                    est_cost=selected_hotel.get("est_cost", ""),
                    score=selected_hotel.get("score"),
                    rating=selected_hotel.get("rating"),
                    distance_km=hotel_distance,
                    travel_minutes=hotel_minutes,
                    map_url=selected_hotel.get("map_url", ""),
                    nav_url=selected_hotel.get("nav_url", ""),
                    extra={"hotel_selected": True, "hotel_stage": "stay"},
                )
                if hotel_has_coords:
                    base_start_lat = float(selected_hotel["lat"])
                    base_start_lon = float(selected_hotel["lon"])

            daily_itinerary.append(
                {
                    "day_index": day_index,
                    "day_label": _day_title(day_date, day_index),
                    "date": day_date.isoformat(),
                    "items": day_items,
                }
            )
    destination_hub_result = search_transport_hubs(city=city, lat=cur_lat, lon=cur_lon, limit=6)
    destination_hubs = destination_hub_result.get("items") if isinstance(destination_hub_result, dict) else []
    recommended_destination_hub = (
        destination_hub_result.get("recommended") if isinstance(destination_hub_result, dict) else {}
    ) or {}
    destination_query = str(recommended_destination_hub.get("name") or city).strip() or city
    destination_station = str(payload.get("destination_station", "") or "").strip()
    destination_station_code = str(payload.get("destination_station_code", "") or "").strip().upper()
    transport_info = _build_transport_info(
        booking_date=booking_date,
        origin_city=origin_city,
        origin_station=origin_station,
        origin_station_code=origin_station_code,
        destination_city=city,
        destination_station=destination_station,
        destination_station_code=destination_station_code,
        destination_query=destination_query,
        destination_hub=recommended_destination_hub if isinstance(recommended_destination_hub, dict) else {},
        destination_hubs=destination_hubs if isinstance(destination_hubs, list) else [],
        train_date=travel_date,
    )
    trip_weather = _build_trip_weather_rows(city=city, start_date=travel_date, days=days, lat=cur_lat, lon=cur_lon)

    warnings: list[str] = []
    selection_guidance = {
        "should_prompt": False,
        "status": "balanced",
        "title": "",
        "message": "",
        "suggestion": "",
        "estimated_days": estimated_days,
        "estimated_hours": estimated_hours,
        "requested_days": days,
        "selected_attractions": len(selected_attractions),
        "selected_foods": len(selected_foods),
        "excluded_items": len(excluded_attractions) + len(excluded_foods),
    }
    if planning_mode == "candidate" and not selection_active:
        warnings.append("当前仍按系统默认推荐生成，可在推荐列表中标记“想去 / 想吃”或“不感兴趣”后重新生成。")
    if attraction_styles and not any(_theme_match_count(item, attraction_styles) > 0 for item in scored_attractions):
        style_labels = " / ".join([ATTRACTION_THEME_LABELS[key] for key in attraction_styles if key in ATTRACTION_THEME_LABELS])
        warnings.append(f"当前景点偏好为 {style_labels}，但附近可用候选较少，请放宽分类或更换位置。")
    if selected_attractions and not matched_selected_attractions:
        warnings.append("当前勾选的景点未能匹配到可用候选项，请放宽筛选或重新选择。")
    if selected_foods and not matched_selected_foods:
        warnings.append("当前勾选的美食未能匹配到可用候选项，请放宽筛选或重新选择。")
    if selection_active and estimated_days > days:
        warnings.append(f"按当前选择预计更适合约 {estimated_days} 天行程；如坚持 {days} 天，建议减少候选项。")
        selection_guidance = {
            **selection_guidance,
            "should_prompt": True,
            "status": "overloaded",
            "title": "当前天数偏紧",
            "message": f"你当前勾选的景点和美食预计更适合约 {estimated_days} 天；现在只计划 {days} 天，路线会偏赶。",
            "suggestion": "建议减少部分想去景点或想吃美食，再重新生成路线。",
        }
    if selection_active and estimated_days + 1 < days:
        warnings.append(f"按当前选择预计约 {estimated_days} 天即可完成；如计划 {days} 天，建议补充景点或美食。")
        selection_guidance = {
            **selection_guidance,
            "should_prompt": True,
            "status": "underfilled",
            "title": "当前内容偏少",
            "message": f"按你现在的选择，预计约 {estimated_days} 天即可完成；当前计划是 {days} 天，行程可能偏松。",
            "suggestion": "建议补充一些想去景点或想吃美食，再重新生成路线。",
        }
    if not top_attractions and not top_foods:
        warnings.append("当前筛选后可生成的路线内容较少，请减少排除项或补充偏好。")
    warnings.extend([warning for warning in transport_info.get("warnings") or [] if warning])
    transport_status = str(transport_info.get("status") or "")
    if transport_status in {"disabled", "missing_origin"}:
        warnings.append(str(transport_info.get("note") or "").strip())
    elif transport_status == "error":
        warnings.append(str(transport_info.get("note") or "12306 查询失败，请稍后重试。").strip())

    if live_attractions or live_foods:
        poi_source = "amap+search" if (search_attractions or search_foods) else "amap"
    elif search_attractions or search_foods:
        poi_source = "search"
    else:
        poi_source = "none"

    return {
        "weather": {
            "city": weather_raw["city"],
            "date": travel_date.isoformat(),
            "max_temp_c": weather_raw["max_temp_c"],
            "min_temp_c": weather_raw["min_temp_c"],
            "precipitation_mm": weather_raw["precipitation_mm"],
            "wind_speed_mps": weather_raw["wind_speed_mps"],
            "weather_label": label,
            "weather_label_zh": weather_raw.get("weather_condition_zh") or weather_label_zh(label),
            "weather_condition_zh": weather_raw.get("weather_condition_zh") or weather_label_zh(label),
            "tips": weather_tips(label),
            "temperature_tone": _weather_tone(weather_raw["max_temp_c"]),
            "clothing_tip": _clothing_tip(
                weather_raw.get("max_temp_c"),
                weather_raw.get("min_temp_c"),
                str(weather_raw.get("weather_condition_zh") or weather_label_zh(label)),
                weather_raw.get("precipitation_mm"),
            ),
            "source": weather_raw["source"],
        },
        "location": {"city": city, "lat": cur_lat, "lon": cur_lon},
        "planning_mode": planning_mode,
        "estimated_days": estimated_days,
        "estimated_hours": estimated_hours,
        "warnings": warnings,
        "selection_guidance": selection_guidance,
        "transport": transport_info,
        "selection_summary": {
            "attraction_styles": attraction_styles,
            "selected_attractions": selected_attractions,
            "excluded_attractions": excluded_attractions,
            "selected_foods": selected_foods,
            "excluded_foods": excluded_foods,
        },
        "selected_hotel_name": selected_hotel.get("name") if selected_hotel else "",
        "selected_transport": selected_transport if arrival_selection else {},
        "days": days,
        "poi_source": poi_source,
        "attractions": candidate_attractions,
        "foods": candidate_foods,
        "candidate_attractions": candidate_attractions,
        "candidate_foods": candidate_foods,
        "hotels": top_hotels if include_hotel else [],
        "include_hotel": include_hotel,
        "hotel_price_max": hotel_price_max,
        "itinerary": itinerary,
        "daily_itinerary": daily_itinerary,
        "trip_weather": trip_weather,
        "budget_hint": budget_hint,
    }


def build_preview(payload: dict) -> dict:
    preview_payload = dict(payload or {})
    preview_payload["include_hotel"] = True
    preview_payload["selected_hotel_name"] = ""
    preview_payload["selected_transport"] = {}
    preview_payload["planning_mode"] = str(preview_payload.get("planning_mode") or "candidate")
    preview_data = build_plan(preview_payload)
    return {
        "weather": preview_data.get("weather") or {},
        "location": preview_data.get("location") or {},
        "planning_mode": preview_data.get("planning_mode") or "candidate",
        "estimated_days": preview_data.get("estimated_days"),
        "estimated_hours": preview_data.get("estimated_hours"),
        "warnings": preview_data.get("warnings") or [],
        "selection_summary": preview_data.get("selection_summary") or {},
        "transport": preview_data.get("transport") or {},
        "trip_weather": preview_data.get("trip_weather") or [],
        "poi_source": preview_data.get("poi_source") or "none",
        "budget_hint": preview_data.get("budget_hint") or "",
        "attractions": preview_data.get("attractions") or [],
        "foods": preview_data.get("foods") or [],
        "hotels": preview_data.get("hotels") or [],
    }


def build_plan_check(payload: dict) -> dict:
    check_data = build_plan(payload, include_itinerary=False)
    return {
        "weather": check_data.get("weather") or {},
        "location": check_data.get("location") or {},
        "planning_mode": check_data.get("planning_mode") or "candidate",
        "estimated_days": check_data.get("estimated_days"),
        "estimated_hours": check_data.get("estimated_hours"),
        "warnings": check_data.get("warnings") or [],
        "selection_guidance": check_data.get("selection_guidance") or {},
        "selection_summary": check_data.get("selection_summary") or {},
        "transport": check_data.get("transport") or {},
        "selected_hotel_name": check_data.get("selected_hotel_name") or "",
        "days": check_data.get("days"),
    }
