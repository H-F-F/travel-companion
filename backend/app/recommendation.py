from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Poi:
    name: str
    category: str
    indoor: bool
    cost_level: int  # 1 low, 2 medium, 3 high
    tags: tuple[str, ...]
    lat: float | None = None
    lon: float | None = None
    rating: float | None = None
    reviews: int | None = None
    avg_cost: float | None = None
    chain: bool = False


ATTRACTIONS: list[Poi] = [
    Poi("城市博物馆", "museum", True, 1, ("history", "culture", "rainy_friendly")),
    Poi("滨江公园", "park", False, 1, ("view", "walking", "sunny_friendly")),
    Poi("老城区步行街", "district", False, 2, ("culture", "shopping", "sunny_friendly")),
    Poi("当代艺术馆", "gallery", True, 2, ("art", "indoor", "rainy_friendly")),
    Poi("城市观景台", "landmark", True, 3, ("view", "photo", "night")),
    Poi("植物园", "garden", False, 2, ("nature", "walking", "sunny_friendly")),
]

FOODS: list[Poi] = [
    Poi("本地面馆", "noodle", True, 1, ("local", "quick", "savory"), rating=4.4, reviews=320, avg_cost=28),
    Poi("老字号小吃店", "snack", True, 1, ("local", "street_food"), rating=4.5, reviews=460, avg_cost=22),
    Poi("海鲜餐厅", "seafood", True, 3, ("seafood", "dinner"), rating=4.6, reviews=280, avg_cost=218),
    Poi("川味餐馆", "sichuan", True, 2, ("spicy", "dinner"), rating=4.4, reviews=360, avg_cost=86),
    Poi("轻食沙拉吧", "salad", True, 2, ("healthy", "lunch"), rating=4.2, reviews=190, avg_cost=56),
    Poi("甜品咖啡馆", "dessert", True, 2, ("dessert", "afternoon"), rating=4.3, reviews=210, avg_cost=42),
]


def budget_to_cost_level(budget_level: str) -> int:
    if budget_level == "economy":
        return 1
    if budget_level == "premium":
        return 3
    return 2


def weather_label(max_temp_c: float, precipitation_mm: float) -> str:
    if precipitation_mm >= 3:
        return "rainy"
    if max_temp_c >= 32:
        return "hot"
    if max_temp_c <= 5:
        return "cold"
    return "pleasant"


def weather_label_zh(label: str) -> str:
    return {
        "rainy": "雨天",
        "hot": "炎热",
        "cold": "寒冷",
        "pleasant": "舒适",
    }.get(label, "未知")


def weather_tips(label: str) -> list[str]:
    if label == "rainy":
        return ["优先室内景点", "携带雨具", "预留交通缓冲时间"]
    if label == "hot":
        return ["避开中午暴晒", "补水防晒", "增加室内休息点"]
    if label == "cold":
        return ["注意保暖", "优先室内活动", "减少长距离步行"]
    return ["天气适中", "可安排步行路线", "建议提前预约热门点"]


def _is_generic_attraction_name(name: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return False

    generic_exact = {
        "城市风景",
        "风景",
        "景点",
        "景区",
        "风景区",
        "风景名胜",
        "旅游景点",
        "旅游风景区",
        "城市景点",
        "城市景观",
    }
    if text in generic_exact:
        return True

    specific_tokens = (
        "博物馆",
        "美术馆",
        "纪念馆",
        "科技馆",
        "海洋馆",
        "公园",
        "植物园",
        "动物园",
        "湿地",
        "森林",
        "古镇",
        "老街",
        "古城",
        "寺",
        "庙",
        "塔",
        "桥",
        "观景台",
        "步行街",
        "广场",
        "乐园",
        "外滩",
        "山",
        "湖",
        "江",
        "河",
        "海",
        "湾",
        "岛",
    )
    if any(token in text for token in specific_tokens):
        return False

    generic_suffixes = ("风景", "景点", "景区", "风景区", "风景名胜", "旅游景点", "旅游风景区")
    generic_prefixes = ("城市", "城区", "市区", "新区", "县城", "主城区", "本地", "当地")
    for suffix in generic_suffixes:
        if not text.endswith(suffix):
            continue
        prefix = text[: -len(suffix)].strip()
        if not prefix or prefix in generic_prefixes or len(prefix) <= 4:
            return True
    return False


def score_attraction(poi: Poi, weather: str, budget_level: str) -> tuple[float, str]:
    score = 50.0
    reasons: list[str] = []

    desired_cost = budget_to_cost_level(budget_level)
    cost_gap = abs(poi.cost_level - desired_cost)
    score += max(0, 20 - 8 * cost_gap)
    reasons.append("预算匹配")

    if weather == "rainy":
        if poi.indoor:
            score += 24
            reasons.append("雨天友好")
        else:
            score -= 18
    elif weather == "hot":
        if poi.indoor:
            score += 16
            reasons.append("高温可避暑")
        else:
            score -= 8
    elif weather == "cold":
        if poi.indoor:
            score += 14
            reasons.append("低温更舒适")

    if "view" in poi.tags:
        score += 4
    if "culture" in poi.tags:
        score += 3
    if _is_generic_attraction_name(poi.name):
        score -= 22

    return round(max(score, 0), 1), "，".join(reasons)


def normalize_food_preferences(food_preferences: list[str]) -> set[str]:
    alias_map = {
        "本地特色": "local",
        "地方菜": "local",
        "家常": "local",
        "清淡": "healthy",
        "健康餐": "healthy",
        "辣": "spicy",
        "麻辣": "spicy",
        "海鲜": "seafood",
        "甜品": "dessert",
        "咖啡": "dessert",
        "快餐": "quick",
        "小吃": "street_food",
        "夜宵": "dinner",
        "素食": "healthy",
        "火锅": "spicy",
        "烧烤": "savory",
        "面食": "savory",
    }

    normalized: set[str] = set()
    for p in food_preferences:
        raw = p.strip()
        if not raw:
            continue
        normalized.add(raw.lower())
        if raw in alias_map:
            normalized.add(alias_map[raw])
    return normalized


def is_chain_restaurant(name: str) -> bool:
    chain_keywords = (
        "肯德基",
        "kfc",
        "麦当劳",
        "mcdonald",
        "必胜客",
        "pizza hut",
        "汉堡王",
        "burger king",
        "赛百味",
        "subway",
        "华莱士",
        "德克士",
        "真功夫",
        "永和大王",
        "星巴克",
        "starbucks",
        "瑞幸",
        "luckin",
        "costa",
        "manner",
        "蜜雪冰城",
        "喜茶",
        "奈雪",
        "沪上阿姨",
        "古茗",
        "茶百道",
    )
    lowered = name.lower().strip()
    return any(keyword in lowered for keyword in chain_keywords)


def _food_budget_range(budget_level: str) -> tuple[float, float]:
    return {
        "economy": (0.0, 60.0),
        "standard": (40.0, 180.0),
        "premium": (120.0, 400.0),
    }.get(budget_level, (40.0, 180.0))


def score_food(poi: Poi, budget_level: str, food_preferences: list[str]) -> tuple[float, str]:
    score = 48.0
    reasons: list[str] = []

    if poi.avg_cost is not None and poi.avg_cost > 0:
        budget_min, budget_max = _food_budget_range(budget_level)
        if budget_min <= poi.avg_cost <= budget_max:
            score += 18
            reasons.append("人均合适")
        elif poi.avg_cost < budget_min:
            score += 6
            reasons.append("人均偏省")
        else:
            score -= min(18, (poi.avg_cost - budget_max) / 15)
            reasons.append("人均偏高")
    else:
        desired_cost = budget_to_cost_level(budget_level)
        score += max(0, 20 - 8 * abs(poi.cost_level - desired_cost))
        reasons.append("预算匹配")

    prefs = normalize_food_preferences(food_preferences)
    tags = {t.lower() for t in poi.tags}
    matched = prefs.intersection(tags)
    if matched:
        score += 22
        reasons.append("口味偏好匹配")

    if "local" in tags:
        score += 6
        reasons.append("本地特色")

    if poi.rating is not None:
        score += max(-6.0, min(16.0, (poi.rating - 3.8) * 10.0))
        if poi.rating >= 4.4:
            reasons.append("口碑较好")

    if poi.reviews is not None and poi.reviews > 0:
        score += min(8.0, math.log10(max(1, poi.reviews)) * 2.5)
        if poi.reviews >= 100:
            reasons.append("评价较多")

    if poi.chain or is_chain_restaurant(poi.name):
        score -= 18
        reasons.append("连锁餐饮降权")

    if "quick" in tags and "quick" not in prefs:
        score -= 6

    if "dessert" in tags and "dessert" not in prefs:
        score -= 8

    return round(max(score, 0), 1), "，".join(reasons)


def score_hotel(poi: Poi, hotel_price_max: int) -> tuple[float, str]:
    score = 52.0
    reasons: list[str] = []

    if poi.avg_cost is not None and poi.avg_cost > 0:
        if poi.avg_cost <= hotel_price_max:
            score += 22
            reasons.append("价格在酒店预算内")
        else:
            score -= min(30, (poi.avg_cost - hotel_price_max) / 18)
            reasons.append("超出酒店预算")
    else:
        reasons.append("价格待确认")

    if poi.rating is not None:
        score += max(-8.0, min(18.0, (poi.rating - 3.8) * 12.0))
        if poi.rating >= 4.4:
            reasons.append("住宿口碑较好")

    if poi.reviews is not None and poi.reviews > 0:
        score += min(9.0, math.log10(max(1, poi.reviews)) * 3.0)
        if poi.reviews >= 80:
            reasons.append("评价较多")

    return round(max(score, 0), 1), "，".join(reasons)


def cost_level_text(cost_level: int) -> str:
    return {1: "低", 2: "中", 3: "高"}.get(cost_level, "中")
