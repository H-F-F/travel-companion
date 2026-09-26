"""规划 Agent：让 LLM 通过 function calling 调度天气 / POI / 车站 / 行程生成工具。

核心设计：
1. 保留现有规则引擎（engine.build_plan）作为确定性能力与兜底；
2. LLM 作为编排者，按需调用工具，最后调用 generate_itinerary 生成完整行程；
3. LLM 未配置或调用失败时自动降级：从自然语言提取参数 → 规则引擎出方案，
   保证任何情况下前端都有可用结果。

对外主入口：run_agent(message, history) -> AgentResult
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from . import engine
from .llm_client import LLMClient, LLMError

# ---------------------------------------------------------------------------
# 工具定义（JSON Schema，OpenAI function calling 格式）
# ---------------------------------------------------------------------------

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "把城市名或地址解析为经纬度与规范化城市名，例如「西安」「成都武侯区」。",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "城市名或详细地址"},
                },
                "required": ["address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某城市/地址未来 N 天（1-7）天气，返回温度、降水、风力与出行提示。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，例如「西安」"},
                    "address": {"type": "string", "description": "详细地址（可选，优先级高于 city）"},
                    "days": {"type": "integer", "description": "预报天数，默认 3"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pois",
            "description": "搜索某城市的景点或美食 POI，返回名称、分类、室内/室外、距离等信息。kind=attraction 搜景点，kind=food 搜美食。",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "搜索关键词，例如「西安 博物馆」或城市名"},
                    "kind": {"type": "string", "enum": ["attraction", "food"], "description": "搜索类型"},
                    "city": {"type": "string", "description": "城市名（可选）"},
                    "limit": {"type": "integer", "description": "返回条数，默认 8，最多 20"},
                },
                "required": ["q", "kind"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_stations",
            "description": "搜索火车站/高铁站（12306 数据），返回车站名与车站代码。例如「西安」「北京南」。",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "车站关键词"},
                    "limit": {"type": "integer", "description": "返回条数，默认 6"},
                },
                "required": ["q"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_itinerary",
            "description": "生成完整可执行行程：天气 + 景点推荐 + 美食推荐 + 每日路线。信息齐备后必须调用它作为最后一步。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "目的地城市名"},
                    "travel_date": {"type": "string", "description": "出发日期，格式 YYYY-MM-DD"},
                    "days": {"type": "integer", "description": "行程天数，1-14"},
                    "budget_level": {"type": "string", "enum": ["economy", "standard", "premium"], "description": "预算档位：economy 经济 / standard 标准 / premium 舒适"},
                    "attraction_styles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "景点偏好，可选值：人文/自然/地标/室内",
                    },
                    "food_preferences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "口味偏好，例如：本地特色/辣/海鲜/甜品",
                    },
                    "origin_city": {"type": "string", "description": "出发城市（可选）"},
                    "origin_station": {"type": "string", "description": "出发车站名（可选）"},
                    "include_hotel": {"type": "boolean", "description": "是否包含酒店推荐"},
                },
                "required": ["city", "travel_date", "days"],
            },
        },
    },
]

TOOL_NAME_TO_LABEL = {
    "geocode": "定位城市",
    "get_weather": "查询天气",
    "search_pois": "搜索景点美食",
    "search_stations": "查询车站",
    "generate_itinerary": "生成行程",
}

SYSTEM_PROMPT = """你是「智能旅行管家」的规划 Agent。你帮助用户规划可执行的旅行行程。

工作方式：
1. 先理解用户需求。若缺少目的地、出发日期、天数等关键信息，直接向用户提问，不要编造。当用户提出首次需求但参数不全时，先简要列出「已确认信息」和「待确认信息」，询问用户补充或确认，不要直接生成。
2. 按需调用工具获取天气、景点/美食、车站信息。只调用与当前需求相关的工具，不要无意义地把所有工具都调用一遍。
3. 信息齐备后，调用 generate_itinerary 生成完整行程（包含天气、景点推荐、美食推荐、每日路线）。
4. 生成行程后，用简洁自然的中文总结：目的地、天数、预算、天气提示、最值得去的 2-3 个点。不要输出 JSON，不要重复罗列全部工具结果。
5. 若工具返回 error，如实告知用户，并用其他途径继续规划。
6. 若系统反馈你生成的行程未通过校验（如缺少景点、缺少每日安排、天数不符），必须修正后重新调用 generate_itinerary。
7. 用户可能追问行程细节（比如某天安排、某景点怎么去），结合已生成的行程内容直接回答，必要时再次调用工具获取新信息。"""

MAX_AGENT_ROUNDS = 8
MAX_TOOL_RESULT_CHARS = 4000
MAX_HISTORY_TURNS = 6
MAX_ITINERARY_ATTEMPTS = 3  # 生成行程的校验-重试上限（P0-①）


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    tool: str
    args: dict[str, Any]
    summary: str
    status: str = "ok"  # ok | error
    seq: int = 0


@dataclass
class AgentResult:
    reply_text: str
    steps: list[AgentStep] = field(default_factory=list)
    plan: dict[str, Any] | None = None
    mode: str = "llm"  # llm | fallback | needs_input
    brief: dict[str, Any] | None = None   # 结构化确认卡（P0-②）
    meta: dict[str, Any] | None = None    # 可观测元数据（P0-③）


# ---------------------------------------------------------------------------
# 工具执行器
# ---------------------------------------------------------------------------


def _summary_of_weather(data: dict[str, Any]) -> str:
    loc = data.get("location") or {}
    city = str(loc.get("city") or "未知城市")
    rows = data.get("daily") or []
    if not rows:
        return f"{city}：天气数据不可用"
    parts = []
    for row in rows[:4]:
        day = str(row.get("day_label") or row.get("date") or "")
        cond = str(row.get("weather_condition_zh") or row.get("weather_label") or "未知")
        max_t = row.get("max_temp_c")
        min_t = row.get("min_temp_c")
        precip = row.get("precipitation_mm")
        text = f"{day} {cond}"
        if max_t is not None:
            text += f" {min_t}~{max_t}°C"
        if precip is not None:
            text += f" 降水{precip}mm"
        parts.append(text)
    return f"{city}：{'；'.join(parts)}"


def _summary_of_pois(data: dict[str, Any]) -> str:
    kind = data.get("kind", "attraction")
    label = "景点" if kind == "attraction" else "美食"
    items = data.get("items") or []
    if not items:
        return f"未找到相关{label}（{data.get('query') or ''}）"
    parts = []
    for item in items[:8]:
        name = item.get("name") or "未知"
        cat = item.get("display_category") or item.get("category") or ""
        dist = item.get("distance_km")
        cost = item.get("est_cost") or item.get("avg_cost")
        text = name
        if cat:
            text += f"（{cat}）"
        if dist is not None:
            text += f" 距{data.get('center_label') or '中心'}{dist}km"
        if cost:
            text += f" 人均{cost}"
        parts.append(text)
    return f"{label}候选：{'；'.join(parts)}"


def _summary_of_plan(plan: dict[str, Any]) -> str:
    city = str((plan.get("weather") or {}).get("city") or "")
    days = int(plan.get("estimated_days") or 1)
    attractions = plan.get("attractions") or []
    foods = plan.get("foods") or []
    itinerary = plan.get("itinerary") or []
    top_attractions = [a.get("name") for a in attractions[:3] if a.get("name")]
    top_foods = [f.get("name") for f in foods[:3] if f.get("name")]
    itin_text = "；".join(
        f"{i.get('time')} {i.get('name')}" for i in itinerary[:6]
    )
    parts = [f"已生成{city}{days}日行程"]
    if top_attractions:
        parts.append("推荐景点：" + "、".join(top_attractions))
    if top_foods:
        parts.append("推荐美食：" + "、".join(top_foods))
    if itin_text:
        parts.append("行程：" + itin_text)
    if plan.get("budget_hint"):
        parts.append(f"预算：{plan['budget_hint']}")
    return "；".join(parts)


def _normalize_style(values: Any) -> list[str]:
    style_map = {
        "人文": "culture", "文化": "culture", "历史": "culture", "古迹": "culture",
        "自然": "nature", "山水": "nature", "风景": "nature", "户外": "nature",
        "地标": "landmark", "城市地标": "landmark", "打卡": "landmark",
        "室内": "indoor", "博物馆": "indoor", "展馆": "indoor",
    }
    out: list[str] = []
    for raw in values or []:
        key = str(raw).strip()
        if not key:
            continue
        mapped = style_map.get(key)
        if mapped:
            out.append(mapped)
        elif key in {"culture", "nature", "landmark", "indoor"}:
            out.append(key)
    return list(dict.fromkeys(out))  # 去重保序


def _normalize_food_prefs(values: Any) -> list[str]:
    pref_map = {
        "辣": "辣", "麻辣": "辣", "火锅": "辣", "川菜": "辣",
        "海鲜": "海鲜", "甜品": "甜品", "甜": "甜品", "咖啡": "甜品",
        "本地": "本地特色", "本地特色": "本地特色", "特色": "本地特色",
        "小吃": "小吃", "面食": "面食", "清淡": "清淡", "素食": "清淡",
    }
    out: list[str] = []
    for raw in values or []:
        key = str(raw).strip()
        if not key:
            continue
        out.append(pref_map.get(key, key))
    return list(dict.fromkeys(out))


def _execute_tool(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """执行工具。返回 (给 LLM 的结果文本, 若为 generate_itinerary 则附带完整 plan)。"""
    plan: dict[str, Any] | None = None
    try:
        if name == "geocode":
            item = engine.geocode_address(str(args.get("address") or "").strip())
            if not item:
                return "未找到该地址，请检查城市名。", None
            return (
                f"定位成功：{item.get('city') or item.get('address') or ''} "
                f"（lat={item.get('lat')}, lon={item.get('lon')}）",
                None,
            )

        if name == "get_weather":
            try:
                data = engine.fetch_weather_forecast(
                    city=str(args.get("city") or "").strip(),
                    address=str(args.get("address") or "").strip(),
                    days=int(args.get("days") or 3),
                )
            except Exception as exc:  # noqa: BLE001
                return f"天气查询失败：{exc}", None
            return _summary_of_weather(data), None

        if name == "search_pois":
            try:
                data = engine.search_pois(
                    str(args.get("q") or "").strip(),
                    kind=str(args.get("kind") or "attraction"),
                    city=str(args.get("city") or "").strip(),
                    limit=max(1, min(20, int(args.get("limit") or 8))),
                )
            except Exception as exc:  # noqa: BLE001
                return f"POI 搜索失败：{exc}", None
            return _summary_of_pois(data), None

        if name == "search_stations":
            try:
                data = engine.search_train_stations(
                    str(args.get("q") or "").strip(),
                    limit=max(1, min(20, int(args.get("limit") or 6))),
                )
            except Exception as exc:  # noqa: BLE001
                return f"车站查询失败：{exc}", None
            items = data.get("items") or data.get("stations") or []
            if not items:
                return f"未找到车站：{args.get('q') or ''}", None
            parts = []
            for item in items[:8]:
                name = item.get("name") or item.get("station_name") or "未知"
                code = item.get("code") or item.get("station_code") or ""
                parts.append(f"{name}{('(' + code + ')') if code else ''}")
            return "车站候选：" + "；".join(parts), None

        if name == "generate_itinerary":
            city = str(args.get("city") or "").strip()
            travel_date = _parse_iso_date(args.get("travel_date"))
            days = max(1, min(14, int(args.get("days") or 3)))
            if not city:
                return "缺少 city，无法生成行程。", None
            if not travel_date:
                travel_date = (date.today() + timedelta(days=1)).isoformat()
            payload: dict[str, Any] = {
                "city": city,
                "travel_date": travel_date,
                "days": days,
                "budget_level": str(args.get("budget_level") or "standard"),
                "attraction_styles": _normalize_style(args.get("attraction_styles")),
                "food_preferences": _normalize_food_prefs(args.get("food_preferences")),
                "origin_city": str(args.get("origin_city") or "").strip(),
                "origin_station": str(args.get("origin_station") or "").strip(),
                "include_hotel": bool(args.get("include_hotel")),
                "planning_mode": "auto",
            }
            plan = engine.build_plan(payload, include_itinerary=True)
            return _summary_of_plan(plan), plan

        return f"未知工具：{name}", None
    except Exception as exc:  # noqa: BLE001
        return f"工具执行失败：{exc}", plan


# ---------------------------------------------------------------------------
# 行程质量校验 / 评分 / 确认卡（P0）
# ---------------------------------------------------------------------------


def _validate_plan(plan: dict[str, Any] | None) -> list[str]:
    """对生成的行程做确定性校验，返回错误列表（空 = 通过）。"""
    errors: list[str] = []
    if not plan:
        return ["行程为空"]
    weather = plan.get("weather") or {}
    if not weather:
        errors.append("缺少天气数据")
    if not weather.get("city"):
        errors.append("缺少目的地城市")
    if not plan.get("estimated_days"):
        errors.append("缺少行程天数")
    if not (plan.get("attractions") or []):
        errors.append("没有景点推荐")
    if not (plan.get("itinerary") or []):
        errors.append("没有每日行程安排")
    return errors


def _score_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    """确定性评分：POI 丰富度 + 路线覆盖 + 天气完整 + 预算提示。用于多候选择优。"""
    plan = plan or {}
    score = 0
    reasons: list[str] = []
    attractions = plan.get("attractions") or []
    foods = plan.get("foods") or []
    itinerary = plan.get("itinerary") or []
    weather = plan.get("weather") or {}
    if attractions:
        score += min(len(attractions), 8)
        reasons.append(f"景点 {len(attractions)} 个")
    if foods:
        score += min(len(foods), 4)
        reasons.append(f"美食 {len(foods)} 个")
    if itinerary:
        itin_days = len(itinerary)
        score += min(itin_days * 2, 10)
        reasons.append(f"路线覆盖 {itin_days} 天")
    if weather.get("daily"):
        score += 2
        reasons.append("天气完整")
    if plan.get("budget_hint"):
        score += 1
        reasons.append("含预算提示")
    return {"score": score, "reasons": reasons}


def _build_brief(params: dict[str, Any]) -> dict[str, Any]:
    """由提取到的结构化参数构造「确认卡」：字段 + 缺失项。"""
    missing: list[str] = []
    if not params.get("city"):
        missing.append("目的地")
    if not params.get("travel_date"):
        missing.append("出发日期")
    if not params.get("days"):
        missing.append("天数")
    return {
        "city": str(params.get("city") or ""),
        "travel_date": str(params.get("travel_date") or ""),
        "days": int(params.get("days") or 0) or "",
        "budget_level": str(params.get("budget_level") or "standard"),
        "attraction_styles": params.get("attraction_styles") or [],
        "food_preferences": params.get("food_preferences") or [],
        "missing": missing,
        "complete": not missing,
    }


def _payload_from_confirmed(confirmed: dict[str, Any]) -> dict[str, Any]:
    """确认卡参数 → engine.build_plan payload（缺省值兜底）。"""
    days = int(confirmed.get("days") or 3)
    return {
        "city": str(confirmed.get("city") or "").strip(),
        "travel_date": str(confirmed.get("travel_date") or (date.today() + timedelta(days=1)).isoformat()),
        "days": max(1, min(14, days)),
        "budget_level": str(confirmed.get("budget_level") or "standard"),
        "attraction_styles": _normalize_style(confirmed.get("attraction_styles")),
        "food_preferences": _normalize_food_prefs(confirmed.get("food_preferences")),
        "planning_mode": "auto",
    }


def _parse_iso_date(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 自然语言参数解析（降级路径：LLM 不可用时提取结构化参数）
# ---------------------------------------------------------------------------


def _parse_relative_date(text: str, today: date) -> date | None:
    if "今天" in text:
        return today
    if "明天" in text:
        return today + timedelta(days=1)
    if "后天" in text:
        return today + timedelta(days=2)
    if "大后天" in text:
        return today + timedelta(days=3)

    weekday_map = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
    m = re.search(r"(这|本|下)(周|星期|礼拜)([一二三四五六日天])", text)
    if m:
        offset, weekday = m.group(1), weekday_map[m.group(3)]
        days_ahead = (weekday - today.weekday()) % 7
        if offset == "下":
            days_ahead += 7
        return today + timedelta(days=days_ahead)

    m = re.search(r"(\d{1,2})月(\d{1,2})[日号]", text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            return today.replace(month=month, day=day)
        except ValueError:
            return None

    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def parse_natural_language(text: str) -> dict[str, Any]:
    """从一句自然语言中提取结构化规划参数（启发式，用于降级）。"""
    result: dict[str, Any] = {}
    today = date.today()

    # 城市：优先「去/到/在 X 玩/旅游/旅行」等结构（非贪婪，避免吞掉后缀动词）
    city_match = re.search(
        r"(?:去|到|在|游玩|旅游去)\s*([\u4e00-\u9fff]{2,6}?)(?:玩|旅游|旅行|度假|逛|转转|的行程|行程|攻略|，|,|。|\.|吧|！|!|、|\s|$)",
        text,
    )
    if city_match:
        candidate = city_match.group(1)
        if candidate not in {"哪里", "哪儿", "地方", "城市"}:
            result["city"] = candidate
    if not result.get("city"):
        # 循环剥离常见请求动词后再匹配「城市 + 旅行/一日游」结构，避免「帮我安排重庆」被整体当城市
        stripped = text
        for _ in range(4):
            cleaned = re.sub(
                r"^(?:帮我|请你|麻烦你|请|我想|我要|想要|打算|想|安排|规划|计划|推荐|给我|来|预约|订|有没有|有)",
                "",
                stripped,
            )
            if cleaned == stripped:
                break
            stripped = cleaned
        m = re.search(r"([\u4e00-\u9fff]{2,6})(?:旅行|旅游|游玩|攻略|自由行|跟团|\d{1,2}日游|[一两二三四五六七]日游)", stripped)
        if m:
            result["city"] = m.group(1)

    # 天数（支持阿拉伯数字与中文数字：一天/两天/3天）
    cn_num = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7}
    m = re.search(r"(\d{1,2})\s*(?:天|日)|([一两二三四五六七])\s*(?:天|日)", text)
    if m:
        if m.group(1):
            result["days"] = max(1, min(14, int(m.group(1))))
        else:
            result["days"] = cn_num[m.group(2)]

    # 日期
    parsed_date = _parse_relative_date(text, today)
    if parsed_date:
        result["travel_date"] = parsed_date.isoformat()

    # 预算
    if any(k in text for k in ("穷游", "省钱", "经济", "便宜")):
        result["budget_level"] = "economy"
    elif any(k in text for k in ("奢侈", "豪华", "高档", "舒适游")):
        result["budget_level"] = "premium"

    # 景点偏好
    styles: list[str] = []
    if any(k in text for k in ("人文", "历史", "古迹", "文化")):
        styles.append("culture")
    if any(k in text for k in ("自然", "山水", "风景", "户外")):
        styles.append("nature")
    if any(k in text for k in ("地标", "打卡")):
        styles.append("landmark")
    if any(k in text for k in ("室内", "博物馆", "展馆")):
        styles.append("indoor")
    if styles:
        result["attraction_styles"] = styles

    # 口味
    prefs: list[str] = []
    if "辣" in text or "火锅" in text or "川菜" in text:
        prefs.append("辣")
    if "海鲜" in text:
        prefs.append("海鲜")
    if "甜" in text or "甜品" in text:
        prefs.append("甜品")
    if any(k in text for k in ("本地", "特色")):
        prefs.append("本地特色")
    if "小吃" in text:
        prefs.append("小吃")
    if prefs:
        result["food_preferences"] = prefs

    return result


# ---------------------------------------------------------------------------
# Agent 主流程
# ---------------------------------------------------------------------------


def _build_messages(message: str, history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        role = str(turn.get("role") or "").strip()
        content = str(turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})
    return messages


def run_agent(
    message: str,
    history: list[dict[str, Any]] | None = None,
    llm: LLMClient | None = None,
    confirmed: dict[str, Any] | None = None,
) -> AgentResult:
    """执行一轮 Agent 对话的对外入口：包装 _run_agent_inner，统一回填可观测元数据。"""
    client = llm or LLMClient()
    t0 = time.monotonic()
    result = _run_agent_inner(message, history, client, confirmed)
    if client.available and result.meta is not None:
        result.meta["token_usage"] = dict(client.total_usage)
        result.meta["calls"] = client.calls
        result.meta["latency_ms"] = int((time.monotonic() - t0) * 1000)
    return result


def _run_agent_inner(
    message: str,
    history: list[dict[str, Any]] | None,
    client: LLMClient,
    confirmed: dict[str, Any] | None,
) -> AgentResult:
    """执行一轮 Agent 对话，返回文本回复、工具步骤、确认卡（可选）、行程（可选）与元数据。

    - confirmed 为 None：普通对话轮次（LLM 编排工具 / 追问 / 出确认卡）。
    - confirmed 为 dict：用户在前端确认了结构化参数 → 直接以确认参数生成行程。
    """
    text = (message or "").strip()
    if not text and not confirmed:
        return AgentResult(reply_text="请描述你的旅行需求，例如：帮我规划下周六去西安玩 2 天。", mode="needs_input", meta={})

    meta: dict[str, Any] = {
        "strategy": "llm",
        "attempts": 0,
        "validation": {"ok": True, "errors": []},
        "score": None,
        "model": client.model if client.available else "",
        "token_usage": {},
        "calls": 0,
        "latency_ms": 0,
    }

    # 用户已确认参数 → 直接生成（LLM 可用则 LLM 生成，否则规则引擎兜底）
    if confirmed:
        return _run_with_confirmed(client, text, history, confirmed, meta)

    steps: list[AgentStep] = []
    plan: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []  # 校验通过的候选（P0-① 择优）

    if client.available:
        try:
            messages = _build_messages(text, history)
            for round_index in range(MAX_AGENT_ROUNDS):
                msg = client.chat(messages, tools=TOOL_SPECS)
                messages.append(msg)
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    reply = str(msg.get("content") or "").strip()
                    if not reply:
                        reply = "我已经完成规划，你可以继续追问行程细节。"
                    # 没生成行程 → 尝试给用户出「确认卡」
                    brief = _build_brief(parse_natural_language(text))
                    if not plan and brief.get("city"):
                        return AgentResult(
                            reply_text=reply,
                            steps=steps,
                            mode="needs_input",
                            brief=brief,
                            meta=meta,
                        )
                    return AgentResult(reply_text=reply, steps=steps, plan=plan, mode="llm", meta=meta)

                retry_needed = False
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    name = str(fn.get("name") or "")
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                    except json.JSONDecodeError:
                        args = {}
                    result_text, result_plan = _execute_tool(name, args)
                    status = "error" if result_text.startswith(("工具执行失败", "未知工具")) else "ok"

                    if name == "generate_itinerary":
                        meta["attempts"] += 1
                        plan = result_plan
                        errors = _validate_plan(plan)
                        if errors:
                            meta["validation"]["ok"] = False
                            meta["validation"]["errors"] = errors
                            if meta["attempts"] < MAX_ITINERARY_ATTEMPTS:
                                # 带校验错误反馈给 LLM，让它修正后重试（P0-①）
                                messages.append({
                                    "role": "user",
                                    "content": (
                                        f"[行程校验未通过] 你刚生成的行程存在以下问题：{'；'.join(errors)}。"
                                        "请修正后重新调用 generate_itinerary 工具，不要再调用其他工具。"
                                    ),
                                })
                                steps.append(
                                    AgentStep(
                                        tool=name, args=args,
                                        summary=f"第 {meta['attempts']} 次尝试校验未通过：{'；'.join(errors)}",
                                        status="error", seq=len(steps) + 1,
                                    )
                                )
                                plan = None
                                retry_needed = True
                                break
                            # 尝试次数用尽 → 规则引擎兜底
                            return _fallback_plan(
                                text, steps,
                                reason=f"LLM 行程生成连续 {meta['attempts']} 次未通过校验：{'；'.join(errors)}",
                                meta=meta,
                            )
                        candidates.append(plan)
                    else:
                        steps.append(
                            AgentStep(
                                tool=name,
                                args=args,
                                summary=result_text[:160],
                                status=status,
                                seq=len(steps) + 1,
                            )
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id") or f"call_{round_index}_{name}",
                                "content": result_text[:MAX_TOOL_RESULT_CHARS],
                            }
                        )
                if retry_needed:
                    continue

            # 多候选择优（P0-①）
            return _pick_best_candidate(text, steps, candidates, meta)
        except LLMError as exc:
            # LLM 失败 → 降级
            return _fallback_plan(text, steps, reason=str(exc), meta=meta)
        except Exception as exc:  # noqa: BLE001
            return _fallback_plan(text, steps, reason=f"agent 异常: {exc}", meta=meta)
    return _fallback_plan(text, steps, reason="LLM 未配置", meta=meta)


def _pick_best_candidate(
    text: str,
    steps: list[AgentStep],
    candidates: list[dict[str, Any]],
    meta: dict[str, Any],
) -> AgentResult:
    """从校验通过的多个候选中确定性打分选优（P0-①）。"""
    if not candidates:
        return _fallback_plan(text, steps, reason="未生成有效候选", meta=meta)
    if len(candidates) == 1:
        plan = candidates[0]
    else:
        scored = sorted(
            ((_score_plan(c)["score"], c) for c in candidates),
            key=lambda pair: pair[0],
            reverse=True,
        )
        plan = scored[0][1]
    score = _score_plan(plan)
    meta["score"] = score["score"]
    meta["strategy"] = "llm_retry" if meta["attempts"] > 1 else "llm"
    meta["validation"]["ok"] = True
    meta["candidates"] = len(candidates)
    steps.append(
        AgentStep(
            tool="generate_itinerary",
            args={"city": (plan.get("weather") or {}).get("city", ""), "days": plan.get("estimated_days")},
            summary=_summary_of_plan(plan),
            status="ok",
            seq=len(steps) + 1,
        )
    )
    return AgentResult(
        reply_text=_summary_of_plan(plan),
        steps=steps,
        plan=plan,
        mode="llm",
        meta=meta,
    )


def _run_with_confirmed(
    client: LLMClient,
    text: str,
    history: list[dict[str, Any]] | None,
    confirmed: dict[str, Any],
    meta: dict[str, Any],
) -> AgentResult:
    """用户确认确认卡后：LLM 直接生成（带校验重试），失败降级规则引擎。"""
    steps: list[AgentStep] = []
    payload = _payload_from_confirmed(confirmed)
    if not payload.get("city"):
        return AgentResult(
            reply_text="确认参数缺少目的地，请重新描述你的旅行需求。",
            steps=steps,
            mode="needs_input",
            meta=meta,
        )

    if client.available:
        try:
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT
                        + "\n\n用户已确认以下规划参数，请直接调用 generate_itinerary 生成行程（不要追问、不要重复调用其他工具）：\n"
                        + json.dumps({k: v for k, v in payload.items() if v}, ensure_ascii=False)
                    ),
                }
            ]
            for turn in (history or [])[-MAX_HISTORY_TURNS:]:
                role = str(turn.get("role") or "").strip()
                content = str(turn.get("content") or "").strip()
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": "请按以上确认参数生成我的行程。"})

            for round_index in range(MAX_AGENT_ROUNDS):
                msg = client.chat(messages, tools=TOOL_SPECS)
                messages.append(msg)
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    reply = str(msg.get("content") or "").strip() or "已完成规划。"
                    return AgentResult(reply_text=reply, steps=steps, mode="llm", meta=meta)
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    name = str(fn.get("name") or "")
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
                    except json.JSONDecodeError:
                        args = {}
                    if name != "generate_itinerary":
                        result_text, _ = _execute_tool(name, args)
                        steps.append(
                            AgentStep(
                                tool=name, args=args,
                                summary=result_text[:160],
                                status="error" if result_text.startswith(("工具执行失败", "未知工具")) else "ok",
                                seq=len(steps) + 1,
                            )
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id") or f"call_{round_index}_{name}",
                            "content": result_text[:MAX_TOOL_RESULT_CHARS],
                        })
                        continue
                    meta["attempts"] += 1
                    plan = _execute_tool(name, args)[1]
                    errors = _validate_plan(plan)
                    if errors:
                        meta["validation"] = {"ok": False, "errors": errors}
                        if meta["attempts"] < MAX_ITINERARY_ATTEMPTS:
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"[行程校验未通过] 存在以下问题：{'；'.join(errors)}。"
                                    "请修正后重新调用 generate_itinerary，不要再调用其他工具。"
                                ),
                            })
                            steps.append(
                                AgentStep(
                                    tool=name, args=args,
                                    summary=f"第 {meta['attempts']} 次尝试校验未通过：{'；'.join(errors)}",
                                    status="error", seq=len(steps) + 1,
                                )
                            )
                            break
                        return _confirmed_engine_plan(payload, steps, meta,
                                                     reason=f"连续 {meta['attempts']} 次未通过校验：{'；'.join(errors)}")
                    score = _score_plan(plan)
                    meta["score"] = score["score"]
                    meta["strategy"] = "llm_retry" if meta["attempts"] > 1 else "llm"
                    meta["validation"]["ok"] = True
                    steps.append(
                        AgentStep(
                            tool="generate_itinerary",
                            args=args,
                            summary=f"已生成 {payload['city']}{payload['days']} 日行程（第 {meta['attempts']} 次尝试）",
                            status="ok",
                            seq=len(steps) + 1,
                        )
                    )
                    return AgentResult(
                        reply_text=_summary_of_plan(plan),
                        steps=steps,
                        plan=plan,
                        mode="llm",
                        meta=meta,
                    )
        except LLMError as exc:
            return _confirmed_engine_plan(payload, steps, meta, reason=str(exc))

    return _confirmed_engine_plan(payload, steps, meta, reason="LLM 未配置")


def _confirmed_engine_plan(
    payload: dict[str, Any],
    steps: list[AgentStep],
    meta: dict[str, Any],
    reason: str = "",
) -> AgentResult:
    """确认参数 → 规则引擎直接生成（LLM 不可用/失败时的确认路径兜底）。"""
    try:
        plan = engine.build_plan(payload, include_itinerary=True)
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            reply_text=f"规划失败：{exc}",
            steps=steps,
            mode="needs_input",
            meta=meta,
        )
    meta["strategy"] = "engine"
    meta["score"] = _score_plan(plan)["score"]
    steps.append(
        AgentStep(
            tool="generate_itinerary",
            args=payload,
            summary=f"规则引擎生成 {payload['city']}{payload['days']} 日行程",
            status="ok",
            seq=len(steps) + 1,
        )
    )
    note = f"（{reason}）" if reason else ""
    return AgentResult(
        reply_text=f"已按确认参数为你生成 {payload['city']} {payload['days']} 日行程{note}。",
        steps=steps,
        plan=plan,
        mode="fallback",
        meta=meta,
    )


def _fallback_plan(
    text: str,
    steps: list[AgentStep],
    reason: str,
    meta: dict[str, Any] | None = None,
) -> AgentResult:
    """LLM 不可用时：从自然语言提取参数 → 规则引擎生成行程。"""
    if meta is not None:
        meta["strategy"] = "engine"
    params = parse_natural_language(text)
    city = params.get("city")
    travel_date = params.get("travel_date") or (date.today() + timedelta(days=1)).isoformat()
    days = int(params.get("days") or 3)

    if not city:
        return AgentResult(
            reply_text=(
                "我暂时无法使用大模型（" + reason + "），也没能自动识别目的地。"
                "你可以这样告诉我：『帮我规划下周六去西安玩 2 天，预算标准，喜欢历史和美食』，"
                "或者直接配置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 后获得完整的对话式规划。"
            ),
            steps=steps,
            mode="needs_input",
            meta=meta,
        )

    try:
        payload: dict[str, Any] = {
            "city": city,
            "travel_date": travel_date,
            "days": days,
            "budget_level": str(params.get("budget_level") or "standard"),
            "attraction_styles": params.get("attraction_styles") or [],
            "food_preferences": params.get("food_preferences") or [],
            "planning_mode": "auto",
        }
        plan = engine.build_plan(payload, include_itinerary=True)
        date_note = ""
        if not params.get("travel_date"):
            date_note = f"（未指定出发日期，默认按 {travel_date} 规划）"
        steps.append(
            AgentStep(
                tool="generate_itinerary",
                args=payload,
                summary=f"规则引擎生成 {city}{days} 日行程{date_note}",
                status="ok",
                seq=len(steps) + 1,
            )
        )
        if meta is not None:
            meta["score"] = _score_plan(plan)["score"]
        return AgentResult(
            reply_text=(
                f"已按规则引擎为你生成 {city} {days} 日行程{date_note}。"
                "配置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 后，我会用大模型为你做更灵活的对话式规划。"
            ),
            steps=steps,
            plan=plan,
            mode="fallback",
            meta=meta,
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            reply_text=f"规划失败：{exc}",
            steps=steps,
            mode="needs_input",
            meta=meta,
        )
