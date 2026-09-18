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
1. 先理解用户需求。若缺少目的地、出发日期、天数等关键信息，直接向用户提问，不要编造。
2. 按需调用工具获取天气、景点/美食、车站信息。只调用与当前需求相关的工具，不要无意义地把所有工具都调用一遍。
3. 信息齐备后，调用 generate_itinerary 生成完整行程（包含天气、景点推荐、美食推荐、每日路线）。
4. 生成行程后，用简洁自然的中文总结：目的地、天数、预算、天气提示、最值得去的 2-3 个点。不要输出 JSON，不要重复罗列全部工具结果。
5. 若工具返回 error，如实告知用户，并用其他途径继续规划。
6. 用户可能追问行程细节（比如某天安排、某景点怎么去），结合已生成的行程内容直接回答，必要时再次调用工具获取新信息。"""

MAX_AGENT_ROUNDS = 8
MAX_TOOL_RESULT_CHARS = 4000
MAX_HISTORY_TURNS = 6


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
) -> AgentResult:
    """执行一轮 Agent 对话，返回文本回复、工具步骤与（可选）完整行程。"""
    text = (message or "").strip()
    if not text:
        return AgentResult(reply_text="请描述你的旅行需求，例如：帮我规划下周六去西安玩 2 天。", mode="needs_input")

    client = llm or LLMClient()
    steps: list[AgentStep] = []
    plan: dict[str, Any] | None = None

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
                    return AgentResult(reply_text=reply, steps=steps, plan=plan, mode="llm")

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
                    if result_plan is not None:
                        plan = result_plan
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
            return AgentResult(
                reply_text="我已经完成规划，你可以继续追问行程细节。",
                steps=steps,
                plan=plan,
                mode="llm",
            )
        except LLMError as exc:
            # LLM 失败 → 降级
            return _fallback_plan(text, steps, reason=str(exc))
        except Exception as exc:  # noqa: BLE001
            return _fallback_plan(text, steps, reason=f"agent 异常: {exc}")
    return _fallback_plan(text, steps, reason="LLM 未配置")


def _fallback_plan(text: str, steps: list[AgentStep], reason: str) -> AgentResult:
    """LLM 不可用时：从自然语言提取参数 → 规则引擎生成行程。"""
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
        return AgentResult(
            reply_text=(
                f"已按规则引擎为你生成 {city} {days} 日行程{date_note}。"
                "配置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL 后，我会用大模型为你做更灵活的对话式规划。"
            ),
            steps=steps,
            plan=plan,
            mode="fallback",
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            reply_text=f"规划失败：{exc}",
            steps=steps,
            mode="needs_input",
        )
