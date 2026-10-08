# -*- coding: utf-8 -*-
"""行程导出：把一份完整 plan 渲染成自包含的 HTML 攻略页（无外部依赖）。"""
from __future__ import annotations

import html
from datetime import datetime


def _esc(v: object) -> str:
    return html.escape(str(v if v is not None else ""))


def _clean(v: object) -> str:
    s = str(v if v is not None else "").strip()
    return s


def export_plan_html(plan: dict, meta: dict | None = None) -> str:
    """生成单文件 HTML 攻略。plan 为完整行程（engine/agent 输出），meta 为可观测元数据（可选）。"""
    weather = plan.get("weather") or {}
    wsum = weather.get("summary") or {}
    location = plan.get("location") or {}
    city = _clean(plan.get("requested_city") or wsum.get("city") or location.get("city") or "目的地")
    days = int(plan.get("days") or plan.get("requested_days") or plan.get("estimated_days") or 1)
    title = f"{city} {days}日游攻略"
    created = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 天气条
    weather_label = _clean(wsum.get("weather_label_zh") or wsum.get("weather_label"))
    temp = ""
    if wsum.get("max_temp_c") is not None:
        temp = f"{_clean(wsum.get('min_temp_c'))}~{_clean(wsum.get('max_temp_c'))}°C"
    tips = _clean(wsum.get("tips"))

    # 每日天气
    trip_weather = plan.get("trip_weather") or []
    tw_html = ""
    if trip_weather:
        cards = []
        for tw in trip_weather[:days]:
            cards.append(
                f"""<div class="tw-card"><span class="tw-day">{_esc(tw.get('day_label') or '')}</span>
                    <span class="tw-w">{_esc(tw.get('weather_label_zh') or tw.get('weather_label') or '')}</span>
                    <span class="tw-t">{_esc(tw.get('min_temp_c') or '')}~{_esc(tw.get('max_temp_c') or '')}°C</span></div>"""
            )
        tw_html = '<div class="tw-row">' + "".join(cards) + "</div>"

    # 每日安排（按天分组）
    itinerary = plan.get("itinerary") or []
    by_day: dict[int, list[dict]] = {}
    for item in itinerary:
        by_day.setdefault(int(item.get("day_index") or 1), []).append(item)
    day_html = ""
    for d in range(1, days + 1):
        items = by_day.get(d, [])
        rows = []
        for it in items:
            raw_act = _clean(it.get("activity_type") or "行程")
            act = {
                "attraction": "景点",
                "food": "美食",
                "hotel": "酒店",
                "stay": "住宿",
                "checkin": "入住",
                "transit": "交通",
                "lunch": "用餐",
                "dinner": "用餐",
                "breakfast": "用餐",
            }.get(raw_act, raw_act)
            name = _clean(it.get("name"))
            time = _clean(it.get("time"))
            theme = _clean(it.get("theme_label"))
            cost = _clean(it.get("est_cost") or it.get("avg_cost"))
            reason = _clean(it.get("reason"))
            badge = {"景点": "#2F6FED", "美食": "#0EA5A4", "酒店": "#7C5CE0", "住宿": "#7C5CE0"}.get(act, "#5d6675")
            rows.append(
                f"""<li class="it-row">
                    <span class="it-time">{_esc(time)}</span>
                    <span class="it-badge" style="background:{badge}">{_esc(act)}</span>
                    <div class="it-main">
                      <b>{_esc(name)}</b>
                      <span class="it-meta">{_esc(theme)}{' · ' + _esc(cost) if cost else ''}</span>
                      {f'<p class="it-reason">{_esc(reason)}</p>' if reason else ''}
                    </div></li>"""
            )
        if not rows:
            rows.append('<li class="it-empty">当日行程待生成</li>')
        day_html += (
            f"""<section class="day-card"><h3 class="day-title">DAY {d}</h3>"""
            + ("".join(rows))
            + "</section>"
        )

    def _poi_list(items: list[dict], kind: str) -> str:
        lis = []
        for p in items or []:
            name = _clean(p.get("name"))
            if not name:
                continue
            meta_bits = [
                _clean(p.get("theme_label")) or _clean(p.get("category")),
                _clean(p.get("avg_cost") or p.get("est_cost")),
                (f"{_clean(p.get('distance_km'))}km" if p.get("distance_km") is not None else ""),
            ]
            meta_s = " · ".join([b for b in meta_bits if b])
            reason = _clean(p.get("reason"))
            lis.append(
                f"""<li class="poi-row"><b>{_esc(name)}</b>
                    <span class="it-meta">{_esc(meta_s)}</span>
                    {f'<p class="it-reason">{_esc(reason)}</p>' if reason else ''}</li>"""
            )
        if not lis:
            lis.append('<li class="it-empty">暂无推荐</li>')
        tag = "景点" if kind == "attraction" else "美食"
        return (
            f'<section class="day-card"><h3 class="day-title">推荐{tag}</h3><ul class="poi-list">'
            + "".join(lis)
            + "</ul></section>"
        )

    attractions_html = _poi_list(plan.get("attractions") or [], "attraction")
    foods_html = _poi_list(plan.get("foods") or [], "food")

    hotels = plan.get("hotels") or []
    hotel_html = ""
    if hotels:
        lis = []
        for h in hotels[:6]:
            name = _clean(h.get("name"))
            if not name:
                continue
            price = _clean(h.get("price")) or _clean(h.get("avg_price"))
            dist = _clean(h.get("distance_km")) or _clean(h.get("distance"))
            lis.append(
                f'<li class="poi-row"><b>{_esc(name)}</b><span class="it-meta">{" · ".join([b for b in [price, (dist + "km" if dist else "")] if b])}</span></li>'
            )
        hotel_html = (
            '<section class="day-card"><h3 class="day-title">酒店参考</h3><ul class="poi-list">'
            + "".join(lis)
            + "</ul></section>"
        )

    # 元数据
    meta_html = ""
    if meta:
        bits = []
        if meta.get("strategy"):
            bits.append(f"策略 {_esc(meta.get('strategy'))}")
        if meta.get("model"):
            bits.append(f"模型 {_esc(meta.get('model'))}")
        if meta.get("score") is not None:
            bits.append(f"质量分 {_esc(meta.get('score'))}")
        tu = meta.get("token_usage") or {}
        if tu.get("total_tokens"):
            bits.append(f"Token {_esc(tu.get('total_tokens'))}")
        if meta.get("latency_ms"):
            bits.append(f"耗时 {_esc(round(int(meta['latency_ms']) / 1000, 1))}s")
        if bits:
            meta_html = '<div class="meta-line">' + " · ".join(bits) + "</div>"

    budget_hint = _clean(plan.get("budget_hint"))

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{_esc(title)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; background: #f6f5f2; color: #1c2430; padding: 32px 16px; }}
  .sheet {{ max-width: 860px; margin: 0 auto; background: #fff; border-radius: 20px; padding: 36px 40px; box-shadow: 0 8px 30px rgba(20,30,45,.08); }}
  h1 {{ font-size: 26px; }}
  .sub {{ color: #5d6675; font-size: 13px; margin-top: 6px; }}
  .hero {{ display:flex; justify-content:space-between; align-items:flex-start; border-bottom: 1px solid #e7e5e0; padding-bottom: 18px; }}
  .meta-line {{ color:#8a93a2; font-size:12px; margin-top: 8px; }}
  .weather-strip {{ display:flex; gap:12px; align-items:center; background:#eef4ff; border-radius:12px; padding:12px 16px; margin:18px 0 6px; }}
  .weather-strip b {{ color:#2F6FED; }}
  .weather-strip span {{ color:#5d6675; font-size:13px; }}
  .tw-row {{ display:flex; gap:10px; flex-wrap:wrap; margin: 12px 0 20px; }}
  .tw-card {{ background:#fafaf8; border:1px solid #e7e5e0; border-radius:12px; padding:10px 14px; display:flex; flex-direction:column; gap:2px; }}
  .tw-day {{ font-size:12px; color:#8a93a2; }}
  .tw-w {{ font-size:13px; font-weight:600; }}
  .tw-t {{ font-size:12px; color:#5d6675; }}
  .day-card {{ margin-top: 22px; }}
  .day-title {{ font-size:16px; border-left: 4px solid #2F6FED; padding-left: 10px; margin-bottom: 10px; }}
  ul {{ list-style:none; }}
  .it-row {{ display:flex; gap:12px; align-items:flex-start; padding: 10px 0; border-bottom: 1px dashed #ecebe6; }}
  .it-time {{ min-width: 52px; font-size:12px; color:#8a93a2; padding-top: 3px; }}
  .it-badge {{ font-size:11px; color:#fff; border-radius:6px; padding:2px 8px; margin-top: 2px; white-space: nowrap; }}
  .it-main {{ flex:1; }}
  .it-main b {{ font-size:14px; }}
  .it-meta {{ color:#8a93a2; font-size:12px; margin-left: 6px; }}
  .it-reason {{ color:#5d6675; font-size:12px; margin-top: 3px; }}
  .poi-row {{ display:flex; gap:10px; align-items:baseline; padding: 7px 0; }}
  .poi-row b {{ font-size:14px; }}
  .it-empty {{ color:#8a93a2; font-size:13px; padding: 8px 0; }}
  .foot {{ margin-top: 26px; padding-top: 14px; border-top: 1px solid #e7e5e0; color:#8a93a2; font-size:12px; display:flex; justify-content:space-between; }}
  @media (max-width:640px) {{ .sheet {{ padding: 22px 18px; }} .it-row {{ flex-wrap: wrap; }} }}
</style>
</head>
<body>
<div class="sheet">
  <div class="hero">
    <div>
      <h1>{_esc(title)}</h1>
      <div class="sub">Travel-Companion · 生成于 {created}</div>
      {meta_html}
    </div>
  </div>
  <div class="weather-strip"><b>{_esc(weather_label)}</b><span>{_esc(temp)}</span><span>{_esc(tips)}</span></div>
  {tw_html}
  {day_html}
  {attractions_html}
  {foods_html}
  {hotel_html}
  <div class="foot"><span>{_esc(budget_hint)}</span><span>由 Travel-Companion AI 生成</span></div>
</div>
</body>
</html>
"""
