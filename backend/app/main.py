from __future__ import annotations

import asyncio
import os
import urllib.request
from pathlib import Path

# 服务进程内所有外部 API（LLM / 高德 / 天气 / 12306）一律直连，
# 不受用户系统代理环境影响；否则失效的本地代理会把请求挂死。
for _proxy_key in (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
):
    os.environ.pop(_proxy_key, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import export as export_mod
from . import storage
from .agent import run_agent
from .engine import (
    build_plan,
    build_plan_check,
    build_preview,
    fetch_weather_forecast,
    geocode_address,
    search_pois,
    search_train_stations,
    search_transport_hubs,
)
from .models import PlanRequest

app = FastAPI(title="AI Travel Companion API", version="0.2.0")


class ChatMessage(BaseModel):
    role: str = Field(default="user", pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    confirm: dict | None = Field(default=None, description="确认卡参数：用户确认后直接生成行程")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/geocode")
async def geocode_location(address: str, city: str = "") -> dict:
    item = geocode_address(address, city)
    if not item:
        raise HTTPException(status_code=404, detail="address not found")
    return item


@app.get("/api/v1/weather")
async def weather_forecast(city: str = "", address: str = "", days: int = 3, lat: float | None = None, lon: float | None = None) -> dict:
    try:
        return fetch_weather_forecast(city=city, address=address, days=days, lat=lat, lon=lon)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/search-poi")
async def search_poi(q: str, kind: str = "attraction", city: str = "", lat: float | None = None, lon: float | None = None, limit: int = 10) -> dict:
    try:
        return search_pois(q, kind=kind, city=city, lat=lat, lon=lon, limit=max(1, min(20, limit)))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/transport/stations")
async def search_transport_stations(q: str, limit: int = 10) -> dict:
    try:
        return search_train_stations(q, limit=max(1, min(20, limit)))
    except RuntimeError as exc:
        if "MCP_12306_URL not set" in str(exc) or "unavailable" in str(exc).lower():
            raise HTTPException(status_code=503, detail="12306 不可用：请配置 MCP_12306_URL，或安装本地 12306 MCP 包。") from exc
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/transport/hubs")
async def search_transport_hub_candidates(city: str = "", q: str = "", lat: float | None = None, lon: float | None = None, limit: int = 8) -> dict:
    try:
        return search_transport_hubs(query=q, city=city, lat=lat, lon=lon, limit=max(1, min(20, limit)))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/plan")
async def generate_plan(req: PlanRequest) -> dict:
    try:
        payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        plan = await asyncio.to_thread(build_plan, payload)
        plan_id = await asyncio.to_thread(storage.save_plan, plan, title=f"{plan.get('location', {}).get('city') or ''}{plan.get('days') or ''}日游", request=payload)
        plan["plan_id"] = plan_id
        return plan
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/plan-check")
async def check_plan(req: PlanRequest) -> dict:
    try:
        payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        return await asyncio.to_thread(build_plan_check, payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/preview")
async def preview_plan(req: PlanRequest) -> dict:
    try:
        payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        return await asyncio.to_thread(build_preview, payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/chat")
async def chat(req: ChatRequest) -> dict:
    """对话式规划入口：LLM Agent 编排工具，支持确认卡（confirm）与校验重试，失败自动降级到规则引擎。"""
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        result = await asyncio.to_thread(run_agent, req.message, history, None, req.confirm)
        plan_id = None
        if result.plan:
            plan_id = await asyncio.to_thread(
                storage.save_plan,
                result.plan,
                title=f"{(result.plan.get('location') or {}).get('city') or ''}{(result.plan.get('days') or '')}日游",
                request=req.confirm or None,
            )
        return {
            "reply_text": result.reply_text,
            "steps": [
                {"tool": s.tool, "args": s.args, "summary": s.summary, "status": s.status, "seq": s.seq}
                for s in result.steps
            ],
            "plan": result.plan,
            "plan_id": plan_id,
            "mode": result.mode,
            "brief": result.brief,
            "meta": result.meta,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------- 历史行程（持久化） ----------------

@app.get("/api/v1/plans")
async def list_plans(limit: int = 30) -> dict:
    """历史行程列表（不含完整 payload，列表页用）。"""
    items = await asyncio.to_thread(storage.list_plans, max(1, min(100, limit)))
    return {"items": items, "total": len(items)}


@app.get("/api/v1/plans/{plan_id}")
async def get_plan_detail(plan_id: int) -> dict:
    plan = await asyncio.to_thread(storage.get_plan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="行程不存在")
    return plan


@app.delete("/api/v1/plans/{plan_id}")
async def delete_plan(plan_id: int) -> dict:
    ok = await asyncio.to_thread(storage.delete_plan, plan_id)
    if not ok:
        raise HTTPException(status_code=404, detail="行程不存在")
    return {"ok": True, "id": plan_id}


@app.get("/api/v1/plans/{plan_id}/export", response_class=HTMLResponse)
async def export_plan(plan_id: int) -> HTMLResponse:
    """导出一份 HTML 旅行攻略（自包含，可下载/打印为 PDF）。"""
    plan = await asyncio.to_thread(storage.get_plan, plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail="行程不存在")
    payload = plan.get("payload") or {}
    meta = (payload.get("meta") if isinstance(payload, dict) else None) or {}
    html_doc = await asyncio.to_thread(export_mod.export_plan_html, payload, meta)
    return HTMLResponse(content=html_doc, headers={"Content-Disposition": f'attachment; filename="travel-plan-{plan_id}.html"'})


# 前端静态文件：单服务跑全栈（API + 页面）
_FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
