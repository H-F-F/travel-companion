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
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import export as export_mod
from . import storage
from .agent import run_agent, run_agent_stream
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
    session_id: int | None = Field(default=None, description="对话会话 ID：传则续写该会话，不传则新建")


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
    """对话式规划入口：LLM Agent 编排工具，支持确认卡（confirm）与校验重试，失败自动降级到规则引擎。

    会话持久化：每轮结束把完整对话（历史 + 用户 + 助手）写入 SQLite，
    前端刷新后可通过 session_id 恢复对话。
    """
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
        # 会话持久化：完整消息 = 历史 + 本轮用户 + 本轮助手
        session_id = req.session_id
        if session_id is not None:
            saved = await asyncio.to_thread(storage.get_session_messages, session_id)
            if saved is None:
                session_id = None  # 会话不存在则新建
        messages = history + [{"role": "user", "content": req.message}]
        if result.reply_text:
            messages.append({"role": "assistant", "content": result.reply_text})
        session_id = await asyncio.to_thread(storage.save_session_messages, messages, session_id)
        return {
            "reply_text": result.reply_text,
            "steps": [
                {"tool": s.tool, "args": s.args, "summary": s.summary, "status": s.status, "seq": s.seq}
                for s in result.steps
            ],
            "plan": result.plan,
            "plan_id": plan_id,
            "session_id": session_id,
            "mode": result.mode,
            "brief": result.brief,
            "meta": result.meta,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------- 对话流式（SSE） ----------------

@app.post("/api/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    """对话式规划流式入口：与 /api/v1/chat 同编排，但 LLM 最终生成按 token 推送。

    事件协议（SSE `data: {...}` 行）：
      {"type":"token","text"}    文本片段
      {"type":"step","step":{}}  工具步骤
      {"type":"brief","brief":{}} 确认卡参数
      {"type":"plan","plan":{}}  行程（含 plan_id）
      {"type":"done","mode","reply_text","steps","meta","session_id"} 结束
    """
    history = [{"role": m.role, "content": m.content} for m in req.history]
    gen = run_agent_stream(req.message, history, None, req.confirm)

    async def event_stream():
        plan_id = None
        session_id = req.session_id
        try:
            while True:
                ev = await asyncio.to_thread(_next_or_none, gen)
                if ev is None:
                    break
                etype = ev.get("type")
                if etype == "plan" and ev.get("plan"):
                    plan_id = await asyncio.to_thread(
                        storage.save_plan,
                        ev["plan"],
                        title=f"{(ev['plan'].get('location') or {}).get('city') or ''}{(ev['plan'].get('days') or '')}日游",
                        request=req.confirm or None,
                    )
                    yield f"data: {_sse(ev | {'plan_id': plan_id})}\n\n"
                elif etype == "done":
                    # 会话持久化：历史 + 本轮用户 + 助手回复
                    if session_id is not None:
                        saved = await asyncio.to_thread(storage.get_session_messages, session_id)
                        if saved is None:
                            session_id = None
                    messages = history + [{"role": "user", "content": req.message}]
                    if ev.get("reply_text"):
                        messages.append({"role": "assistant", "content": ev["reply_text"]})
                    session_id = await asyncio.to_thread(storage.save_session_messages, messages, session_id)
                    yield f"data: {_sse(ev | {'session_id': session_id})}\n\n"
                else:
                    yield f"data: {_sse(ev)}\n\n"
        except Exception as exc:  # 流中途异常：推送 done 让前端收尾
            yield f"data: {_sse({'type': 'done', 'mode': 'fallback', 'reply_text': f'请求失败：{exc}，请稍后重试。', 'steps': [], 'meta': {}, 'session_id': session_id})}\n\n"
        finally:
            try:
                gen.close()
            except Exception:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _next_or_none(gen):
    """取下一个事件；生成器正常结束时返回 None（避免 StopIteration 被 asyncio 包装）。"""
    try:
        return next(gen)
    except StopIteration:
        return None


def _sse(ev: dict) -> str:
    """SSE 数据行：JSON 序列化（兼容 date 对象）。"""
    import json as _json

    return _json.dumps(ev, ensure_ascii=False, default=str)


# ---------------- 对话会话（持久化） ----------------

@app.get("/api/v1/sessions")
async def list_sessions(limit: int = 20) -> dict:
    """历史对话会话列表（标题取首条用户消息，不含完整消息）。"""
    items = await asyncio.to_thread(storage.list_sessions, max(1, min(50, limit)))
    return {"items": items, "total": len(items)}


@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: int) -> dict:
    session = await asyncio.to_thread(storage.get_session_messages, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    return session


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: int) -> dict:
    ok = await asyncio.to_thread(storage.delete_session, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True, "id": session_id}


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
