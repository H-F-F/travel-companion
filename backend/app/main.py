from __future__ import annotations

from fastapi import FastAPI, HTTPException

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

app = FastAPI(title="AI Travel Companion API", version="0.1.0")


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
        return build_plan(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/plan-check")
async def check_plan(req: PlanRequest) -> dict:
    try:
        payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        return build_plan_check(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/preview")
async def preview_plan(req: PlanRequest) -> dict:
    try:
        payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        return build_preview(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
