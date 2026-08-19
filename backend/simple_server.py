from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from backend.app.engine import (
    build_plan,
    build_plan_check,
    build_preview,
    fetch_weather_forecast,
    geocode_address,
    resolve_admin_from_coords,
    search_addresses,
    search_pois,
    search_transport_hubs,
    search_train_stations,
)


class Handler(BaseHTTPRequestHandler):
    FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_text(self, status: int, content: str, content_type: str) -> None:
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, rel_path: str, content_type: str) -> bool:
        base = self.FRONTEND_DIR.resolve()
        file_path = (base / rel_path).resolve()
        if not str(file_path).startswith(str(base)):
            return False
        if not file_path.exists() or not file_path.is_file():
            return False
        self._write_text(200, file_path.read_text(encoding="utf-8"), content_type)
        return True

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route_path = parsed.path

        if route_path == "/health":
            self._write_json(200, {"status": "ok"})
            return
        if route_path == "/api/v1/locate":
            try:
                params = parse_qs(parsed.query)
                lat = float((params.get("lat") or [""])[0])
                lon = float((params.get("lon") or [""])[0])
                admin = resolve_admin_from_coords(lat, lon)
                self._write_json(200, admin)
            except Exception as exc:
                self._write_json(400, {"error": str(exc)})
            return
        if route_path == "/api/v1/search-address":
            try:
                params = parse_qs(parsed.query)
                q = str((params.get("q") or [""])[0])
                limit_raw = (params.get("limit") or ["10"])[0]
                limit = int(limit_raw) if str(limit_raw).isdigit() else 10
                items = search_addresses(q, max(1, min(20, limit)))
                self._write_json(200, {"items": items})
            except Exception as exc:
                self._write_json(400, {"error": str(exc)})
            return
        if route_path == "/api/v1/search-poi":
            try:
                params = parse_qs(parsed.query)
                q = str((params.get("q") or [""])[0])
                kind = str((params.get("kind") or ["attraction"])[0])
                city = str((params.get("city") or [""])[0])
                limit_raw = str((params.get("limit") or ["10"])[0])
                lat_raw = str((params.get("lat") or [""])[0]).strip()
                lon_raw = str((params.get("lon") or [""])[0]).strip()
                lat = float(lat_raw) if lat_raw else None
                lon = float(lon_raw) if lon_raw else None
                limit = int(limit_raw) if limit_raw.isdigit() else 10
                result = search_pois(q, kind=kind, city=city, lat=lat, lon=lon, limit=max(1, min(20, limit)))
                self._write_json(200, result)
            except Exception as exc:
                self._write_json(400, {"error": str(exc)})
            return
        if route_path == "/api/v1/transport/stations":
            try:
                params = parse_qs(parsed.query)
                q = str((params.get("q") or [""])[0])
                limit_raw = str((params.get("limit") or ["10"])[0])
                limit = int(limit_raw) if limit_raw.isdigit() else 10
                result = search_train_stations(q, max(1, min(20, limit)))
                self._write_json(200, result)
            except RuntimeError as exc:
                status = 503 if ("MCP_12306_URL not set" in str(exc) or "unavailable" in str(exc).lower()) else 502
                self._write_json(status, {"error": str(exc)})
            except Exception as exc:
                self._write_json(400, {"error": str(exc)})
            return
        if route_path == "/api/v1/transport/hubs":
            try:
                params = parse_qs(parsed.query)
                q = str((params.get("q") or [""])[0])
                city = str((params.get("city") or [""])[0])
                limit_raw = str((params.get("limit") or ["8"])[0])
                lat_raw = str((params.get("lat") or [""])[0]).strip()
                lon_raw = str((params.get("lon") or [""])[0]).strip()
                lat = float(lat_raw) if lat_raw else None
                lon = float(lon_raw) if lon_raw else None
                limit = int(limit_raw) if limit_raw.isdigit() else 8
                result = search_transport_hubs(query=q, city=city, lat=lat, lon=lon, limit=max(1, min(20, limit)))
                self._write_json(200, result)
            except Exception as exc:
                self._write_json(400, {"error": str(exc)})
            return
        if route_path == "/api/v1/geocode":
            try:
                params = parse_qs(parsed.query)
                address = str((params.get("address") or params.get("q") or [""])[0])
                city = str((params.get("city") or [""])[0])
                item = geocode_address(address, city)
                if not item:
                    self._write_json(404, {"error": "address not found"})
                    return
                self._write_json(200, item)
            except Exception as exc:
                self._write_json(400, {"error": str(exc)})
            return
        if route_path == "/api/v1/weather":
            try:
                params = parse_qs(parsed.query)
                city = str((params.get("city") or [""])[0])
                address = str((params.get("address") or params.get("q") or [""])[0])
                days_raw = str((params.get("days") or ["3"])[0])
                lat_raw = str((params.get("lat") or [""])[0]).strip()
                lon_raw = str((params.get("lon") or [""])[0]).strip()
                lat = float(lat_raw) if lat_raw else None
                lon = float(lon_raw) if lon_raw else None
                days = int(days_raw) if days_raw.isdigit() else 3
                result = fetch_weather_forecast(city=city, address=address, days=days, lat=lat, lon=lon)
                self._write_json(200, result)
            except Exception as exc:
                self._write_json(400, {"error": str(exc)})
            return
        if route_path in ("/", "/index.html") and self._serve_file("index.html", "text/html"):
            return
        if route_path.startswith("/frontend/"):
            rel_path = route_path.replace("/frontend/", "", 1)
            if rel_path.endswith(".css") and self._serve_file(rel_path, "text/css"):
                return
            if rel_path.endswith(".js") and self._serve_file(rel_path, "application/javascript"):
                return
            if rel_path.endswith(".html") and self._serve_file(rel_path, "text/html"):
                return
        self._write_json(404, {"error": "not found", "path": route_path})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/v1/plan", "/api/v1/plan-check", "/api/v1/preview"}:
            self._write_json(404, {"error": "not found", "path": parsed.path})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8"))
            if parsed.path == "/api/v1/preview":
                result = build_preview(payload)
            elif parsed.path == "/api/v1/plan-check":
                result = build_plan_check(payload)
            else:
                result = build_plan(payload)
            self._write_json(200, result)
        except Exception as exc:
            self._write_json(400, {"error": str(exc)})


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    server = HTTPServer((host, port), Handler)
    print(f"AI Travel Companion API running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
