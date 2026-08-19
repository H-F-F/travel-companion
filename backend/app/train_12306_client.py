from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .station_catalog import LOCAL_STATION_CATALOG_ZH


LOCAL_STATION_CATALOG = [
    {"name": "西安", "code": "XAY", "pinyin": "xian", "py_short": "xa", "aliases": ["西安站", "西安火车站"]},
    {"name": "西安北", "code": "EAY", "pinyin": "xianbei", "py_short": "xab", "aliases": ["西安北站", "西安北客站"]},
    {"name": "西安西", "code": "EGY", "pinyin": "xianxi", "py_short": "xax", "aliases": ["西安西站", "阿房宫"]},
    {"name": "引镇", "code": "CAY", "pinyin": "yinzhen", "py_short": "yz", "aliases": ["引镇站", "西安南", "西安南站"]},
    {"name": "汉中", "code": "HOY", "pinyin": "hanzhong", "py_short": "hz", "aliases": ["汉中站"]},
]


def _load_local_env() -> None:
    root = Path(__file__).resolve().parents[2]
    for env_path in (root / ".env", root / "backend" / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)


def _normalize_duration_text(value: object) -> str:
    text = str(value or "").strip()
    if ":" not in text:
        return text
    hour_s, minute_s = (text.split(":", 1) + ["0"])[:2]
    if hour_s.isdigit() and minute_s.isdigit():
        return f"{int(hour_s):02d}:{int(minute_s):02d}"
    return text


def _normalize_station_query(value: object) -> str:
    text = str(value or "").strip().lower()
    for token in ("火车站", "高铁站", "站", "市", "客站", "火车", "高铁"):
        text = text.replace(token, "")
    text = text.replace("'", "").replace("’", "").replace("`", "").replace('"', "")
    return "".join(text.split())


class _StreamReader(threading.Thread):
    def __init__(self, stream, queue: Queue[str]) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self._queue = queue

    def run(self) -> None:
        while True:
            line = self._stream.readline()
            if not line:
                break
            self._queue.put(line.rstrip("\r\n"))


class MCP12306Client:
    DEFAULT_PROTOCOL_VERSION = "2025-03-26"

    def __init__(self, base_url: str | None = None, timeout_sec: float | None = None) -> None:
        _load_local_env()
        self.base_url = (base_url or os.getenv("MCP_12306_URL", "")).strip()
        timeout_raw = timeout_sec
        if timeout_raw is None:
            env_timeout = os.getenv("MCP_12306_TIMEOUT_SEC", "").strip()
            timeout_raw = float(env_timeout) if env_timeout else 12.0
        self.timeout_sec = max(3.0, float(timeout_raw))
        self.node_binary = shutil.which("node") or shutil.which("node.exe")
        self.node_script = self._resolve_node_script()

    def _resolve_node_script(self) -> str:
        env_script = str(os.getenv("MCP_12306_NODE_SCRIPT", "") or "").strip()
        if env_script and Path(env_script).exists():
            return str(Path(env_script).resolve())
        root = Path(__file__).resolve().parents[2]
        local_script = root / ".mcp12306" / "node_modules" / "12306-mcp" / "build" / "index.js"
        if local_script.exists():
            return str(local_script.resolve())
        return ""

    @property
    def mode(self) -> str:
        if self.base_url:
            return "http"
        if self.node_binary and self.node_script:
            return "stdio-local"
        return "disabled"

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    def _headers(self, session_id: str | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "ai-travel-companion/0.1",
            "MCP-Protocol-Version": self.DEFAULT_PROTOCOL_VERSION,
        }
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        return headers

    def _request_json(
        self,
        method: str,
        payload: dict | None = None,
        session_id: str | None = None,
        expect_status: tuple[int, ...] = (200,),
    ) -> tuple[int, dict, dict[str, str]]:
        if not self.base_url:
            raise RuntimeError("MCP_12306_URL not set")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = Request(self.base_url, data=body, headers=self._headers(session_id), method=method)
        try:
            with urlopen(req, timeout=self.timeout_sec) as resp:  # nosec B310
                status = getattr(resp, "status", 200)
                text = resp.read().decode("utf-8")
                headers = dict(resp.headers.items())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"12306 MCP HTTP {exc.code}: {detail or exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"12306 MCP unavailable: {exc.reason}") from exc

        if status not in expect_status:
            raise RuntimeError(f"12306 MCP returned unexpected status {status}")
        data = json.loads(text) if text else {}
        return status, data, headers

    @contextmanager
    def _http_session(self) -> Iterator[str | None]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": self.DEFAULT_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "ai-travel-companion", "version": "0.1.0"},
            },
        }
        _, body, headers = self._request_json("POST", payload=payload)
        if body.get("error"):
            raise RuntimeError(self._format_rpc_error(body["error"]))
        session_id = headers.get("Mcp-Session-Id") or headers.get("mcp-session-id")
        try:
            if session_id:
                notify_payload = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
                self._request_json("POST", payload=notify_payload, session_id=session_id, expect_status=(200, 202, 204))
            yield session_id
        finally:
            if session_id:
                try:
                    self._request_json("DELETE", session_id=session_id, expect_status=(200, 202, 204))
                except Exception:
                    pass

    def _call_http_tool(self, session_id: str | None, request_id: int, tool_name: str, arguments: dict) -> dict | list:
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        _, body, _ = self._request_json("POST", payload=payload, session_id=session_id)
        if body.get("error"):
            raise RuntimeError(self._format_rpc_error(body["error"]))
        result = body.get("result") or {}
        if result.get("isError"):
            raise RuntimeError(self._extract_text(result.get("content") or []) or f"12306 tool `{tool_name}` failed")
        return self._extract_json(result.get("content") or [])

    def _extract_text(self, content: list[dict]) -> str:
        texts = [str(item.get("text") or "").strip() for item in content if str(item.get("type") or "") == "text"]
        return "\n".join([text for text in texts if text]).strip()

    def _extract_json(self, content: list[dict]) -> dict | list:
        text = self._extract_text(content)
        if not text:
            raise RuntimeError("12306 MCP returned empty content")
        if text.startswith("Error:"):
            raise RuntimeError(text)
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise RuntimeError(text)

    def _format_rpc_error(self, error: dict) -> str:
        message = str(error.get("message") or "12306 MCP request failed")
        data = error.get("data")
        if data:
            return f"{message}: {data}"
        return message

    @contextmanager
    def _stdio_session(self) -> Iterator[tuple[subprocess.Popen, Queue[str], Queue[str]]]:
        if not (self.node_binary and self.node_script):
            raise RuntimeError("Local 12306 MCP package not installed")
        proc = subprocess.Popen(
            [self.node_binary, self.node_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        out_queue: Queue[str] = Queue()
        err_queue: Queue[str] = Queue()
        _StreamReader(proc.stdout, out_queue).start()
        _StreamReader(proc.stderr, err_queue).start()
        try:
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": self.DEFAULT_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "ai-travel-companion", "version": "0.1.0"},
                },
            }
            response = self._stdio_request(proc, out_queue, err_queue, init_message, expect_response=True)
            if response.get("error"):
                raise RuntimeError(self._format_rpc_error(response["error"]))
            self._stdio_request(
                proc,
                out_queue,
                err_queue,
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                expect_response=False,
            )
            yield proc, out_queue, err_queue
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _stdio_request(
        self,
        proc: subprocess.Popen,
        out_queue: Queue[str],
        err_queue: Queue[str],
        payload: dict,
        expect_response: bool = True,
    ) -> dict:
        if proc.stdin is None:
            raise RuntimeError("12306 MCP stdio is not writable")
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        if not expect_response:
            return {}
        return self._read_stdio_message(proc, out_queue, err_queue)

    def _read_stdio_message(self, proc: subprocess.Popen, out_queue: Queue[str], err_queue: Queue[str]) -> dict:
        deadline = time.time() + self.timeout_sec
        while time.time() < deadline:
            try:
                line = out_queue.get(timeout=0.2)
            except Empty:
                if proc.poll() is not None:
                    raise RuntimeError(self._stdio_error_text(proc, err_queue) or "Local 12306 MCP exited unexpectedly")
                continue
            if not line.strip():
                continue
            return json.loads(line)
        raise RuntimeError(self._stdio_error_text(proc, err_queue) or "Local 12306 MCP request timed out")

    def _stdio_error_text(self, proc: subprocess.Popen, err_queue: Queue[str]) -> str:
        lines = []
        while True:
            try:
                lines.append(err_queue.get_nowait())
            except Empty:
                break
        if proc.poll() is not None and proc.returncode not in (0, None):
            lines.append(f"process exited with code {proc.returncode}")
        return "\n".join([line for line in lines if line]).strip()

    def _parse_stdio_tool_json(self, result: dict) -> dict | list:
        content = (result.get("result") or {}).get("content") or []
        text = self._extract_text(content)
        if not text:
            return []
        if text.startswith("Error:"):
            raise RuntimeError(text)
        if text.startswith("很抱歉") or "未查询到" in text:
            return []
        return json.loads(text)

    def _map_ticket_prices(self, prices: object) -> dict[str, str]:
        seats: dict[str, str] = {}
        if not isinstance(prices, list):
            return seats
        for price in prices:
            if not isinstance(price, dict):
                continue
            seat_name = str(price.get("seat_name") or "").strip()
            seat_num = str(price.get("num") or "").strip()
            if seat_name and seat_num:
                seats[seat_name] = seat_num
        return seats

    def _map_stdio_ticket(self, item: dict) -> dict:
        return {
            "train_no": str(item.get("start_train_code") or item.get("train_no") or "").strip(),
            "from_station": str(item.get("from_station") or "").strip(),
            "from_station_code": str(item.get("from_station_telecode") or "").strip(),
            "to_station": str(item.get("to_station") or "").strip(),
            "to_station_code": str(item.get("to_station_telecode") or "").strip(),
            "start_time": str(item.get("start_time") or "").strip(),
            "arrive_time": str(item.get("arrive_time") or "").strip(),
            "duration": _normalize_duration_text(item.get("lishi") or item.get("duration") or ""),
            "seats": self._map_ticket_prices(item.get("prices") or []),
        }

    def _search_stations_local(self, query: str, limit: int = 10) -> dict:
        normalized_query = _normalize_station_query(query)
        if not normalized_query:
            return {"success": True, "query": str(query or "").strip(), "count": 0, "stations": []}

        items: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for station in LOCAL_STATION_CATALOG_ZH:
            name = str(station.get("name") or "").strip()
            code = str(station.get("code") or "").strip().upper()
            aliases = [name, *[str(alias or "").strip() for alias in station.get("aliases") or []]]
            pinyin = str(station.get("pinyin") or "").strip()
            py_short = str(station.get("py_short") or "").strip()
            haystacks = aliases + [pinyin, py_short]
            if not any(normalized_query in _normalize_station_query(value) for value in haystacks if value):
                continue
            key = (name, code)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "name": name,
                    "code": code,
                    "pinyin": pinyin,
                    "py_short": py_short,
                    "display_name": " · ".join([part for part in [name, code, py_short or pinyin] if part]),
                }
            )
            if len(items) >= limit:
                break

        return {
            "success": True,
            "query": str(query or "").strip(),
            "count": len(items),
            "stations": items[:limit],
        }

    def _search_stations_http(self, query: str, limit: int = 10) -> dict:
        with self._http_session() as session_id:
            result = self._call_http_tool(
                session_id,
                request_id=2,
                tool_name="search-stations",
                arguments={"query": query, "limit": max(1, min(20, int(limit)))},
            )
        return result if isinstance(result, dict) else {}

    def _search_stations_stdio(self, query: str, limit: int = 10) -> dict:
        normalized_query = str(query or "").strip()
        station_query = normalized_query[:-1] if normalized_query.endswith("站") else normalized_query
        city_query = station_query.replace("市", "")
        items = []
        seen: set[tuple[str, str]] = set()

        def add_item(name: str, code: str) -> None:
            key = (name, code)
            if not name or key in seen:
                return
            seen.add(key)
            items.append(
                {
                    "name": name,
                    "code": code,
                    "pinyin": "",
                    "py_short": "",
                    "display_name": " · ".join([part for part in [name, code] if part]),
                }
            )

        with self._stdio_session() as (proc, out_queue, err_queue):
            for request_id, (tool_name, arguments) in enumerate(
                [
                    ("get-station-code-by-names", {"stationNames": station_query}),
                    ("get-stations-code-in-city", {"city": city_query}),
                ],
                start=2,
            ):
                try:
                    response = self._stdio_request(
                        proc,
                        out_queue,
                        err_queue,
                        {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}},
                    )
                    data = self._parse_stdio_tool_json(response)
                except Exception:
                    continue
                if isinstance(data, dict):
                    for station in data.values():
                        if not isinstance(station, dict):
                            continue
                        add_item(str(station.get("station_name") or "").strip(), str(station.get("station_code") or "").strip())
                elif isinstance(data, list):
                    for station in data:
                        if not isinstance(station, dict):
                            continue
                        add_item(str(station.get("station_name") or "").strip(), str(station.get("station_code") or "").strip())
                if len(items) >= limit:
                    break

        return {
            "success": True,
            "query": normalized_query,
            "count": min(len(items), limit),
            "stations": items[:limit],
        }

    def search_stations(self, query: str, limit: int = 10) -> dict:
        normalized_query = str(query or "").strip()
        local_result = self._search_stations_local(normalized_query, limit=limit)
        merged: list[dict] = []
        seen: set[tuple[str, str]] = set()

        def extend_items(rows: object) -> None:
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name") or "").strip()
                code = str(row.get("code") or "").strip().upper()
                if not name or not code:
                    continue
                key = (name, code)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(
                    {
                        "name": name,
                        "code": code,
                        "pinyin": str(row.get("pinyin") or "").strip(),
                        "py_short": str(row.get("py_short") or "").strip(),
                        "display_name": str(row.get("display_name") or "").strip(),
                    }
                )

        extend_items(local_result.get("stations") or [])

        try:
            if self.mode == "http":
                remote = self._search_stations_http(normalized_query, limit=limit)
                extend_items((remote or {}).get("stations") or [])
            elif self.mode == "stdio-local":
                remote = self._search_stations_stdio(normalized_query, limit=limit)
                extend_items((remote or {}).get("stations") or [])
            elif not merged:
                raise RuntimeError("MCP_12306_URL not set and local 12306 MCP package is unavailable")
        except Exception:
            if not merged and self.mode == "disabled":
                raise RuntimeError("MCP_12306_URL not set and local 12306 MCP package is unavailable")

        return {
            "success": True,
            "query": normalized_query,
            "count": min(len(merged), limit),
            "stations": merged[:limit],
        }

    def _fetch_trip_options_http(self, from_station: str, to_station: str, train_date: str) -> dict:
        result = {"tickets": {}, "transfers": {}, "errors": {}}
        with self._http_session() as session_id:
            try:
                result["tickets"] = self._call_http_tool(
                    session_id,
                    request_id=2,
                    tool_name="query-tickets",
                    arguments={
                        "from_station": from_station,
                        "to_station": to_station,
                        "train_date": train_date,
                    },
                )
            except Exception as exc:
                result["errors"]["tickets"] = str(exc)

            try:
                result["transfers"] = self._call_http_tool(
                    session_id,
                    request_id=3,
                    tool_name="query-transfer",
                    arguments={
                        "from_station": from_station,
                        "to_station": to_station,
                        "train_date": train_date,
                        "middle_station": "",
                        "isShowWZ": "N",
                        "purpose_codes": "00",
                    },
                )
            except Exception as exc:
                result["errors"]["transfers"] = str(exc)
        return result

    def _fetch_trip_options_stdio(self, from_station: str, to_station: str, train_date: str) -> dict:
        result = {"tickets": {}, "transfers": {}, "errors": {}}
        with self._stdio_session() as (proc, out_queue, err_queue):
            tickets: list[dict] = []
            try:
                response = self._stdio_request(
                    proc,
                    out_queue,
                    err_queue,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "get-tickets",
                            "arguments": {
                                "date": train_date,
                                "fromStation": from_station,
                                "toStation": to_station,
                                "limitedNum": 8,
                                "format": "json",
                                "sortFlag": "duration",
                            },
                        },
                    },
                )
                ticket_rows = self._parse_stdio_tool_json(response)
                if isinstance(ticket_rows, list):
                    tickets = [self._map_stdio_ticket(item) for item in ticket_rows if isinstance(item, dict)]
                result["tickets"] = {
                    "success": True,
                    "from_station": from_station,
                    "to_station": to_station,
                    "train_date": train_date,
                    "count": len(tickets),
                    "trains": tickets,
                }
            except Exception as exc:
                result["errors"]["tickets"] = str(exc)

            try:
                response = self._stdio_request(
                    proc,
                    out_queue,
                    err_queue,
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": "get-interline-tickets",
                            "arguments": {
                                "date": train_date,
                                "fromStation": from_station,
                                "toStation": to_station,
                                "limitedNum": 5,
                                "format": "json",
                            },
                        },
                    },
                )
                transfer_rows = self._parse_stdio_tool_json(response)
                transfers = []
                if isinstance(transfer_rows, list):
                    for item in transfer_rows:
                        if not isinstance(item, dict):
                            continue
                        segments = [self._map_stdio_ticket(segment) for segment in item.get("ticketList") or [] if isinstance(segment, dict)]
                        transfers.append(
                            {
                                "middle_station": str(item.get("middle_station_name") or "").strip(),
                                "total_duration": _normalize_duration_text(item.get("lishi") or ""),
                                "wait_time": str(item.get("wait_time") or "").strip(),
                                "segments": segments,
                            }
                        )
                result["transfers"] = {
                    "success": True,
                    "from_station": from_station,
                    "to_station": to_station,
                    "train_date": train_date,
                    "count": len(transfers),
                    "transfers": transfers,
                }
            except Exception as exc:
                result["errors"]["transfers"] = str(exc)
        return result

    def fetch_trip_options(
        self,
        from_station: str,
        to_station: str,
        train_date: str,
        from_station_code: str = "",
        to_station_code: str = "",
    ) -> dict:
        from_candidates: list[str] = []
        to_candidates: list[str] = []
        for candidate in (str(from_station_code or "").strip(), str(from_station or "").strip()):
            if candidate and candidate not in from_candidates:
                from_candidates.append(candidate)
        for candidate in (str(to_station_code or "").strip(), str(to_station or "").strip()):
            if candidate and candidate not in to_candidates:
                to_candidates.append(candidate)

        if not from_candidates or not to_candidates:
            raise RuntimeError("missing station query")

        last_error: Exception | None = None
        for from_candidate in from_candidates:
            for to_candidate in to_candidates:
                try:
                    if self.mode == "http":
                        return self._fetch_trip_options_http(from_candidate, to_candidate, train_date)
                    if self.mode == "stdio-local":
                        return self._fetch_trip_options_stdio(from_candidate, to_candidate, train_date)
                except Exception as exc:
                    last_error = exc
                    continue
        if last_error is not None:
            raise last_error
        raise RuntimeError("MCP_12306_URL not set and local 12306 MCP package is unavailable")
