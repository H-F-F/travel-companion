# -*- coding: utf-8 -*-
"""行程持久化：SQLite 存储（列表 / 详情 / 删除 / 保存），线程安全。"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_DB = Path(__file__).resolve().parents[2] / "data" / "plans.db"
_LOCK = threading.Lock()


def _conn() -> sqlite3.Connection:
    _DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS plans(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            city TEXT DEFAULT '',
            days INTEGER DEFAULT 0,
            title TEXT DEFAULT '',
            summary TEXT DEFAULT '',
            payload TEXT NOT NULL,
            request TEXT DEFAULT ''
        )"""
    )
    # 迁移：旧表无 request 列时补充
    cols = [r[1] for r in conn.execute("PRAGMA table_info(plans)").fetchall()]
    if "request" not in cols:
        conn.execute("ALTER TABLE plans ADD COLUMN request TEXT DEFAULT ''")
    return conn


def save_plan(plan: dict, title: str = "", request: dict | None = None) -> int:
    """保存一份完整行程 plan，返回自增 id。request 为生成时的请求参数（前端渲染历史用）。"""
    payload = json.dumps(plan, ensure_ascii=False, default=str)
    request_json = json.dumps(request or {}, ensure_ascii=False, default=str)
    weather = plan.get("weather") or {}
    summary = plan.get("summary") or {}
    city = (
        summary.get("city")
        or weather.get("city")
        or (plan.get("location") or {}).get("city")
        or ""
    )
    days = int(plan.get("days") or plan.get("requested_days") or plan.get("estimated_days") or 0)
    with _LOCK:
        conn = _conn()
        try:
            cur = conn.execute(
                "INSERT INTO plans(created_at, city, days, title, summary, payload, request) VALUES(?,?,?,?,?,?,?)",
                (
                    time.strftime("%Y-%m-%d %H:%M"),
                    city,
                    days,
                    title,
                    json.dumps(summary, ensure_ascii=False),
                    payload,
                    request_json,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def list_plans(limit: int = 50) -> list[dict]:
    with _LOCK:
        conn = _conn()
        try:
            rows = conn.execute(
                "SELECT id, created_at, city, days, title, summary FROM plans ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    d["summary"] = json.loads(d["summary"] or "{}")
                except Exception:
                    d["summary"] = {}
                out.append(d)
            return out
        finally:
            conn.close()


def get_plan(pid: int) -> dict | None:
    with _LOCK:
        conn = _conn()
        try:
            row = conn.execute("SELECT * FROM plans WHERE id=?", (int(pid),)).fetchone()
            if not row:
                return None
            d = dict(row)
            try:
                d["payload"] = json.loads(d["payload"])
            except Exception:
                d["payload"] = {}
            try:
                d["request"] = json.loads(d.get("request") or "{}")
            except Exception:
                d["request"] = {}
            try:
                d["summary"] = json.loads(d["summary"] or "{}")
            except Exception:
                d["summary"] = {}
            return d
        finally:
            conn.close()


def delete_plan(pid: int) -> bool:
    with _LOCK:
        conn = _conn()
        try:
            cur = conn.execute("DELETE FROM plans WHERE id=?", (int(pid),))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


# ---------------- 对话会话（Session） ----------------

def _sconn() -> sqlite3.Connection:
    _DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            messages TEXT NOT NULL
        )"""
    )
    return conn


def save_session_messages(messages: list[dict], session_id: int | None = None) -> int:
    """保存 / 追加一轮会话消息，返回 session_id（None 时新建会话）。"""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    messages_json = json.dumps(messages or [], ensure_ascii=False, default=str)
    with _LOCK:
        if session_id is not None:
            conn = _sconn()
            try:
                row = conn.execute("SELECT id FROM sessions WHERE id=?", (int(session_id),)).fetchone()
                if row:
                    conn.execute(
                        "UPDATE sessions SET updated_at=?, messages=? WHERE id=?",
                        (now, messages_json, int(session_id)),
                    )
                    conn.commit()
                    return int(session_id)
            finally:
                conn.close()
        conn = _sconn()
        try:
            cur = conn.execute(
                "INSERT INTO sessions(created_at, updated_at, messages) VALUES(?,?,?)",
                (now, now, messages_json),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def get_session_messages(session_id: int) -> dict | None:
    with _LOCK:
        conn = _sconn()
        try:
            row = conn.execute(
                "SELECT id, created_at, updated_at, messages FROM sessions WHERE id=?", (int(session_id),)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            try:
                d["messages"] = json.loads(d["messages"] or "[]")
            except Exception:
                d["messages"] = []
            return d
        finally:
            conn.close()


def list_sessions(limit: int = 20) -> list[dict]:
    with _LOCK:
        conn = _sconn()
        try:
            rows = conn.execute(
                "SELECT id, created_at, updated_at, messages FROM sessions ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                try:
                    msgs = json.loads(d["messages"] or "[]")
                except Exception:
                    msgs = []
                # 第一条用户消息作为会话标题
                first_user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                d["title"] = (first_user or "")[:40]
                d["message_count"] = len(msgs)
                d.pop("messages", None)
                out.append(d)
            return out
        finally:
            conn.close()


def delete_session(session_id: int) -> bool:
    with _LOCK:
        conn = _sconn()
        try:
            cur = conn.execute("DELETE FROM sessions WHERE id=?", (int(session_id),))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
