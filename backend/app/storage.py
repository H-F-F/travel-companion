# -*- coding: utf-8 -*-
"""行程持久化：SQLite 存储（行程 / 对话会话 / 用户），线程安全。

多账号设计：
- 所有业务表都带 ``user_id``；``NULL`` 表示游客（匿名）数据，保持旧数据兼容；
- 传 ``user_id=None`` 的查询只命中游客数据，传具体 ID 则只命中该用户数据；
- 旧库首次打开时自动 ``ALTER TABLE`` 补列，无需手工迁移。
"""
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
    # 迁移：旧表缺列时补齐
    cols = [r[1] for r in conn.execute("PRAGMA table_info(plans)").fetchall()]
    if "request" not in cols:
        conn.execute("ALTER TABLE plans ADD COLUMN request TEXT DEFAULT ''")
    if "user_id" not in cols:
        conn.execute("ALTER TABLE plans ADD COLUMN user_id INTEGER")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    return conn


def _owner_clause(user_id: int | None) -> tuple[str, tuple]:
    """按归属生成 WHERE 片段：None=游客（user_id IS NULL），否则精确匹配。"""
    if user_id is None:
        return "user_id IS NULL", ()
    return "user_id = ?", (int(user_id),)


# ---------------- 用户 ----------------

def create_user(username: str, password_hash: str) -> int | None:
    """新建用户；用户名已存在时返回 None。"""
    with _LOCK:
        conn = _conn()
        try:
            cur = conn.execute(
                "INSERT INTO users(username, password_hash, created_at) VALUES(?,?,?)",
                (username, password_hash, time.strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
            return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()


def get_user(user_id: int) -> dict | None:
    """按 ID 取用户（不含密码哈希，供接口返回）。"""
    with _LOCK:
        conn = _conn()
        try:
            row = conn.execute(
                "SELECT id, username, created_at FROM users WHERE id=?", (int(user_id),)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def get_user_by_username(username: str) -> dict | None:
    """按用户名取用户（含密码哈希，仅供登录校验）。"""
    with _LOCK:
        conn = _conn()
        try:
            row = conn.execute(
                "SELECT id, username, password_hash, created_at FROM users WHERE username=?",
                (username,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


# ---------------- 行程 ----------------

def save_plan(plan: dict, title: str = "", request: dict | None = None, user_id: int | None = None) -> int:
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
                "INSERT INTO plans(created_at, city, days, title, summary, payload, request, user_id) VALUES(?,?,?,?,?,?,?,?)",
                (
                    time.strftime("%Y-%m-%d %H:%M"),
                    city,
                    days,
                    title,
                    json.dumps(summary, ensure_ascii=False),
                    payload,
                    request_json,
                    None if user_id is None else int(user_id),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def list_plans(limit: int = 50, user_id: int | None = None) -> list[dict]:
    where, params = _owner_clause(user_id)
    with _LOCK:
        conn = _conn()
        try:
            rows = conn.execute(
                f"SELECT id, created_at, city, days, title, summary FROM plans WHERE {where} ORDER BY id DESC LIMIT ?",
                params + (int(limit),),
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


def get_plan(pid: int, user_id: int | None = None) -> dict | None:
    where, params = _owner_clause(user_id)
    with _LOCK:
        conn = _conn()
        try:
            row = conn.execute(
                f"SELECT * FROM plans WHERE id=? AND {where}", (int(pid),) + params
            ).fetchone()
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


def delete_plan(pid: int, user_id: int | None = None) -> bool:
    where, params = _owner_clause(user_id)
    with _LOCK:
        conn = _conn()
        try:
            cur = conn.execute(f"DELETE FROM plans WHERE id=? AND {where}", (int(pid),) + params)
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
    cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
    if "user_id" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN user_id INTEGER")
    return conn


def save_session_messages(messages: list[dict], session_id: int | None = None, user_id: int | None = None) -> int:
    """保存 / 追加一轮会话消息，返回 session_id（None 或归属不匹配时新建会话）。"""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    messages_json = json.dumps(messages or [], ensure_ascii=False, default=str)
    where, params = _owner_clause(user_id)
    with _LOCK:
        if session_id is not None:
            conn = _sconn()
            try:
                row = conn.execute(
                    f"SELECT id FROM sessions WHERE id=? AND {where}", (int(session_id),) + params
                ).fetchone()
                if row:
                    conn.execute(
                        f"UPDATE sessions SET updated_at=?, messages=? WHERE id=? AND {where}",
                        (now, messages_json, int(session_id)) + params,
                    )
                    conn.commit()
                    return int(session_id)
            finally:
                conn.close()
        conn = _sconn()
        try:
            cur = conn.execute(
                "INSERT INTO sessions(created_at, updated_at, messages, user_id) VALUES(?,?,?,?)",
                (now, now, messages_json, None if user_id is None else int(user_id)),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def get_session_messages(session_id: int, user_id: int | None = None) -> dict | None:
    where, params = _owner_clause(user_id)
    with _LOCK:
        conn = _sconn()
        try:
            row = conn.execute(
                f"SELECT id, created_at, updated_at, messages FROM sessions WHERE id=? AND {where}",
                (int(session_id),) + params,
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


def list_sessions(limit: int = 20, user_id: int | None = None) -> list[dict]:
    where, params = _owner_clause(user_id)
    with _LOCK:
        conn = _sconn()
        try:
            rows = conn.execute(
                f"SELECT id, created_at, updated_at, messages FROM sessions WHERE {where} ORDER BY id DESC LIMIT ?",
                params + (int(limit),),
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


def delete_session(session_id: int, user_id: int | None = None) -> bool:
    where, params = _owner_clause(user_id)
    with _LOCK:
        conn = _sconn()
        try:
            cur = conn.execute(f"DELETE FROM sessions WHERE id=? AND {where}", (int(session_id),) + params)
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
