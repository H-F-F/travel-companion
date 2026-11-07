# -*- coding: utf-8 -*-
"""账号与鉴权：PBKDF2 密码哈希 + 标准库实现的 HS256 JWT。

设计取舍：
- 不引入额外依赖（PyJWT / passlib / bcrypt），只用标准库，部署零负担；
- 令牌默认 7 天有效，密钥从环境变量 ``JWT_SECRET`` 读取；
- 鉴权是「可选」的：无 token 视为游客，沿用原有匿名能力（不破坏演示与旧数据）。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import storage
from .llm_client import _load_local_env

PBKDF2_ITERATIONS = 120_000
TOKEN_TTL_SECONDS = 7 * 24 * 3600
_DEFAULT_SECRET = "travel-companion-dev-secret-change-me"
_USERNAME_RE = re.compile(r"[\w\u4e00-\u9fff]{2,20}", re.UNICODE)  # 中文/字母/数字/下划线，2~20 位


# ---------------- 密码哈希 ----------------

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def hash_password(password: str) -> str:
    """PBKDF2-SHA256 加盐哈希，返回 ``pbkdf2_sha256$salt$hash``。"""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}".format(_b64url(salt), _b64url(digest))


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, salt_b64, digest_b64 = (stored or "").split("$", 2)
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    try:
        salt = _b64url_decode(salt_b64)
        expected = _b64url_decode(digest_b64)
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return hmac.compare_digest(actual, expected)


# ---------------- JWT（HS256） ----------------

def _secret() -> str:
    _load_local_env()
    return os.environ.get("JWT_SECRET") or _DEFAULT_SECRET


def create_token(user_id: int, username: str, ttl: int = TOKEN_TTL_SECONDS) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": str(user_id), "username": username, "iat": now, "exp": now + int(ttl)}
    seg = (
        _b64url(json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    )
    signature = hmac.new(_secret().encode("utf-8"), seg.encode("utf-8"), hashlib.sha256).digest()
    return seg + "." + _b64url(signature)


def decode_token(token: str) -> dict[str, Any] | None:
    """校验签名与过期时间；失败返回 None。"""
    try:
        parts = (token or "").split(".")
        if len(parts) != 3:
            return None
        seg = parts[0] + "." + parts[1]
        signature = hmac.new(_secret().encode("utf-8"), seg.encode("utf-8"), hashlib.sha256).digest()
        if not hmac.compare_digest(signature, _b64url_decode(parts[2])):
            return None
        payload = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
        if not isinstance(payload, dict) or not payload.get("sub"):
            return None
        if int(payload.get("exp") or 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


# ---------------- 输入校验 ----------------

def validate_username(username: str) -> str:
    value = (username or "").strip()
    if not _USERNAME_RE.fullmatch(value):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名需为 2~20 位中文、字母或数字")
    return value


def validate_password(password: str) -> str:
    value = password or ""
    if not (6 <= len(value) <= 64):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="密码长度需为 6~64 位")
    return value


# ---------------- FastAPI 依赖 ----------------

_bearer = HTTPBearer(auto_error=False)


def optional_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict[str, Any] | None:
    """有合法 token 返回用户信息；无 token 返回 None（游客模式）。"""
    if credentials is None or not (credentials.credentials or "").strip():
        return None
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已过期，请重新登录")
    try:
        user_id = int(str(payload.get("sub")))
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的登录凭证")
    user = storage.get_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="账号不存在，请重新登录")
    return user


def require_user(user: dict[str, Any] | None = Depends(optional_user)) -> dict[str, Any]:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录")
    return user
