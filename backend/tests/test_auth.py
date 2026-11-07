# -*- coding: utf-8 -*-
"""账号体系测试：密码哈希、JWT、行程归属隔离、注册/登录接口。"""
import time

import pytest
from fastapi.testclient import TestClient

from backend.app import auth, storage
from backend.app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DB", tmp_path / "plans.db")
    with TestClient(app) as c:
        yield c


# ---------------- 纯函数 ----------------

def test_password_hash_roundtrip():
    stored = auth.hash_password("secret123")
    assert stored.startswith("pbkdf2_sha256$")
    assert auth.verify_password("secret123", stored)
    assert not auth.verify_password("secret124", stored)
    assert not auth.verify_password("secret123", "bad-format")
    assert not auth.verify_password("secret123", "")


def test_token_roundtrip_and_tamper():
    token = auth.create_token(7, "胡飞飞")
    payload = auth.decode_token(token)
    assert payload and payload["sub"] == "7" and payload["username"] == "胡飞飞"
    # 篡改签名 / 结构
    assert auth.decode_token(token[:-2] + "xx") is None
    assert auth.decode_token("a.b.c") is None
    assert auth.decode_token("") is None
    # 过期
    expired = auth.create_token(7, "test", ttl=-1)
    assert auth.decode_token(expired) is None


# ---------------- 存储层 ----------------

def test_storage_user_crud(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DB", tmp_path / "plans.db")
    uid = storage.create_user("alice", auth.hash_password("pw123456"))
    assert uid is not None
    assert storage.create_user("alice", auth.hash_password("other")) is None  # 重名
    user = storage.get_user(uid)
    assert user["username"] == "alice" and "password_hash" not in user
    raw = storage.get_user_by_username("alice")
    assert raw["password_hash"].startswith("pbkdf2_sha256$")


def test_storage_plan_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DB", tmp_path / "plans.db")
    u1 = storage.create_user("u1", auth.hash_password("pw123456"))
    u2 = storage.create_user("u2", auth.hash_password("pw123456"))
    guest_pid = storage.save_plan({"location": {"city": "西安"}})
    p1 = storage.save_plan({"location": {"city": "北京"}}, user_id=u1)
    p2 = storage.save_plan({"location": {"city": "上海"}}, user_id=u2)

    assert [p["city"] for p in storage.list_plans(user_id=u1)] == ["北京"]
    assert [p["city"] for p in storage.list_plans(user_id=u2)] == ["上海"]
    assert [p["city"] for p in storage.list_plans()] == ["西安"]

    assert storage.get_plan(p1, user_id=u1) is not None
    assert storage.get_plan(p1, user_id=u2) is None       # 跨账号不可见
    assert storage.get_plan(p1) is None                    # 游客不可见账号数据
    assert storage.get_plan(guest_pid, user_id=u1) is None

    assert storage.delete_plan(p1, user_id=u2) is False
    assert storage.delete_plan(p1, user_id=u1) is True


def test_storage_session_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DB", tmp_path / "plans.db")
    u1 = storage.create_user("s1", auth.hash_password("pw123456"))
    sid = storage.save_session_messages([{"role": "user", "content": "去西安"}], user_id=u1)
    assert storage.get_session_messages(sid, user_id=u1) is not None
    assert storage.get_session_messages(sid) is None
    # 用别人的 session_id 续写：不能改写他人会话
    other = storage.save_session_messages([{"role": "user", "content": "偷偷续写"}], sid)
    assert other != sid
    assert storage.get_session_messages(sid, user_id=u1)["messages"][0]["content"] == "去西安"


# ---------------- 接口层 ----------------

def test_register_login_me(client):
    r = client.post("/api/v1/auth/register", json={"username": "tester", "password": "pw123456"})
    assert r.status_code == 200
    token = r.json()["token"]
    assert r.json()["user"]["username"] == "tester"

    # 重名
    assert client.post("/api/v1/auth/register", json={"username": "tester", "password": "pw123456"}).status_code == 409
    # 密码错误
    assert client.post("/api/v1/auth/login", json={"username": "tester", "password": "wrongpw"}).status_code == 401
    # 正确登录
    r = client.post("/api/v1/auth/login", json={"username": "tester", "password": "pw123456"})
    assert r.status_code == 200
    token = r.json()["token"]

    r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.json()["username"] == "tester"
    # 无 token
    assert client.get("/api/v1/auth/me").status_code == 401
    # 伪造 token
    assert client.get("/api/v1/auth/me", headers={"Authorization": "Bearer x.y.z"}).status_code == 401


def test_plan_api_user_isolation(client):
    u1 = client.post("/api/v1/auth/register", json={"username": "iso1", "password": "pw123456"}).json()
    u2 = client.post("/api/v1/auth/register", json={"username": "iso2", "password": "pw123456"}).json()
    h1 = {"Authorization": f"Bearer {u1['token']}"}
    h2 = {"Authorization": f"Bearer {u2['token']}"}

    pid = storage.save_plan({"location": {"city": "北京"}}, user_id=u1["user"]["id"])
    assert client.get(f"/api/v1/plans/{pid}", headers=h1).status_code == 200
    assert client.get(f"/api/v1/plans/{pid}", headers=h2).status_code == 404
    assert client.get(f"/api/v1/plans/{pid}").status_code == 404

    r1 = client.get("/api/v1/plans", headers=h1).json()["items"]
    r2 = client.get("/api/v1/plans", headers=h2).json()["items"]
    assert len(r1) == 1 and len(r2) == 0

    assert client.delete(f"/api/v1/plans/{pid}", headers=h2).status_code == 404
    assert client.delete(f"/api/v1/plans/{pid}", headers=h1).status_code == 200
