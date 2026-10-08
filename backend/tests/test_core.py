# -*- coding: utf-8 -*-
"""单元测试：存储 / 导出 / 引擎校验（不依赖外部网络与 LLM）。"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

# 保证可以从项目根以 backend.app 包形式导入
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest

from backend.app import engine
from backend.app import export as export_mod
from backend.app import storage

# ---------- storage ----------


def test_storage_save_and_get(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DB", tmp_path / "plans.db")
    plan = {
        "location": {"city": "西安"},
        "days": 2,
        "weather": {"summary": {"city": "西安"}},
        "itinerary": [{"day_index": 1, "time": "09:00", "activity_type": "attraction", "name": "大雁塔"}],
    }
    pid = storage.save_plan(plan, title="西安 2日游", request={"city": "西安", "days": 2})
    assert pid > 0

    row = storage.get_plan(pid)
    assert row is not None
    assert row["city"] == "西安"
    assert row["days"] == 2
    assert row["title"] == "西安 2日游"
    assert row["payload"]["location"]["city"] == "西安"
    assert row["request"]["days"] == 2
    assert row["payload"]["itinerary"][0]["name"] == "大雁塔"


def test_storage_list_and_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_DB", tmp_path / "plans.db")
    p1 = storage.save_plan({"location": {"city": "北京"}}, title="北京游")
    p2 = storage.save_plan({"location": {"city": "上海"}}, title="上海游")
    items = storage.list_plans(limit=10)
    ids = [i["id"] for i in items]
    assert p2 in ids and p1 in ids
    # 列表不带 payload，只带摘要
    assert all("payload" not in i for i in items)

    assert storage.delete_plan(p1) is True
    assert storage.get_plan(p1) is None
    assert storage.delete_plan(p1) is False  # 已删除


def test_storage_save_with_date_object(tmp_path, monkeypatch):
    """build_plan 输出里可能带 date 对象，default=str 必须能序列化。"""
    monkeypatch.setattr(storage, "_DB", tmp_path / "plans.db")
    plan = {"location": {"city": "西安"}, "days": 1, "when": datetime.date.today()}
    pid = storage.save_plan(plan)
    row = storage.get_plan(pid)
    assert row["payload"]["when"] == datetime.date.today().isoformat()


# ---------- export ----------


def test_export_plan_html_basic():
    plan = {
        "location": {"city": "西安"},
        "days": 2,
        "weather": {"summary": {"city": "西安", "weather_label_zh": "晴", "max_temp_c": 26, "min_temp_c": 16}},
        "trip_weather": [
            {"day_label": "第1天", "weather_label_zh": "晴", "max_temp_c": 26, "min_temp_c": 16},
            {"day_label": "第2天", "weather_label_zh": "多云", "max_temp_c": 24, "min_temp_c": 15},
        ],
        "itinerary": [
            {"day_index": 1, "time": "09:00", "activity_type": "attraction", "name": "大雁塔", "theme_label": "地标景点", "est_cost": "低"},
            {"day_index": 1, "time": "12:30", "activity_type": "food", "name": "肉夹馍", "theme_label": "本地特色"},
            {"day_index": 2, "time": "10:00", "activity_type": "attraction", "name": "兵马俑", "theme_label": "人文景点"},
        ],
        "attractions": [{"name": "大雁塔", "theme_label": "地标景点", "est_cost": "低", "distance_km": 1.2}],
        "foods": [{"name": "肉夹馍", "category": "snack", "est_cost": "低"}],
        "hotels": [{"name": "某酒店", "price": 300}],
        "budget_hint": "推荐标准预算",
    }
    html_doc = export_mod.export_plan_html(plan, meta={"strategy": "llm", "score": 21, "latency_ms": 3500})
    assert "<!doctype html>" in html_doc
    assert "西安 2日游攻略" in html_doc
    assert "DAY 1" in html_doc and "DAY 2" in html_doc
    # activity_type 中文化
    assert "景点" in html_doc and "美食" in html_doc
    assert "attraction" not in html_doc and "food" not in html_doc
    assert "推荐景点" in html_doc and "推荐美食" in html_doc
    assert "某酒店" in html_doc
    assert "质量分 21" in html_doc and "耗时 3.5s" in html_doc
    assert "大雁塔" in html_doc and "兵马俑" in html_doc


def test_export_plan_html_escapes():
    """HTML 注入必须被转义。"""
    plan = {
        "location": {"city": "西安"},
        "days": 1,
        "itinerary": [{"day_index": 1, "time": "09:00", "activity_type": "attraction", "name": "<script>alert(1)</script>"}],
    }
    html_doc = export_mod.export_plan_html(plan)
    assert "<script>alert" not in html_doc
    assert "&lt;script&gt;" in html_doc


# ---------- engine 校验 ----------


def test_engine_plan_requires_travel_date():
    with pytest.raises(ValueError):
        engine.build_plan({"city": "西安", "days": 2})


def test_engine_plan_days_bounds():
    with pytest.raises(ValueError):
        engine.build_plan({"city": "西安", "travel_date": "2026-10-10", "days": 0})
    with pytest.raises(ValueError):
        engine.build_plan({"city": "西安", "travel_date": "2026-10-10", "days": 15})


def test_engine_plan_check_structure():
    """plan-check 返回结构化字段（不触发外部网络的关键路径）。"""
    data = engine.build_plan_check(
        {
            "city": "西安",
            "travel_date": "2026-10-10",
            "days": 2,
            "budget_level": "standard",
            "planning_mode": "candidate",
            "include_hotel": False,
        }
    )
    for key in ("weather", "location", "planning_mode", "estimated_days", "selection_guidance", "selection_summary", "transport"):
        assert key in data, f"plan-check 缺少字段 {key}"
    assert data["planning_mode"] == "candidate"


def test_engine_budget_level_defaults():
    """非法预算值应回退 standard 而不崩溃。"""
    data = engine.build_plan_check(
        {
            "city": "西安",
            "travel_date": "2026-10-10",
            "days": 1,
            "budget_level": "invalid-level",
        }
    )
    assert "warnings" in data


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
