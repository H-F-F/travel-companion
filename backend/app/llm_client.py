"""OpenAI 兼容 LLM 客户端。

支持任意提供 OpenAI 兼容 /chat/completions 接口的厂商
（DeepSeek / 通义千问 / 智谱 / 豆包 Ark / Moonshot 等），通过环境变量配置：

- LLM_BASE_URL : OpenAI 兼容端点根地址，例如 https://api.deepseek.com/v1
- LLM_API_KEY  : API Key
- LLM_MODEL    : 模型名
- LLM_TIMEOUT  : 请求超时（秒，默认 30）

未配置或调用失败时抛出 LLMError，由上层降级到规则引擎，保证服务不中断。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import json

_loaded_env = False


def _load_local_env() -> None:
    """从项目根目录 .env / backend/.env 加载环境变量（与 train_12306_client 一致）。"""
    global _loaded_env
    if _loaded_env:
        return
    _loaded_env = True
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


class LLMError(RuntimeError):
    """LLM 调用失败（未配置、网络、鉴权、响应解析等）。"""


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        _load_local_env()
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).strip().rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "")
        raw_timeout = timeout if timeout is not None else os.environ.get("LLM_TIMEOUT", "30")
        try:
            self.timeout = float(raw_timeout)
        except (TypeError, ValueError):
            self.timeout = 30.0
        # 累计 token 用量（供 Agent 返回可观测元数据）
        self.total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.calls: int = 0

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.4,
    ) -> dict[str, Any]:
        """调用 chat/completions，返回 OpenAI 消息对象（含 content / tool_calls）。"""
        if not self.available:
            raise LLMError("LLM 未配置：需要设置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                resp = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM 请求失败: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"LLM 返回 HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
            usage = data.get("usage") or {}
            self.calls += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                try:
                    self.total_usage[key] = self.total_usage.get(key, 0) + int(usage.get(key) or 0)
                except (TypeError, ValueError):
                    pass
            return data["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM 响应解析失败: {exc}") from exc

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.4,
    ) -> Any:
        """流式调用 chat/completions，逐帧产出文本片段。

        生成器事件（tuple[str, Any]）：
          ("delta", text)   —— 本轮生成的一段文本（无工具调用时）
          ("message", msg)  —— 整包消息（工具轮：含 tool_calls，已完整累积）
          ("error", msg)    —— 失败原因（调用方应抛出 LLMError）

        工具轮处理：OpenAI 兼容接口首帧出现 delta.tool_calls 时，本方法累积
        完整 tool_calls 后以 ("message", ...) 返回，等价于一次非流式调用，
        保证工具编排逻辑与 chat() 完全一致。
        """
        if not self.available:
            yield ("error", "LLM 未配置：需要设置 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL")
            return

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        tool_call_mode: bool | None = None  # None=未知, True=工具轮, False=文本轮
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}

        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        body = resp.read().decode("utf-8", errors="replace")[:300]
                        yield ("error", f"LLM 返回 HTTP {resp.status_code}: {body}")
                        return
                    for line in resp.iter_lines():
                        line = (line or "").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("usage"):
                            usage = chunk.get("usage") or {}
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                            if tool_call_mode is None:
                                tool_call_mode = False
                            yield ("delta", delta["content"])
                        if delta.get("tool_calls"):
                            tool_call_mode = True
                            for tc in delta["tool_calls"]:
                                idx = int(tc.get("index") or 0)
                                acc = tool_calls_acc.setdefault(
                                    idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                                )
                                if tc.get("id"):
                                    acc["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    acc["function"]["name"] += fn["name"]
                                if fn.get("arguments"):
                                    acc["function"]["arguments"] += fn["arguments"]
        except httpx.HTTPError as exc:
            yield ("error", f"LLM 请求失败: {exc}")
            return

        self.calls += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            try:
                self.total_usage[key] = self.total_usage.get(key, 0) + int(usage.get(key) or 0)
            except (TypeError, ValueError):
                pass

        if tool_call_mode:
            tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
            yield ("message", {"role": "assistant", "content": "".join(content_parts), "tool_calls": tool_calls})
            return
        # 文本轮：整包消息仅作结束标记（内容已逐帧 yield）
        yield ("message", {"role": "assistant", "content": "".join(content_parts)})
