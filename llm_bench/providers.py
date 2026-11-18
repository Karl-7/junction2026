"""Thin tool-calling adapters for Anthropic, OpenAI and Gemini over plain HTTPS (stdlib only).

Each adapter keeps its own conversation in the provider's native format, so provider-specific
state (thinking blocks, thought signatures, reasoning items) is passed back verbatim.

Common interface:
    p = make_provider(spec, log)
    p.start(system, user_text)
    turn = p.step()                 # -> Turn(calls=[ToolCall], text=str, usage={...})
    p.add_results([(call, result_dict), ...])   # or p.add_user(text) when there were no calls
"""
from __future__ import annotations

import copy
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from .config import ModelSpec
from .tools import TOOLS

Log = Callable[[dict], None]


class ProviderError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    parse_error: str | None = None


@dataclass
class Turn:
    calls: list[ToolCall]
    text: str = ""
    thinking: str = ""
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def _key(var: str) -> str:
    k = os.environ.get(var, "").strip()
    if not k:
        raise ProviderError(f"{var} is not set (put it in .env)")
    return k


def post_json(url: str, headers: dict, body: dict, log: Log, attempts: int = 8, timeout: int = 600) -> dict:
    """POST with exponential backoff on rate limits, overload and server errors."""
    data = json.dumps(body).encode()
    delay = 5.0
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            log({"event": "http_error", "code": e.code, "attempt": attempt, "message": msg[:4000]})
            if "PerDay" in msg or "insufficient_quota" in msg:
                raise ProviderError(f"quota exhausted (HTTP {e.code}): {msg[:300]}")
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529) and attempt < attempts - 1:
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise ProviderError(f"HTTP {e.code}: {msg[:500]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log({"event": "net_error", "attempt": attempt, "message": str(e)})
            if attempt == attempts - 1:
                raise ProviderError(f"network error: {e}")
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise ProviderError("request kept failing")


def _parse_args(raw) -> tuple[dict, str | None]:
    if isinstance(raw, dict):
        return raw, None
    try:
        v = json.loads(raw or "{}")
        return (v, None) if isinstance(v, dict) else ({}, "arguments must be a JSON object")
    except json.JSONDecodeError as e:
        return {}, f"arguments are not valid JSON: {e}"


def _result_text(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False)


# ============================================================================ Anthropic
class AnthropicProvider:
    """Messages API. The latest message gets a cache breakpoint so the growing prefix is cached."""

    def __init__(self, spec: ModelSpec, log: Log):
        self.spec, self.log = spec, log
        self.url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        self.headers = {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"}
        self.tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                      for t in TOOLS]
        self.system: list = []
        self.messages: list = []

    def start(self, system: str, user: str):
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self.messages = [{"role": "user", "content": [{"type": "text", "text": user}]}]

    def step(self) -> Turn:
        msgs = copy.deepcopy(self.messages)
        last = msgs[-1]["content"]
        if isinstance(last, list) and last:
            last[-1]["cache_control"] = {"type": "ephemeral"}
        body = {"model": self.spec.id, "max_tokens": self.spec.max_tokens, "system": self.system,
                "tools": self.tools, "messages": msgs, **self.spec.extra}
        resp = post_json(self.url, self.headers, body, self.log)
        content = resp.get("content", [])
        self.messages.append({"role": "assistant", "content": content})
        calls = [ToolCall(b["id"], b["name"], b.get("input") or {}) for b in content if b.get("type") == "tool_use"]
        text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
        thinking = "\n".join(b.get("thinking", "") for b in content if b.get("type") == "thinking").strip()
        u = resp.get("usage", {})
        usage = {"input_tokens": u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                 + u.get("cache_creation_input_tokens", 0), "output_tokens": u.get("output_tokens", 0)}
        return Turn(calls, text, thinking, usage, resp)

    def add_results(self, results):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": c.id, "content": _result_text(r), **({"is_error": True} if "error" in r else {})}
            for c, r in results]})

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})


# ============================================================================ OpenAI
class OpenAIProvider:
    """Responses API by default (keeps reasoning state server-side via previous_response_id).
    Set OPENAI_API_MODE=chat (or `"api": "chat"` in the model's extra) for Chat Completions,
    e.g. for OpenAI-compatible endpoints."""

    def __init__(self, spec: ModelSpec, log: Log):
        self.spec, self.log = spec, log
        extra = dict(spec.extra)
        self.mode = extra.pop("api", os.environ.get("OPENAI_API_MODE", "responses"))
        self.extra = extra
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.url = base + ("/responses" if self.mode == "responses" else "/chat/completions")
        self.headers = {"Authorization": f"Bearer {_key('OPENAI_API_KEY')}"}
        if self.mode == "responses":
            self.tools = [{"type": "function", "name": t["name"], "description": t["description"],
                           "parameters": t["parameters"]} for t in TOOLS]
        else:
            self.tools = [{"type": "function", "function": t} for t in TOOLS]
        self.instructions = ""
        self.pending: list = []
        self.prev_id: str | None = None
        self.messages: list = []

    def start(self, system: str, user: str):
        self.instructions = system
        self.pending = [{"role": "user", "content": user}]
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def step(self) -> Turn:
        if self.mode == "responses":
            body = {"model": self.spec.id, "instructions": self.instructions, "input": self.pending,
                    "tools": self.tools, **self.extra}
            if self.prev_id:
                body["previous_response_id"] = self.prev_id
            resp = post_json(self.url, self.headers, body, self.log)
            self.prev_id = resp.get("id")
            self.pending = []
            calls, texts, thoughts = [], [], []
            for item in resp.get("output", []):
                if item.get("type") == "function_call":
                    args, err = _parse_args(item.get("arguments"))
                    calls.append(ToolCall(item["call_id"], item["name"], args, err))
                elif item.get("type") == "message":
                    texts += [c.get("text", "") for c in item.get("content", []) if "text" in c]
                elif item.get("type") == "reasoning":
                    thoughts += [s.get("text", "") for s in item.get("summary", [])]
            u = resp.get("usage", {})
            usage = {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)}
            return Turn(calls, "\n".join(texts).strip(), "\n".join(thoughts).strip(), usage, resp)

        body = {"model": self.spec.id, "messages": self.messages, "tools": self.tools, **self.extra}
        resp = post_json(self.url, self.headers, body, self.log)
        msg = (resp.get("choices") or [{}])[0].get("message") or {"role": "assistant", "content": ""}
        self.messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls = []
        for tc in msg.get("tool_calls") or []:
            args, err = _parse_args(tc["function"].get("arguments"))
            calls.append(ToolCall(tc["id"], tc["function"]["name"], args, err))
        u = resp.get("usage", {})
        usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}
        return Turn(calls, (msg.get("content") or "").strip(), "", usage, resp)

    def add_results(self, results):
        if self.mode == "responses":
            self.pending += [{"type": "function_call_output", "call_id": c.id, "output": _result_text(r)}
                             for c, r in results]
        else:
            self.messages += [{"role": "tool", "tool_call_id": c.id, "content": _result_text(r)} for c, r in results]

    def add_user(self, text: str):
        if self.mode == "responses":
            self.pending.append({"role": "user", "content": text})
        else:
            self.messages.append({"role": "user", "content": text})


# ============================================================================ Gemini
class GeminiProvider:
    """generateContent with function calling. Model turns are appended verbatim (thought signatures)."""

    def __init__(self, spec: ModelSpec, log: Log):
        self.spec, self.log = spec, log
        base = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.url = f"{base}/models/{spec.id}:generateContent"
        self.headers = {"x-goog-api-key": _key("GEMINI_API_KEY")}
        self.tools = [{"functionDeclarations": TOOLS}]
        self.system = ""
        self.contents: list = []

    def start(self, system: str, user: str):
        self.system = system
        self.contents = [{"role": "user", "parts": [{"text": user}]}]

    def step(self) -> Turn:
        body = {"systemInstruction": {"parts": [{"text": self.system}]}, "contents": self.contents,
                "tools": self.tools, "generationConfig": {"thinkingConfig": {"includeThoughts": True}}}
        for k, v in self.spec.extra.items():
            body[k] = {**body[k], **v} if isinstance(v, dict) and isinstance(body.get(k), dict) else v
        resp = post_json(self.url, self.headers, body, self.log)
        cand = (resp.get("candidates") or [{}])[0]
        content = cand.get("content") or {"role": "model", "parts": []}
        content.setdefault("role", "model")
        content.setdefault("parts", [])
        self.contents.append(content)
        calls, texts, thoughts = [], [], []
        for i, p in enumerate(content["parts"]):
            if "functionCall" in p:
                fc = p["functionCall"]
                calls.append(ToolCall(fc.get("id") or f"call_{len(self.contents)}_{i}", fc.get("name", ""),
                                      fc.get("args") or {}))
            elif "text" in p:
                (thoughts if p.get("thought") else texts).append(p["text"])
        u = resp.get("usageMetadata", {})
        usage = {"input_tokens": u.get("promptTokenCount", 0),
                 "output_tokens": u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)}
        return Turn(calls, "\n".join(texts).strip(), "\n".join(thoughts).strip(), usage, resp)

    def add_results(self, results):
        parts = []
        for c, r in results:
            fr = {"name": c.name, "response": {"result": r}}
            if not c.id.startswith("call_"):
                fr["id"] = c.id
            parts.append({"functionResponse": fr})
        self.contents.append({"role": "user", "parts": parts})

    def add_user(self, text: str):
        self.contents.append({"role": "user", "parts": [{"text": text}]})


PROVIDERS = {"anthropic": AnthropicProvider, "openai": OpenAIProvider, "gemini": GeminiProvider}


def make_provider(spec: ModelSpec, log: Log):
    if spec.provider not in PROVIDERS:
        raise ProviderError(f"unknown provider {spec.provider!r}")
    return PROVIDERS[spec.provider](spec, log)
