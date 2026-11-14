"""Gemini as the agent: function calling over the lab's MCP tools. Saves every raw API response.

    GEMINI_API_KEY=... python -m policies.gemini_agent --url http://127.0.0.1:8801/mcp \
        --model gemini-3.5-flash --out runs/gemini/seed101.jsonl

The model sees only the task instruction and tool results, exactly like the Harbor agent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hidden_element.config import get_config  # noqa: E402
from hidden_element.rules import rules_text  # noqa: E402
from policies.mcp_env import McpEnv  # noqa: E402
from scripts.build_tasks import INSTRUCTION  # noqa: E402

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

S = lambda d="": {"type": "string", "description": d}  # noqa: E731
N = lambda d="": {"type": "number", "description": d}  # noqa: E731


def _tool(name, desc, props=None, required=None):
    t = {"name": name, "description": desc}
    if props:
        t["parameters"] = {"type": "object", "properties": props, "required": required or list(props)}
    return t


TOOLS = [
    _tool("rules", "Free. Full rules, scoring formula, budgets, public data on known elements and reagents."),
    _tool("list_samples", "Free. Current sample ids and states, reagent units left, actions left, clock."),
    _tool("dissolve", "Dissolve a solid sample (same id).", {"sample_id": S()}),
    _tool("precipitate", "Add a reagent to a solution and filter into a solid precipitate and a filtrate solution.",
          {"sample_id": S(), "reagent": S("e.g. reagent_A")}),
    _tool("crystallize", "Fractional crystallisation of a solution; crystals hold `fraction` (0.05-0.95) of the mass.",
          {"sample_id": S(), "fraction": N()}),
    _tool("combine", "Merge two or more samples of the same state.",
          {"sample_ids": {"type": "array", "items": {"type": "string"}}}),
    _tool("wait", "Let simulated time pass (hours); radioactive elements decay.", {"hours": N()}),
    _tool("measure_activity", "Count radioactivity for `minutes` (0.1-60). Subtract the background.",
          {"sample_id": S(), "minutes": N()}),
    _tool("weigh", "Mass of a sample in grams (noisy).", {"sample_id": S()}),
    _tool("spectrum", "Emission lines (nm, intensity) of elements above their detection limit.", {"sample_id": S()}),
    _tool("atomic_weight", "Apparent (mass-fraction-weighted) atomic weight of the whole sample.", {"sample_id": S()}),
    _tool("submit", "End the episode: submit a sample, the target's atomic weight and the carrier's symbol.",
          {"sample_id": S(), "atomic_weight": N(), "carrier": S()}),
]


def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY")
    if not k and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        sys.exit("GEMINI_API_KEY not set")
    return k


def _post(model: str, key: str, body: dict, log) -> dict:
    req = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    delay = 10.0
    for attempt in range(12):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            quota = []
            try:
                for det in json.loads(msg)["error"].get("details", []):
                    quota += [v.get("quotaId") for v in det.get("violations", [])]
            except (ValueError, KeyError, TypeError):
                pass
            log({"event": "http_error", "code": e.code, "attempt": attempt, "quota": quota, "message": msg})
            if any("PerDay" in (q or "") for q in quota):
                raise RuntimeError(f"Gemini daily quota exhausted: {quota}")   # retrying cannot help
            if e.code in (429, 500, 503):
                time.sleep(delay)
                delay = min(delay * 1.7, 120)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            log({"event": "net_error", "attempt": attempt, "message": str(e)})
            time.sleep(delay)
    raise RuntimeError("Gemini API kept failing")


def run(url: str, model: str, out: Path, max_requests: int = 150, config: str = "default-v0") -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w")

    def log(rec):
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()

    key = _key()
    env = McpEnv(url)
    instruction = INSTRUCTION.format(rules=rules_text(get_config(config)))
    contents = [{"role": "user", "parts": [{"text": instruction + "\nBegin. Use the tools; finish with submit."}]}]
    log({"event": "start", "model": model, "url": url, "instruction": instruction})
    submitted, nudges = None, 0
    try:
        for n in range(max_requests):
            body = {"contents": contents, "tools": [{"functionDeclarations": TOOLS}],
                    "generationConfig": {"thinkingConfig": {"includeThoughts": True}}}
            resp = _post(model, key, body, log)
            log({"event": "response", "n": n, "raw": resp})
            cand = (resp.get("candidates") or [{}])[0]
            content = cand.get("content") or {"role": "model", "parts": []}
            content.setdefault("role", "model")
            contents.append(content)   # verbatim, keeps thought signatures
            calls = [p["functionCall"] for p in content.get("parts", []) if "functionCall" in p]
            if not calls:
                if nudges >= 3:
                    break
                nudges += 1
                contents.append({"role": "user", "parts": [{"text":
                    "Continue working with the lab tools. The episode only ends when you call submit."}]})
                continue
            parts = []
            for c in calls:
                name, args = c.get("name"), c.get("args") or {}
                try:
                    result = env.call(name, **args)
                except Exception as e:  # tool transport problems are reported to the model
                    result = {"error": f"tool call failed: {e}"}
                log({"event": "tool", "n": n, "name": name, "args": args, "result": result})
                parts.append({"functionResponse": {"name": name, "response": {"result": result}}})
                if name == "submit" and "error" not in result:
                    submitted = {"args": args, "result": result}
            contents.append({"role": "user", "parts": parts})
            if submitted:
                break
    finally:
        env.close()
        log({"event": "end", "submitted": submitted})
        fh.close()
    return submitted or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-requests", type=int, default=150)
    a = ap.parse_args()
    print(json.dumps(run(a.url, a.model, Path(a.out), a.max_requests)))


if __name__ == "__main__":
    main()
