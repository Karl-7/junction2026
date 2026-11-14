"""MCP server (streamable HTTP). Runs in its own container; the agent only sees the tools.

Env:
  LAB_SEED       episode seed (never sent to the agent)
  LAB_CONFIG     config name (default "default-v0")
  GRADER_TOKEN   shared secret for /_grade (known to the verifier only)
  LAB_LOG_DIR    where the full trajectory (with hidden truth) is written after every call
"""
from __future__ import annotations

import hmac
import json
import os
import threading
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import get_config
from .lab import Lab
from .rules import rules_text

CFG = get_config(os.environ.get("LAB_CONFIG", "default-v0"))
SEED = int(os.environ.get("LAB_SEED", "0"))
TOKEN = os.environ.get("GRADER_TOKEN", "")
LOG_DIR = Path(os.environ.get("LAB_LOG_DIR", "/var/lab"))

_lab = Lab(SEED, CFG)
_lock = threading.Lock()

mcp = FastMCP(
    "hidden-element-lab",
    instructions=rules_text(CFG),
    host=os.environ.get("LAB_HOST", "0.0.0.0"),
    port=int(os.environ.get("LAB_PORT", "8000")),
)


def _save():
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "trajectory.json").write_text(json.dumps(_lab.trajectory(), default=str))
    except OSError:
        pass


def _call(tool: str, **args) -> dict:
    with _lock:
        out = _lab.call(tool, **args)
        _save()
        return out


@mcp.tool()
def rules() -> dict:
    """Free. Full rules, scoring formula, budgets, public data on known elements and reagents."""
    return _call("rules")


@mcp.tool()
def list_samples() -> dict:
    """Free. Current sample ids and states, reagent units left, actions left, simulated clock."""
    return _call("list_samples")


@mcp.tool()
def dissolve(sample_id: str) -> dict:
    """Dissolve a solid sample (same id). Loses the per-operation loss fraction."""
    return _call("dissolve", sample_id=sample_id)


@mcp.tool()
def precipitate(sample_id: str, reagent: str) -> dict:
    """Add a reagent to a solution and filter: returns a solid precipitate and a filtrate solution."""
    return _call("precipitate", sample_id=sample_id, reagent=reagent)


@mcp.tool()
def crystallize(sample_id: str, fraction: float) -> dict:
    """Fractional crystallisation of a solution: crystals hold `fraction` (0.05-0.95) of the mass."""
    return _call("crystallize", sample_id=sample_id, fraction=fraction)


@mcp.tool()
def combine(sample_ids: list[str]) -> dict:
    """Merge two or more samples of the same state into one."""
    return _call("combine", sample_ids=sample_ids)


@mcp.tool()
def wait(hours: float) -> dict:
    """Let simulated time pass; radioactive elements decay."""
    return _call("wait", hours=hours)


@mcp.tool()
def measure_activity(sample_id: str, minutes: float = 1.0) -> dict:
    """Count radioactivity for `minutes`. Poisson counting noise plus background; subtract background."""
    return _call("measure_activity", sample_id=sample_id, minutes=minutes)


@mcp.tool()
def weigh(sample_id: str) -> dict:
    """Mass of a sample in grams (noisy)."""
    return _call("weigh", sample_id=sample_id)


@mcp.tool()
def spectrum(sample_id: str) -> dict:
    """Emission lines (nm, intensity) of elements above their detection limit."""
    return _call("spectrum", sample_id=sample_id)


@mcp.tool()
def atomic_weight(sample_id: str) -> dict:
    """Apparent (mass-fraction-weighted) atomic weight of the whole sample (noisy)."""
    return _call("atomic_weight", sample_id=sample_id)


@mcp.tool()
def submit(sample_id: str, atomic_weight: float, carrier: str) -> dict:
    """End the episode: submit a sample, the target's atomic weight, and its carrier's symbol."""
    return _call("submit", sample_id=sample_id, atomic_weight=atomic_weight, carrier=carrier)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


@mcp.custom_route("/_grade", methods=["GET"])
async def grade(request: Request) -> JSONResponse:
    given = request.headers.get("x-grader-token", "")
    if not TOKEN or not hmac.compare_digest(given, TOKEN):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    with _lock:
        if not _lab.done:
            _lab._finish(None)   # grading ends the episode; no submission -> 0
        _save()
        payload = {"reward": _lab.final_reward(), "result": _lab.result, "trajectory": _lab.trajectory()}
    return JSONResponse(json.loads(json.dumps(payload, default=str)))


def main():
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
