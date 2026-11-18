"""Run one model on one seed and save a run record with the key reason behind every decision.

    python -m llm_bench.run --models claude-opus-5-5 gpt-5 gemini-3.5-flash --seeds 101 102 103

The model only ever sees the task prompt and tool results. The lab (with its hidden state) lives
in this process; hidden truth is written to the run record only after the episode ends.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from hidden_element.config import get_config
from hidden_element.lab import Lab

from .config import RAW_DIR, RUNS_DIR, ModelSpec
from .providers import ProviderError, make_provider
from .tools import PREDICTION_FIELDS, SYSTEM_PROMPT, TOOL_NAMES, task_prompt

NUDGE = "Continue working with the lab tools (remember the one-sentence `reason`). The episode only ends when you call submit."
MISSING_REASON = ("Your call was executed, but it had no `reason`. Every call must include a one-sentence "
                  "`reason` giving the key reason for the decision.")


_THROTTLE: dict = {}
_THROTTLE_LOCK = threading.Lock()


def _throttle(spec: ModelSpec):
    """Space requests to stay under the model's requests-per-minute limit (shared across threads)."""
    if not spec.rpm:
        return
    gap = 60.0 / spec.rpm * 1.05
    with _THROTTLE_LOCK:
        nxt = max(time.time(), _THROTTLE.get(spec.id, 0.0))
        _THROTTLE[spec.id] = nxt + gap
    time.sleep(max(0.0, nxt - time.time()))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_record(lab: Lab, agent: dict, **extra) -> dict:
    """Common run record for LLMs, human players and the scripted expert protocol."""
    if not lab.done:
        lab._finish(None)
    traj = lab.trajectory()
    steps = []
    for e in traj["log"]:
        res = dict(e["result"])
        if e["tool"] == "rules":   # the long rules text is the same for everyone
            res = {k: v for k, v in res.items() if k != "rules"}
        st = {"i": e["i"], "tool": e["tool"], "args": e["args"], "reason": e.get("reason"), "result": res}
        if e.get("prediction"):
            st["prediction"] = e["prediction"]
        steps.append(st)
    truth = traj["truth"]
    return {
        "agent": agent, "seed": lab.seed, "config": lab.cfg.name,
        "steps": steps, "actions": lab.actions, "time_h": round(lab.t, 2),
        "result": lab.result,
        "truth": {"carrier": truth["carrier"], "interferer_carrier": truth["interferer_carrier"],
                  "target_atomic_weight": truth["elements"]["target"]["atomic_weight"]},
        **extra,
    }


def save_record(rec: dict, runs_dir: Path = RUNS_DIR) -> Path:
    p = runs_dir / rec["agent"]["id"].replace("/", "_") / f"seed{rec['seed']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    return p


def run_episode(spec: ModelSpec, seed: int, *, config: str = "default-v1", max_requests: int = 200,
                runs_dir: Path = RUNS_DIR, raw_dir: Path = RAW_DIR, provider=None, verbose: bool = True) -> dict:
    cfg = get_config(config)
    lab = Lab(seed, cfg)
    raw_path = raw_dir / spec.slug / f"seed{seed}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    fh = raw_path.open("w")

    def log(rec):
        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        fh.flush()

    started, t0 = _now(), time.time()
    usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}
    status, final_text, nudges, missing = "max_requests", "", 0, 0
    try:
        p = provider or make_provider(spec, log)
        p.start(SYSTEM_PROMPT, task_prompt(cfg))
        log({"event": "start", "model": spec.id, "seed": seed})
        for n in range(max_requests):
            _throttle(spec)
            turn = p.step()
            usage["requests"] += 1
            for k in ("input_tokens", "output_tokens"):
                usage[k] += int(turn.usage.get(k, 0) or 0)
            log({"event": "response", "n": n, "raw": turn.raw})
            if turn.text:
                final_text = turn.text
            if not turn.calls:
                if nudges >= 3:
                    status = "gave_up"
                    break
                nudges += 1
                p.add_user(NUDGE)
                continue
            results, no_reason = [], False
            for c in turn.calls:
                args = dict(c.args)
                reason = args.pop("reason", None)
                reason = str(reason).strip() if reason is not None else None
                prediction = {k: args.pop(k) for k in list(args) if k in PREDICTION_FIELDS}
                prediction = {k: v for k, v in prediction.items() if v not in (None, "", "unsure")}
                if c.parse_error:
                    res = {"error": c.parse_error}
                elif c.name not in TOOL_NAMES:
                    res = {"error": f"unknown tool {c.name!r}"}
                else:
                    res = lab.call(c.name, reason=reason or None, **args)
                    if lab.log:
                        lab.log[-1]["thinking"] = (turn.thinking or turn.text)[:2000] or None
                        if prediction:
                            lab.log[-1]["prediction"] = prediction
                if not reason:
                    no_reason = True
                    missing += 1
                results.append((c, res))
                if verbose:
                    print(f"  [{spec.label} s{seed}] {c.name}({', '.join(f'{k}={v}' for k, v in args.items())})"
                          f"  — {reason or '(no reason)'}", flush=True)
                if lab.done:
                    break
            if no_reason and not lab.done:
                results[-1] = (results[-1][0], {**results[-1][1], "note": MISSING_REASON})
            p.add_results(results)
            if lab.done:
                status = "submitted" if lab.result and lab.result.get("reason") != "no submission" else "budget_exhausted"
                break
    except ProviderError as e:
        status = f"error: {e}"
        log({"event": "error", "message": str(e)})
    finally:
        fh.close()

    agent = {"id": spec.id, "label": spec.label, "family": spec.family, "provider": spec.provider, "kind": "llm"}
    rec = build_record(lab, agent, status=status, started_at=started, finished_at=_now(),
                       wall_s=round(time.time() - t0, 1), usage=usage, final_message=final_text[:4000],
                       missing_reasons=missing)
    # keep the per-step thinking excerpt next to the reason (useful for spotting reason/action mismatch)
    for st, e in zip(rec["steps"], lab.log):
        if e.get("thinking"):
            st["thinking"] = e["thinking"]
    if not status.startswith("error") or rec["steps"]:
        save_record(rec, runs_dir)
    return rec
