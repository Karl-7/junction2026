"""Collect every run under results/runs/, compute metrics, and write docs/data.js for the site.

    python -m llm_bench.site
Open docs/index.html (leaderboard) and docs/runs.html (step-by-step reasons, side by side).
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, RUNS_DIR
from .metrics import CONCEPTS, PHASES, STRATEGY, summarize

DOCS = ROOT / "docs"
FAMILY_ORDER = ["Baseline", "Human", "Claude", "GPT", "Gemini"]
GROUNDED_KEYS = ["grs", "decision_accuracy", "evidence_consistency", "claim_truth", "faithfulness",
                 "prediction_accuracy", "prediction_rate"]
COUNT_KEYS = ["n_decisions", "n_measured_decisions", "n_claims", "n_faithfulness", "n_predictions"]
CATEGORIES = ["correct", "correct_unmeasured", "against_evidence_correct", "misled_by_evidence",
              "against_evidence", "wrong_unmeasured"]


def load_runs(runs_dir: Path = RUNS_DIR) -> list:
    return [json.loads(p.read_text()) for p in sorted(runs_dir.glob("*/seed*.json"))]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def collect(runs_dir: Path = RUNS_DIR) -> dict:
    runs = load_runs(runs_dir)
    humans_by_seed = defaultdict(list)
    for r in runs:
        if r["agent"]["kind"] == "human":          # real players only, never the scripted baseline
            humans_by_seed[r["seed"]].append(r)
    all_humans = [h for hs in humans_by_seed.values() for h in hs]

    out_runs = []
    for r in runs:
        refs = humans_by_seed.get(r["seed"]) or all_humans
        m = summarize(r, refs)
        phases, checks = m.pop("phases"), m.pop("checks")
        steps = []
        for s, ph, ck in zip(r["steps"], phases, checks):
            st = {"i": s["i"], "tool": s["tool"], "args": s["args"], "reason": s.get("reason"),
                  "result": s["result"], "phase": ph}
            if s.get("prediction"):
                st["prediction"] = s["prediction"]
            if ck:
                st["checks"] = ck
            steps.append(st)
        out_runs.append({"agent": r["agent"], "seed": r["seed"], "status": r.get("status", ""),
                         "usage": r.get("usage"), "final_message": r.get("final_message", ""),
                         "truth": r["truth"], "metrics": m, "steps": steps})

    agents = {}
    for r in out_runs:
        agents.setdefault(r["agent"]["id"], {**r["agent"], "runs": []})["runs"].append(r)
    table = []
    for a in agents.values():
        good = [r for r in a["runs"] if not r["status"].startswith("error")]
        ms = [r["metrics"] for r in good]
        gs = [m["grounded"] for m in ms]
        sims = [m["similarity"] for m in ms if "similarity" in m]
        row = {k: a[k] for k in ("id", "label", "family", "provider", "kind")}
        row.update({
            "seeds": sorted(r["seed"] for r in good),
            "errors": len(a["runs"]) - len(good),
            "reward": _mean([m["reward"] for m in ms]),
            "purity": _mean([m["purity"] for m in ms]),
            "recovery": _mean([m["recovery"] for m in ms]),
            "carrier": _mean([float(m["carrier_correct"]) for m in ms]),
            "weight_error": _mean([m["weight_error"] for m in ms]),
            "actions": _mean([m["actions"] for m in ms]),
            "reason_coverage": _mean([m["reasons"]["coverage"] for m in ms]),
            "reason_words": _mean([m["reasons"]["mean_words"] for m in ms]),
            "grounded": {k: _mean([g[k] for g in gs]) for k in GROUNDED_KEYS},
            "counts": {k: sum(g[k] for g in gs) for k in COUNT_KEYS},
            "categories": {c: sum(g["decision_categories"].get(c, 0) for g in gs) for c in CATEGORIES},
            "strategy": {k: _mean([float(m["strategy"][k]) for m in ms]) for k in STRATEGY},
            "concepts": {k: _mean([m["concepts"][k] for m in ms]) for k in CONCEPTS},
            "similarity": {k: _mean([s[k] for s in sims]) for k in
                           ("human_likeness", "strategy", "sequence", "tool_mix", "concepts")} if sims else None,
        })
        table.append(row)
    fam = {f: i for i, f in enumerate(FAMILY_ORDER)}
    table.sort(key=lambda r: (fam.get(r["family"], 9), -((r["grounded"]["grs"]) or 0)))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "strategy": STRATEGY, "concepts": list(CONCEPTS), "phases": PHASES, "categories": CATEGORIES,
        "families": FAMILY_ORDER, "has_humans": bool(all_humans), "agents": table, "runs": out_runs,
    }


def build(runs_dir: Path = RUNS_DIR, out: Path = DOCS) -> Path:
    data = collect(runs_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "data.js"
    p.write_text("window.BENCH = " + json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str) + ";\n")
    return p


if __name__ == "__main__":
    print(build())
