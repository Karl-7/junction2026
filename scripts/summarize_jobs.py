"""Summarise Harbor job outputs produced by scripts/run_harbor.sh.

Reads each trial's verifier/grade.json (written by our verifier from the lab server) and
prints one row per (seed, kind) plus a per-kind mean.

    python -m scripts.summarize_jobs jobs --prefix 20261114-120000 [--csv out.csv]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

JOB_RE = re.compile(r"-s(\d+)-(reference|early-stop|do-nothing|agent)$")


def collect(jobs: Path, prefix: str):
    rows = []
    for job in sorted(p for p in jobs.iterdir() if p.is_dir() and p.name.startswith(prefix)):
        m = JOB_RE.search(job.name)
        if not m:
            continue
        seed, kind = int(m.group(1)), m.group(2)
        for rt in sorted(job.rglob("verifier/reward.txt")):
            trial = rt.parent.parent
            row = {"seed": seed, "kind": kind, "trial": trial.name, "reward": 0.0}
            try:
                row["reward"] = float(rt.read_text().strip() or 0)
            except ValueError:
                pass
            g = rt.parent / "grade.json"
            if g.exists():
                d = json.loads(g.read_text())
                res = d.get("result") or {}
                tr = d.get("trajectory") or {}
                row.update(purity=res.get("purity"), recovery=res.get("recovery"),
                           actions=tr.get("actions"), reason=res.get("reason", ""))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--csv")
    a = ap.parse_args()
    rows = collect(Path(a.jobs), a.prefix)
    fmt = lambda v: "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    print(f"{'seed':>4} {'kind':<11} {'reward':>7} {'purity':>7} {'recov':>7} {'acts':>5}  trial")
    for r in sorted(rows, key=lambda r: (r["seed"], r["kind"])):
        print(f"{r['seed']:>4} {r['kind']:<11} {fmt(r['reward']):>7} {fmt(r.get('purity')):>7} "
              f"{fmt(r.get('recovery')):>7} {fmt(r.get('actions')):>5}  {r['trial']}")
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r["reward"])
    print()
    for k, v in sorted(by.items()):
        sd = statistics.pstdev(v) if len(v) > 1 else 0.0
        print(f"{k:<11} n={len(v):<3} mean={statistics.mean(v):.3f} sd={sd:.3f}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "kind", "trial", "reward", "purity", "recovery", "actions", "reason"])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
