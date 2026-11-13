"""Run the reference policy and baselines on N seeds in-process and print a score table.

    python -m scripts.eval_table --seeds 20 [--logdir runs/local]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hidden_element.config import get_config  # noqa: E402
from hidden_element.lab import Lab  # noqa: E402
from policies import baselines, reference  # noqa: E402

POLICIES = {
    "reference": reference.run,
    "do_nothing": baselines.do_nothing,
    "submit_ore": baselines.submit_ore,
    "one_step": baselines.one_step,
    "random": None,  # seeded per episode below
}


def run_one(name: str, seed: int, cfg):
    lab = Lab(seed, cfg)
    if name == "random":
        baselines.random_ops(lab, seed=seed)
    else:
        POLICIES[name](lab)
    if not lab.done:
        lab._finish(None)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    cfg = get_config()
    seeds = range(a.start, a.start + a.seeds)
    print(f"config={cfg.name} seeds={a.start}..{a.start + a.seeds - 1}")
    print(f"{'policy':<12}{'mean':>8}{'min':>8}{'max':>8}{'purity':>9}{'recov':>8}{'actions':>9}")
    for name in POLICIES:
        rs, ps, rec, acts = [], [], [], []
        for s in seeds:
            lab = run_one(name, s, cfg)
            res = lab.result or {}
            rs.append(lab.final_reward()); ps.append(res.get("purity", 0.0))
            rec.append(res.get("recovery", 0.0)); acts.append(lab.actions)
            if a.verbose and name == "reference":
                print(f"  seed {s}: r={rs[-1]:.3f} purity={ps[-1]:.3f} rec={rec[-1]:.3f} "
                      f"w={res.get('weight_credit', 0):.2f} c={res.get('carrier_credit', 0):.0f} acts={lab.actions}")
            if a.logdir:
                d = Path(a.logdir) / name
                d.mkdir(parents=True, exist_ok=True)
                (d / f"seed{s}.json").write_text(json.dumps(lab.trajectory(), indent=1, default=str))
        print(f"{name:<12}{statistics.mean(rs):>8.3f}{min(rs):>8.3f}{max(rs):>8.3f}"
              f"{statistics.mean(ps):>9.3f}{statistics.mean(rec):>8.3f}{statistics.mean(acts):>9.1f}")


if __name__ == "__main__":
    main()
