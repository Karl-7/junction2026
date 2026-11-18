"""Print a run as a transcript: every tool call, the key reason given for it, and the raw response.

    python -m scripts.show_trajectory results/runs/<agent>/seed101.json [--truth]
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--truth", action="store_true", help="also print the score and hidden truth")
    a = ap.parse_args()
    d = json.load(open(a.path))
    for e in d.get("log") or d["steps"]:
        args = ", ".join(f"{k}={v!r}" for k, v in e["args"].items())
        res = dict(e["result"])
        if e["tool"] == "rules":
            res = {"rules": "<rules text omitted>", **{k: v for k, v in res.items() if k != "rules"}}
        print(f"[{e['i']:>3}] {e['tool']}({args})")
        if e.get("reason"):
            print(f"      why: {e['reason']}")
        print("      -> " + json.dumps(res, separators=(",", ":")))
    if a.truth:
        print("\n=== result ===")
        print(json.dumps(d["result"], indent=1))
        t = d["truth"]
        w = t["elements"]["target"]["atomic_weight"] if "elements" in t else t["target_atomic_weight"]
        print("=== hidden truth ===")
        print(json.dumps({"carrier": t["carrier"], "interferer_carrier": t["interferer_carrier"],
                          "target_atomic_weight": w}, indent=1))


if __name__ == "__main__":
    main()
