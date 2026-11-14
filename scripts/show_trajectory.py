"""Print a lab trajectory as a raw transcript: every tool call, its arguments and the raw response.

    python -m scripts.show_trajectory path/to/trajectory.json [--truth]
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
    for e in d["log"]:
        args = ", ".join(f"{k}={v!r}" for k, v in e["args"].items())
        res = dict(e["result"])
        if e["tool"] == "rules":
            res = {"rules": "<rules text omitted>", **{k: v for k, v in res.items() if k != "rules"}}
        print(f"[{e['i']:>3}] {e['tool']}({args})")
        print("      -> " + json.dumps(res, separators=(",", ":")))
    if a.truth:
        print("\n=== result ===")
        print(json.dumps(d["result"], indent=1))
        t = d["truth"]
        print("=== hidden truth ===")
        print(json.dumps({"carrier": t["carrier"], "interferer_carrier": t["interferer_carrier"],
                          "target_atomic_weight": t["elements"]["target"]["atomic_weight"]}, indent=1))


if __name__ == "__main__":
    main()
