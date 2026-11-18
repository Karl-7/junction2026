"""Human baselines.

1. Marie Curie ground truth (scripted expert protocol, with her reason for each decision):
       python -m llm_bench.human protocol --seeds 101 102 103

2. Play yourself in the terminal; every call asks for your one-sentence reason:
       python -m llm_bench.human play --seed 101 --name alice
   Commands look like tool calls:  precipitate sample_id=ore reagent=reagent_A
   `help` lists tools, `samples` = list_samples, `quit` ends without submitting.
"""
from __future__ import annotations

import argparse
import json

from hidden_element.config import get_config
from hidden_element.lab import Lab
from hidden_element.rules import rules_text
from policies import reference

from .agent import _now, build_record, save_record
from .tools import TOOLS

PROTOCOL_AGENT = {"id": "human-expert-protocol", "label": "Marie Curie (ground truth)", "family": "Human",
                  "provider": "script", "kind": "human"}
LIST_ARGS = {"sample_ids"}
FLOAT_ARGS = {"fraction", "hours", "minutes", "atomic_weight"}


def protocol(seeds, config="default-v1"):
    for s in seeds:
        lab = Lab(s, get_config(config))
        reference.run(lab)
        rec = build_record(lab, PROTOCOL_AGENT, status="submitted", missing_reasons=0)
        print(save_record(rec), f"reward={lab.final_reward():.3f}")


def _parse(line: str):
    tool, *rest = line.split()
    kw = {}
    for item in rest:
        k, _, v = item.partition("=")
        if k in LIST_ARGS:
            kw[k] = [x for x in v.split(",") if x]
        elif k in FLOAT_ARGS:
            kw[k] = float(v)
        else:
            kw[k] = v
    return tool, kw


def play(seed: int, name: str, config="default-v1"):
    lab = Lab(seed, get_config(config))
    print(rules_text(lab.cfg))
    print(json.dumps({k: v for k, v in lab.call("rules", reason="read the rules").items() if k != "rules"}, indent=1))
    started = _now()
    while not lab.done:
        try:
            line = input("\nlab> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        if line == "help":
            for t in TOOLS:
                args = [k for k in t["parameters"]["properties"] if k != "reason"]
                print(f"  {t['name']}({', '.join(args)}): {t['description']}")
            continue
        if line == "samples":
            line = "list_samples"
        try:
            tool, kw = _parse(line)
        except ValueError as e:
            print("could not parse:", e)
            continue
        reason = ""
        while not reason:
            reason = input("  key reason (one sentence)> ").strip()
        print(json.dumps(lab.call(tool, reason=reason, **kw), indent=1))
    agent = {"id": f"human-{name}", "label": f"Human ({name})", "family": "Human", "provider": "human", "kind": "human"}
    rec = build_record(lab, agent, status="submitted" if lab.result and "submitted" in lab.result else "quit",
                       started_at=started, finished_at=_now(), missing_reasons=0)
    print("saved", save_record(rec), "reward", lab.final_reward())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("protocol")
    p1.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    p2 = sub.add_parser("play")
    p2.add_argument("--seed", type=int, required=True)
    p2.add_argument("--name", required=True)
    a = ap.parse_args()
    if a.cmd == "protocol":
        protocol(a.seeds)
    else:
        play(a.seed, a.name)


if __name__ == "__main__":
    main()
