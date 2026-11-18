"""Per-run metrics.

Primary (llm_bench/grounded.py): the grounded reasoning metrics, checked against the hidden state.
Secondary (here):
  strategy      a checklist of what a careful radiochemist does (waiting out the interferer,
                comparing both products, fractional crystallisation, recycling mother liquors, ...)
  concepts      which scientific concepts the one-sentence reasons invoke (descriptive)
  similarity    to *real human players* on the same seed, when any exist: agreement on the strategy
                checklist, phase-sequence similarity, tool-mix similarity, concept similarity.
                The scripted Curie-style baseline is not a human and is never used as reference.
"""
from __future__ import annotations

import math
import re
from collections import Counter

PHASES = ["orient", "decay", "separation", "crystallisation", "recycling", "analysis", "submit"]
LAB_TOOLS = ["dissolve", "precipitate", "crystallize", "combine", "wait", "measure_activity",
             "weigh", "spectrum", "atomic_weight", "submit"]

STRATEGY = {
    "waited_out_interferer": "Waited ≥ 60 h (several interferer half-lives) before crystallising",
    "compared_both_products": "Measured both products of most separations",
    "fractional_crystallisation": "Used repeated fractional crystallisation (≥ 3 rounds)",
    "recycled_mother_liquor": "Recovered target from a mother liquor and merged it back",
    "checked_spectrum": "Checked the spectrum during purification",
    "used_budget": "Did not stop early (purity goal reached or ≤ 25 actions left)",
    "submitted": "Submitted a sample",
}

CONCEPTS = {
    "background": r"background",
    "decay / half-life": r"half-?li|decay|short-lived|interferer",
    "activity signal": r"activit|cpm|\bcount|radioactiv|signal",
    "specific activity": r"specific activit|per gram|per unit mass|normali[sz]",
    "carrier": r"carrier",
    "spectral lines": r"spectr|emission|\blines?\b|unassigned",
    "recovery / mother liquor": r"recover|mother liquor|recycl|\bloss|losing",
    "purity / enrichment": r"purit|\bpure|contamina|impur|enrich",
    "budget": r"budget|actions? left|remaining action|reagent unit",
    "noise / uncertainty": r"nois|poisson|uncertain|statistic|re-?measure|repeat|confirm|within error",
    "atomic weight": r"atomic weight",
    "scoring": r"\bscor|reward|goal|threshold",
}
_CONCEPT_RE = {k: re.compile(v, re.I) for k, v in CONCEPTS.items()}


# ============================================================================ per-run features
def ok_steps(run: dict) -> list:
    return [s for s in run["steps"] if "error" not in s["result"]]


def label_phases(run: dict) -> list:
    """Phase for every step (errors get the phase of what they attempted)."""
    mothers, recovered, started = set(), set(), False
    out = []
    for s in run["steps"]:
        t, a, r = s["tool"], s["args"], s["result"]
        ok = "error" not in r
        if t in ("rules", "list_samples"):
            ph = "orient"
        elif t == "wait":
            ph = "decay"
        elif t == "precipitate":
            ph = "separation"
        elif t == "crystallize":
            src = a.get("sample_id")
            ph = "recycling" if src in mothers or src in recovered else "crystallisation"
            started = True
            if ok:
                mothers.add(r["mother_liquor"])
                if ph == "recycling":
                    recovered.add(r["crystals"])
        elif t == "combine":
            ph = "recycling"
        elif t in ("dissolve", "measure_activity"):
            sid = a.get("sample_id")
            ph = "recycling" if sid in recovered else ("crystallisation" if started else "separation")
        elif t in ("spectrum", "atomic_weight", "weigh"):
            ph = "analysis"
        elif t == "submit":
            ph = "submit"
        else:
            ph = "orient"
        out.append(ph)
    return out


def strategy(run: dict) -> dict:
    steps = ok_steps(run)
    res = run.get("result") or {}
    cfg_goal = 0.9
    waited, first_cry = 0.0, None
    for s in steps:
        if s["tool"] == "crystallize":
            first_cry = s
            break
        if s["tool"] == "wait":
            waited += float(s["args"].get("hours", 0))
    measured = {s["args"].get("sample_id") for s in steps if s["tool"] == "measure_activity"}
    precs = [s["result"] for s in steps if s["tool"] == "precipitate"]
    both = sum(1 for r in precs if r["precipitate"] in measured and r["filtrate"] in measured)
    crys = [s for s in steps if s["tool"] == "crystallize"]
    derived = set()
    for s in crys:
        derived.add(s["result"]["mother_liquor"])
        if s["args"].get("sample_id") in derived:
            derived.add(s["result"]["crystals"])
    recycled = any(set(s["args"].get("sample_ids", [])) & derived for s in steps if s["tool"] == "combine")
    after_cry = False
    spec_after = False
    for s in steps:
        after_cry = after_cry or s["tool"] == "crystallize"
        spec_after = spec_after or (after_cry and s["tool"] == "spectrum")
    submit = next((s for s in steps if s["tool"] == "submit"), None)
    left = submit["result"].get("actions_left", 0) if submit else 0
    submitted = bool(submit) and "purity" in res
    return {
        "waited_out_interferer": waited >= 60 and first_cry is not None,
        "compared_both_products": bool(precs) and both / len(precs) >= 0.6,
        "fractional_crystallisation": len(crys) >= 3,
        "recycled_mother_liquor": recycled,
        "checked_spectrum": spec_after,
        "used_budget": submitted and (left <= 25 or res.get("purity", 0) >= cfg_goal),
        "submitted": submitted,
    }


def concept_profile(run: dict) -> dict:
    reasons = [s["reason"] for s in run["steps"] if s.get("reason")]
    n = max(1, len(reasons))
    return {k: sum(1 for r in reasons if rx.search(r)) / n for k, rx in _CONCEPT_RE.items()}


def tool_mix(run: dict) -> dict:
    c = Counter(s["tool"] for s in ok_steps(run) if s["tool"] in LAB_TOOLS)
    n = sum(c.values()) or 1
    return {t: c.get(t, 0) / n for t in LAB_TOOLS}


def phase_sequence(run: dict) -> list:
    seq = []
    for s, ph in zip(run["steps"], label_phases(run)):
        if ph == "orient" or "error" in s["result"]:
            continue
        if not seq or seq[-1] != ph:
            seq.append(ph)
    return seq


def reason_stats(run: dict) -> dict:
    steps = run["steps"]
    rs = [s.get("reason") or "" for s in steps]
    given = [r for r in rs if r.strip()]
    words = [len(r.split()) for r in given]
    one_sentence = [r for r in given if len(re.findall(r"[.!?](\s|$)", r.strip())) <= 1]
    return {"coverage": len(given) / len(steps) if steps else 0.0,
            "mean_words": sum(words) / len(words) if words else 0.0,
            "one_sentence": len(one_sentence) / len(given) if given else 0.0}


# ============================================================================ similarity
def _edit_sim(a: list, b: list) -> float:
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))


def _js_sim(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    if not any(p.values()) or not any(q.values()):
        return 0.0
    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in keys}

    def kl(x):
        return sum(x[k] * math.log2(x[k] / m[k]) for k in keys if x.get(k, 0) > 0)
    js = (kl(p) + kl(q)) / 2
    return 1 - math.sqrt(max(0.0, js))


def _cos(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    dot = sum(p.get(k, 0) * q.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in p.values()))
    nb = math.sqrt(sum(v * v for v in q.values()))
    return dot / (na * nb) if na and nb else 0.0


def similarity(run: dict, human: dict) -> dict:
    da, db = strategy(run), strategy(human)
    s = {
        "strategy": sum(da[k] == db[k] for k in STRATEGY) / len(STRATEGY),
        "sequence": _edit_sim(phase_sequence(run), phase_sequence(human)),
        "tool_mix": _js_sim(tool_mix(run), tool_mix(human)),
        "concepts": _cos(concept_profile(run), concept_profile(human)),
    }
    s["human_likeness"] = sum(s.values()) / 4
    return s


def summarize(run: dict, humans: list) -> dict:
    """Per-run metrics. `humans` are real human runs to compare with (same seed preferred)."""
    from .grounded import evaluate
    res = run.get("result") or {}
    truth_w = run["truth"]["target_atomic_weight"]
    sub = res.get("submitted") or {}
    g = evaluate(run)
    out = {
        "reward": res.get("reward", 0.0),
        "purity": res.get("purity", 0.0),
        "recovery": res.get("recovery", 0.0),
        "carrier_correct": bool(res.get("carrier_credit")),
        "weight_error": abs(sub["atomic_weight"] - truth_w) / truth_w if sub.get("atomic_weight") is not None else None,
        "actions": run.get("actions", 0),
        "grounded": g["metrics"],
        "checks": g["checks"],
        "strategy": strategy(run),
        "concepts": concept_profile(run),
        "phases": label_phases(run),
        "reasons": reason_stats(run),
    }
    refs = [h for h in humans if h is not run and h["agent"]["id"] != run["agent"]["id"]]
    if refs:
        sims = [similarity(run, h) for h in refs]
        out["similarity"] = {k: sum(s[k] for s in sims) / len(sims) for k in sims[0]}
    return out
