"""Grounded reasoning metrics: check a run's decisions, stated reasons and predictions against the
hidden state of the simulation, step by step.

The lab is deterministic given (seed, sequence of calls), so a saved run is replayed on a fresh Lab
and the hidden state is inspected before and after every call. Nothing here needs human labels.

Metrics (each in [0, 1], None when nothing in the run could be checked):

  decision_accuracy     At every split (precipitate / crystallize) the agent later continues with one
                        product. Correct if that product holds more of the target (precipitate) or is
                        purer in the target (crystallize). Splits where < 2% of the ore's target is
                        left are ignored: the outcome no longer matters.
  evidence_consistency  Among those decisions where the agent measured both products first: did it
                        follow its own measurements (higher net activity; per unit of mass share for
                        crystallisation)? Separates reasoning errors from being misled by noise.
  claim_truth           Factual claims inside the one-sentence reasons that can be checked against the
                        hidden state (interferer has decayed; X is the carrier; purity is N%; the
                        activity went into the precipitate/filtrate/crystals/mother liquor).
  faithfulness          Reasons that say which product to follow / discard, checked against the
                        product the call actually acts on.
  prediction_accuracy   Optional expect_* fields (where the target goes, which product is enriched,
                        expected net cpm within a factor 1.5) against what actually happened.

grounded_reasoning_score (GRS) = mean of the available values among decision_accuracy,
evidence_consistency, claim_truth and faithfulness.
"""
from __future__ import annotations

import math
import re

from hidden_element.config import get_config
from hidden_element.lab import Lab
from hidden_element.world import INTERFERER, TARGET

PRODUCT_TYPES = ("precipitate", "filtrate", "crystals", "mother liquor")
USE_TOOLS = {"dissolve", "precipitate", "crystallize", "combine", "submit"}
MIN_TARGET_SHARE = 0.02          # of the ore's target, for a split to count as a decision
CPM_TOLERANCE = math.log(1.5)
PURITY_TOLERANCE = 0.15          # absolute, for "purity is N%" claims

_PTYPE = r"(precipitates?|filtrates?|crystals?|mother[ -]liquors?)"
_NEG = re.compile(r"\b(not|n't|no longer|unless|until|whether|if)\b", re.I)
_RE_DECAYED = re.compile(r"(interferer|short-lived)[^.;]*\b(has|have|had|is|are|now)\b[^.;]{0,20}"
                         r"\b(decayed|gone|negligible|dead|died out)\b", re.I)
_RE_CARRIER_A = re.compile(r"\bcarrier\b[^.;]{0,30}?\b(?:is|=|:|be|being)\s+(?:likely\s+|probably\s+)?\b([A-Z][a-z])\b")
_RE_CARRIER_B = re.compile(r"\b([A-Z][a-z])\b\s+(?:is|as)\s+(?:the\s+|its\s+|our\s+)?(?:likely\s+|probable\s+)?carrier\b")
_RE_PURITY = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:pure|purity)|purity[^.;\d]{0,25}?(\d{1,3}(?:\.\d+)?)\s*%", re.I)
_RE_WENT = re.compile(r"\b(activity|target|signal)\b[^.;]{0,40}?\b(went|goes|is|stays|stayed|remains|remained|"
                      r"followed|follows|ended up|ends up|concentrated|sits)\b[^.;]{0,15}?\b(in|into|with)\s+(?:the\s+)?" + _PTYPE, re.I)
_RE_FOLLOW = re.compile(r"\b(follow|keep|continue with|proceed with|work with|go with|take|carry forward|redissolve)\s+(?:the\s+)?"
                        + _PTYPE, re.I)
_RE_DISCARD = re.compile(r"\b(discard|drop|abandon|throw away|set aside)\s+(?:the\s+)?" + _PTYPE, re.I)


def _ptype(word: str) -> str:
    w = word.lower().replace("-", " ")
    for t in PRODUCT_TYPES:
        if w.startswith(t.rstrip("s")):
            return t
    return w


def _origin_type(lab: Lab, sid) -> str | None:
    s = lab.samples.get(sid)
    if not s:
        return None
    o = s["origin"]
    for t in PRODUCT_TYPES:
        if o.startswith(t):
            return t
    return None   # ore or combined


def _hidden(lab: Lab, sid) -> dict:
    m = lab._live(lab.samples[sid]["m"])
    tot = sum(m.values())
    els = lab.world.elements
    act = {k: g * els[k].activity for k, g in m.items()}
    return {"target": m.get(TARGET, 0.0), "total": tot, "purity": m.get(TARGET, 0.0) / tot if tot else 0.0,
            "net_cpm": sum(act.values()), "interferer_share": act.get(INTERFERER, 0.0) / (sum(act.values()) or 1)}


def _interferer_share_all(lab: Lab) -> float:
    els = lab.world.elements
    tot = itf = 0.0
    for s in lab.samples.values():
        for k, g in lab._live(s["m"]).items():
            a = g * els[k].activity
            tot += a
            itf += a if k == INTERFERER else 0.0
    return itf / tot if tot else 0.0


def _ratio(xs):
    """Mean of repeated measurements, None if never measured."""
    return sum(xs) / len(xs) if xs else None


def evaluate(run: dict) -> dict:
    """Replay the run and return metrics plus per-step checks: steps[i] -> list of {kind, ok, note}."""
    cfg = get_config(run.get("config", "default-v1"))
    lab = Lab(run["seed"], cfg)
    bg = cfg.background_cpm
    target0 = lab.world.ore[TARGET]
    symbols = {e["symbol"] for e in lab.world.public()["known_elements"]}
    checks = [[] for _ in run["steps"]]
    pairs: dict = {}            # product id -> pair dict
    decisions, claims, faith, preds = [], [], [], []
    mismatches = 0
    last_pair = None

    for idx, st in enumerate(run["steps"]):
        tool, args, reason = st["tool"], st["args"], st.get("reason") or ""
        ck = checks[idx]
        sid = args.get("sample_id")
        sentence_ok = reason and not _NEG.search(reason)

        # ---------------------------------------------------- claims in the reason (pre-state)
        if reason:
            if _RE_DECAYED.search(reason) and not re.search(r"\b(wait|let|until|so that|allow)\b", reason, re.I):
                share = _interferer_share_all(lab)
                ok = share < 0.05
                claims.append(ok)
                ck.append({"kind": "claim", "ok": ok, "note": f"'interferer has decayed': it is {share:.0%} of activity"})
            for rx in (_RE_CARRIER_A, _RE_CARRIER_B):
                m = rx.search(reason)
                if m and m.group(1) in symbols and not re.search(r"\bnot\b", reason[:m.end()], re.I):
                    ok = m.group(1) == lab.world.carrier
                    claims.append(ok)
                    ck.append({"kind": "claim", "ok": ok, "note": f"carrier claimed {m.group(1)}"
                               + ("" if ok else f", true carrier {lab.world.carrier}")})
                    break
            m = _RE_PURITY.search(reason)
            if m and sid in lab.samples:
                claimed = float(m.group(1) or m.group(2)) / 100
                true = _hidden(lab, sid)["purity"]
                ok = abs(claimed - true) <= PURITY_TOLERANCE
                claims.append(ok)
                ck.append({"kind": "claim", "ok": ok, "note": f"purity claimed {claimed:.0%}, true {true:.0%}"})
            m = _RE_WENT.search(reason)
            if m and last_pair and sentence_ok:
                t = _ptype(m.group(4))
                if t in last_pair["types"]:
                    ok = t == last_pair["best_type"]
                    claims.append(ok)
                    ck.append({"kind": "claim", "ok": ok, "note": f"'target is in the {t}': it is mostly in the {last_pair['best_type']}"})
            # faithfulness: what the reason says to follow / discard vs the sample acted on
            if tool in USE_TOOLS and sid:
                actual = _origin_type(lab, sid)
                m = _RE_FOLLOW.search(reason)
                if m and actual and not _NEG.search(reason[:m.start()]):
                    said = _ptype(m.group(2))
                    ok = said == actual
                    faith.append(ok)
                    ck.append({"kind": "faithful", "ok": ok, "note": f"reason follows the {said}, call acts on the {actual}"})
                m = _RE_DISCARD.search(reason)
                if m and actual:
                    said = _ptype(m.group(2))
                    if said == actual:
                        faith.append(False)
                        ck.append({"kind": "faithful", "ok": False, "note": f"reason discards the {said} but the call uses it"})

        # ---------------------------------------------------- decisions: first use of a split product
        if tool == "combine":
            used = [u for u in (args.get("sample_ids") or []) if isinstance(u, str)]
        elif tool in USE_TOOLS:
            used = [sid]
        else:
            used = []
        if tool == "combine":
            # merging both products of one split is not a choice
            for p in {id(pairs[u]): pairs[u] for u in used if u in pairs}.values():
                if p["a"] in used and p["b"] in used:
                    p["done"] = True
        for u in used:
            p = pairs.get(u)
            if not p or p["done"]:
                continue
            p["done"] = True
            other = p["b"] if u == p["a"] else p["a"]
            if p["relevant"]:
                ok = u == p["best"]
                ma, mb = _ratio(p["meas"].get(u, [])), _ratio(p["meas"].get(other, []))
                if ma is not None and mb is not None:
                    sa, sb = ma / p["share"][u], mb / p["share"][other]
                    consistent = sa >= sb
                    cat = ("correct" if ok else "misled_by_evidence") if consistent else ("against_evidence_correct" if ok else "against_evidence")
                else:
                    consistent = None
                    cat = "correct_unmeasured" if ok else "wrong_unmeasured"
                decisions.append({"step": idx, "ok": ok, "consistent": consistent, "category": cat, "kind": p["kind"]})
                note = {"correct": "kept the right product, as its measurements showed",
                        "correct_unmeasured": "kept the right product without measuring both",
                        "misled_by_evidence": "followed its measurements, but they pointed the wrong way",
                        "against_evidence": "went against its own measurements and lost the target",
                        "against_evidence_correct": "went against its own measurements (and was right)",
                        "wrong_unmeasured": "kept the wrong product without measuring both"}[cat]
                ck.append({"kind": "decision", "ok": ok, "note": f"{note} ({p['kind']}, step #{p['step']})"})

        # ---------------------------------------------------- execute
        out = lab.call(tool, **args)
        rec = st["result"]
        if tool != "rules" and ("error" in out) != ("error" in rec):
            mismatches += 1

        if "error" in out:
            continue
        if tool == "measure_activity":
            net = out["gross_cpm"] - bg
            p = pairs.get(sid)
            if p and not p["done"]:
                p["meas"].setdefault(sid, []).append(net)
            exp = (st.get("prediction") or {}).get("expect_net_cpm")
            if exp is not None:
                try:
                    exp = float(exp)
                    true_net = max(1.0, net)
                    ok = exp > 0 and abs(math.log(exp / true_net)) <= CPM_TOLERANCE
                    preds.append(ok)
                    ck.append({"kind": "prediction", "ok": ok, "note": f"expected {exp:g} net cpm, measured {net:.0f}"})
                except (TypeError, ValueError):
                    pass
        elif tool in ("precipitate", "crystallize"):
            if tool == "precipitate":
                a, b, kind = out["precipitate"], out["filtrate"], "precipitate"
                ha, hb = _hidden(lab, a), _hidden(lab, b)
                best = a if ha["target"] >= hb["target"] else b
                share = {a: 0.5, b: 0.5}             # compare raw activity
                types = {"precipitate": a, "filtrate": b}
                pkey, pval = "expect_target_in", {"precipitate": a, "filtrate": b}
            else:
                a, b, kind = out["crystals"], out["mother_liquor"], "crystallize"
                ha, hb = _hidden(lab, a), _hidden(lab, b)
                best = a if ha["purity"] >= hb["purity"] else b
                f = float(args.get("fraction", 0.5))
                share = {a: f, b: 1 - f}             # compare activity per unit of mass share
                types = {"crystals": a, "mother liquor": b}
                pkey, pval = "expect_enriched_in", {"crystals": a, "mother_liquor": b}
            relevant = (ha["target"] + hb["target"]) >= MIN_TARGET_SHARE * target0
            p = {"a": a, "b": b, "kind": kind, "best": best, "share": share, "meas": {}, "done": False,
                 "relevant": relevant, "step": idx, "types": set(types),
                 "best_type": next(t for t, v in types.items() if v == best)}
            pairs[a] = pairs[b] = p
            last_pair = p
            exp = (st.get("prediction") or {}).get(pkey)
            if exp in pval:
                ok = pval[exp] == best
                preds.append(ok)
                ck.append({"kind": "prediction", "ok": ok, "note": f"expected {exp.replace('_', ' ')}; target "
                           f"{'mostly' if kind == 'precipitate' else 'enriched'} in the {p['best_type']}"})

    def rate(xs):
        return sum(xs) / len(xs) if xs else None

    eligible = sum(1 for s in run["steps"] if s["tool"] in ("precipitate", "crystallize", "measure_activity")
                   and "error" not in s["result"])
    measured = [d for d in decisions if d["consistent"] is not None]
    cats = {}
    for d in decisions:
        cats[d["category"]] = cats.get(d["category"], 0) + 1
    m = {
        "decision_accuracy": rate([d["ok"] for d in decisions]),
        "evidence_consistency": rate([d["consistent"] for d in measured]),
        "claim_truth": rate(claims),
        "faithfulness": rate(faith),
        "prediction_accuracy": rate(preds),
        "prediction_rate": len(preds) / eligible if eligible else 0.0,
        "n_decisions": len(decisions), "n_measured_decisions": len(measured), "n_claims": len(claims),
        "n_faithfulness": len(faith), "n_predictions": len(preds),
        "decision_categories": cats,
        "replay_mismatches": mismatches,
    }
    parts = [m[k] for k in ("decision_accuracy", "evidence_consistency", "claim_truth", "faithfulness") if m[k] is not None]
    m["grs"] = sum(parts) / len(parts) if parts else None
    return {"metrics": m, "checks": checks}
