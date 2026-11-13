"""Reference policy. Uses only the public tool interface (`env.call(tool, **args)`), never hidden state.

Fixed procedure:
  1. Read the rules. Wait out the short-lived interferer (its half-life range is public).
  2. Dissolve the ore. For every reagent: precipitate, measure both products (background
     subtracted, repeated if the two are close), follow the stronger one.
  3. Fractional crystallisation, following the product with the higher *specific* activity
     (net cpm divided by its known share of the mass, `fraction` vs 1-fraction), until the
     spectrum shows an unassigned line and the known lines account for <= 15% of the mass
     (purity estimated from public line sensitivities).
  4. Report: carrier = dominant known element right after step 2; atomic weight = the measured apparent
     weight corrected for the known-element fractions read from the spectrum.
"""
from __future__ import annotations

CRYSTAL_FRACTION = 0.50
STOP_KNOWN_FRACTION = 0.15
COUNT_MIN = 2.0


def _net(env, sid, bg):
    r = env.call("measure_activity", sample_id=sid, minutes=COUNT_MIN)
    return r["gross_cpm"] - bg


def _follow(env, a, b, bg):
    """Return (stronger, weaker); re-measure once if the difference is within noise."""
    na, nb = _net(env, a, bg), _net(env, b, bg)
    if abs(na - nb) < 0.15 * max(na, nb, 1.0):
        na = (na + _net(env, a, bg)) / 2
        nb = (nb + _net(env, b, bg)) / 2
    return (a, b) if na >= nb else (b, a)


def _follow_specific(env, cry, liq, frac, bg):
    """Crystals hold `frac` of the mass by construction, so specific activity needs no weighing."""
    sc = _net(env, cry, bg) / frac
    sl = _net(env, liq, bg) / (1 - frac)
    return (cry, liq) if sc >= sl else (liq, cry)


def _known_fractions(spec, known):
    """Mass fraction of each known element estimated from its line intensities."""
    out = {}
    for el in known:
        ints = [ln["intensity"] for ln in spec["lines"]
                if any(abs(ln["nm"] - x) < 0.05 for x in el["spectral_lines_nm"])]
        if ints:
            out[el["symbol"]] = (sum(ints) / len(ints)) / el["line_sensitivity"]
    return out


def _unassigned(spec, known):
    all_known = [x for el in known for x in el["spectral_lines_nm"]]
    return [ln for ln in spec["lines"] if all(abs(ln["nm"] - x) >= 0.05 for x in all_known)]


def run(env) -> dict:
    info = env.call("rules")
    known = info["known_elements"]
    rules = info["rules"]
    bg = float(rules.split("background ")[1].split(" cpm")[0])
    env.call("wait", hours=240)

    env.call("dissolve", sample_id="ore")
    cur = "ore"
    for reagent in info["reagents"]:
        r = env.call("precipitate", sample_id=cur, reagent=reagent)
        keep, _ = _follow(env, r["precipitate"], r["filtrate"], bg)
        cur = keep
        if keep == r["precipitate"]:
            env.call("dissolve", sample_id=cur)

    # The carrier is the known element that dominates the target's group after separation.
    fr0 = _known_fractions(env.call("spectrum", sample_id=cur), known)
    carrier = max(fr0, key=fr0.get) if fr0 else ""
    best_aw, best_fr = None, None
    for _ in range(20):
        r = env.call("crystallize", sample_id=cur, fraction=CRYSTAL_FRACTION)
        keep, _ = _follow_specific(env, r["crystals"], r["mother_liquor"], CRYSTAL_FRACTION, bg)
        cur = keep
        if keep == r["crystals"]:
            env.call("dissolve", sample_id=cur)
        spec = env.call("spectrum", sample_id=cur)
        fr = _known_fractions(spec, known)
        if _unassigned(spec, known):
            aw = env.call("atomic_weight", sample_id=cur)
            if "apparent_atomic_weight" in aw:
                best_aw, best_fr = aw["apparent_atomic_weight"], fr
            if sum(fr.values()) <= STOP_KNOWN_FRACTION or "error" in aw:
                break
        if env.call("list_samples")["actions_left"] < 8:
            break

    weights = {el["symbol"]: el["atomic_weight"] for el in known}
    if best_aw is None:
        best_aw = env.call("atomic_weight", sample_id=cur).get("apparent_atomic_weight", 0.0)
        best_fr = {}
    k = min(0.9, sum(best_fr.values()))
    est = (best_aw - sum(f * weights[s] for s, f in best_fr.items())) / (1 - k)
    return env.call("submit", sample_id=cur, atomic_weight=round(est, 2), carrier=carrier or "")
