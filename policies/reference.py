"""Human-expert reference ("Curie protocol"). Uses only the public tool interface, never hidden state.

This is the procedure a trained radiochemist would follow, modelled on how Marie and Pierre Curie
isolated their new element in 1898-1902: follow the radioactivity through every separation, then concentrate the new
element by repeated fractional crystallisation while recycling the mother liquors. Every call
carries the one-sentence reason a human expert would give, so LLM runs can be compared with it
decision by decision.

Fixed procedure:
  1. Read the rules. Wait out the short-lived interferer (its half-life range is public).
  2. Dissolve the ore. For every reagent: precipitate, measure both products (background
     subtracted, repeated if the two are close), follow the stronger one.
  3. Fractional crystallisation with mother-liquor recycling. Each round: crystallise the
     head sample, follow the product with the higher *specific* activity (net cpm divided by
     its known share of the mass); crystallise that round's mother liquor once more and merge
     its crystals back into the head; the second mother liquor is discarded. Stop when the
     spectrum shows an unassigned line and the known lines account for <= 15% of the mass
     (purity estimated from public line sensitivities), or the budget runs low.
  4. Report: carrier = dominant known element right after step 2; atomic weight = the measured apparent
     weight corrected for the known-element fractions read from the spectrum.
"""
from __future__ import annotations

CRYSTAL_FRACTION = 0.40
RECYCLE_FRACTION = 0.40
STOP_KNOWN_FRACTION = 0.15
COUNT_MIN = 2.0


def _net(env, sid, bg, why="Measure the activity so the target's signal can be followed."):
    r = env.call("measure_activity", sample_id=sid, minutes=COUNT_MIN, reason=why)
    return r["gross_cpm"] - bg


def _follow(env, a, b, bg):
    """Return (stronger, weaker); re-measure once if the difference is within noise."""
    na = _net(env, a, bg, "Count the precipitate, minus background, to see whether the target's activity went with it.")
    nb = _net(env, b, bg, "Count the filtrate too, because only the comparison shows where the activity went.")
    if abs(na - nb) < 0.15 * max(na, nb, 1.0):
        why = "The two background-subtracted counts are within noise of each other, so count again before choosing."
        na = (na + _net(env, a, bg, why)) / 2
        nb = (nb + _net(env, b, bg, why)) / 2
    return (a, b) if na >= nb else (b, a)


def _follow_specific(env, cry, liq, frac, bg):
    """Crystals hold `frac` of the mass by construction, so specific activity needs no weighing."""
    sc = _net(env, cry, bg, "Count the crystals; activity per gram tells whether crystallisation enriched the target.") / frac
    sl = _net(env, liq, bg, "Count the mother liquor to compare specific activity with the crystals.") / (1 - frac)
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
    info = env.call("rules", reason="Read the rules and public data before touching the ore.")
    known = info["known_elements"]
    rules = info["rules"]
    bg = float(rules.split("background ")[1].split(" cpm")[0])
    env.call("wait", hours=240, reason="Let the short-lived interferer decay away so that "
             "remaining activity belongs to the long-lived target.")

    env.call("dissolve", sample_id="ore", reason="The ore must be in solution before any chemical separation.")
    cur = "ore"
    for reagent in info["reagents"]:
        r = env.call("precipitate", sample_id=cur, reagent=reagent,
                     reason=f"Split the active fraction with {reagent} to learn which side the target goes to.")
        keep, _ = _follow(env, r["precipitate"], r["filtrate"], bg)
        cur = keep
        if keep == r["precipitate"]:
            env.call("dissolve", sample_id=cur, reason="The activity followed the precipitate, "
                     "so dissolve it to continue separating.")

    # The carrier is the known element that dominates the target's group after separation.
    fr0 = _known_fractions(env.call("spectrum", sample_id=cur, reason="See which known element "
                                    "travelled with the activity through every reagent: that is the carrier."), known)
    carrier = max(fr0, key=fr0.get) if fr0 else ""
    best_aw, best_fr = None, None
    for _ in range(30):
        if env.call("list_samples", reason="Check the remaining budget before another round.")["actions_left"] < 10:
            break
        r = env.call("crystallize", sample_id=cur, fraction=CRYSTAL_FRACTION,
                     reason="The target cannot be split from its carrier by reagents, "
                            "so enrich it by fractional crystallisation.")
        keep, _ = _follow_specific(env, r["crystals"], r["mother_liquor"], CRYSTAL_FRACTION, bg)
        if keep == r["crystals"]:
            env.call("dissolve", sample_id=keep, reason="Redissolve the enriched crystals for the next round.")
            # Recycle: recover target left in the mother liquor.
            r2 = env.call("crystallize", sample_id=r["mother_liquor"], fraction=RECYCLE_FRACTION,
                          reason="The mother liquor still holds a lot of target; "
                                 "recrystallise it so recovery does not collapse.")
            env.call("dissolve", sample_id=r2["crystals"], reason="Dissolve the recovered crystals "
                     "so they can be merged back into the main fraction.")
            keep = env.call("combine", sample_ids=[keep, r2["crystals"]],
                            reason="Merge the recovered target back into the main fraction to keep recovery high.")["sample_id"]
        cur = keep
        spec = env.call("spectrum", sample_id=cur, reason="Estimate purity from how much of "
                        "the sample the known elements' lines still account for.")
        fr = _known_fractions(spec, known)
        if _unassigned(spec, known):
            aw = env.call("atomic_weight", sample_id=cur, reason="The new element's lines are visible; "
                          "measure the apparent atomic weight to correct for the remaining contaminants.")
            if "apparent_atomic_weight" in aw:
                best_aw, best_fr = aw["apparent_atomic_weight"], fr
            if sum(fr.values()) <= STOP_KNOWN_FRACTION or "error" in aw:
                break

    weights = {el["symbol"]: el["atomic_weight"] for el in known}
    if best_aw is None:
        best_aw = env.call("atomic_weight", sample_id=cur, reason="Out of budget for purification; "
                           "measure the atomic weight as it is.").get("apparent_atomic_weight", 0.0)
        best_fr = {}
    k = min(0.9, sum(best_fr.values()))
    est = (best_aw - sum(f * weights[s] for s, f in best_fr.items())) / (1 - k)
    return env.call("submit", sample_id=cur, atomic_weight=round(est, 2), carrier=carrier or "",
                    reason="Purity is near the goal with recovery kept high, so submit with the "
                           "contaminant-corrected atomic weight.")
