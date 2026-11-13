"""Public rules. This text is returned by the `rules` tool and also used in the task instruction."""
from __future__ import annotations

from .config import Config

RULES_TEXT = """\
HIDDEN ELEMENT: RULES

Goal
  The ore is more radioactive than its known constituents explain. It contains two trace,
  unidentified radioactive elements:
    * the TARGET: long-lived (no measurable decay during an episode);
    * an INTERFERER: short-lived (half-life between {ih_lo:g} and {ih_hi:g} hours).
  Isolate the target as pure as possible without losing too much of it, then submit one
  sample together with a report: the target's atomic weight, and the symbol of the known
  element it follows chemically (its "carrier").

What is public and what is hidden
  Public: the list of known elements (atomic weight, spectral lines, line sensitivity, ore
  mass fraction, specific activity), the reagent names, every rule and formula below.
  Hidden, different for every seed: how each element behaves with each reagent and under
  crystallisation, and every property of the two unknown elements.
  One known element is weakly radioactive (see specific_activity_cpm_per_g).

Samples
  The episode starts with one solid sample "ore" ({ore:g} g). Each sample is a mixture of
  elements; operations split or merge samples. Mass is conserved exactly except for:
    * a loss of {loss:.0%} of every element in each dissolve, precipitate and crystallize;
    * radioactive decay.

Tools (each costs one action unless marked free)
  rules()                              free. This text plus public data.
  list_samples()                       free. Sample ids, states, budgets, clock.
  dissolve(sample_id)                  solid -> solution (same id).
  precipitate(sample_id, reagent)      solution -> a solid "precipitate" and a "filtrate"
                                       solution. Each element i goes to the precipitate with a
                                       fixed hidden fraction f_i(reagent); the input sample is
                                       consumed. Uses one unit of that reagent.
  crystallize(sample_id, fraction)     solution -> solid "crystals" holding `fraction`
                                       (0.05-0.95) of the mass, and a "mother liquor" solution.
                                       Element i goes to the crystals with probability
                                       L*b_i / (L*b_i + 1), b_i a hidden per-element affinity and
                                       L solved so the crystal mass matches `fraction`. An element
                                       with higher b than its neighbours is enriched in crystals.
  combine(sample_ids)                  merge >= 2 samples of the same state.
  wait(hours)                          let time pass (max {maxwait:g} h). Activities decay.
  measure_activity(sample_id, minutes) counts over `minutes` (0.1-{maxmin:g}). Counts are Poisson
                                       around (true activity * (1 +/- {anoise:.0%} gaussian) +
                                       background {bg:g} cpm) * minutes. Subtract the background.
  weigh(sample_id)                     mass in grams, noise ~{wrel:.1%} + {wabs:g} g.
  spectrum(sample_id)                  emission lines (nm, intensity). An element's lines appear
                                       when its mass fraction exceeds its detection limit:
                                       {kth:g} for known elements, between {uth_lo:g} and {uth_hi:g}
                                       for unknown ones. Intensity = mass fraction * the element's
                                       line sensitivity (+/-{snoise:.0%}). Lines that match no known
                                       element come from an unknown element.
  atomic_weight(sample_id)             mass-fraction-weighted mean atomic weight of everything in
                                       the sample (+/-{awn:.1%}); needs >= {awmin:g} g. An impure
                                       sample reads between the target and its contaminants.
  submit(sample_id, atomic_weight, carrier)  ends the episode.

Time
  Operations take simulated time (dissolve 0.5 h, precipitate 1 h, crystallize 4 h,
  combine 0.25 h, weigh 0.25 h, spectrum 0.5 h, atomic_weight 1 h, measure 0.1 h + counting
  time, wait as requested). Real thinking time is free; the simulation only advances on tool calls.

Budgets
  {maxact} actions in total (free tools excluded). {units} units of each reagent.
  If actions run out before submit, the episode ends with reward 0. Invalid calls cost nothing.

Reward (computed by the server from the true contents of the submitted sample)
  purity   = target mass / sample mass
  recovery = target mass in sample / target mass in the ore
  separation = min(1, purity/{pg:g}) * min(1, recovery/{rg:g})
  report   = 0.5 * weight_credit + 0.5 * carrier_credit, counted only if purity >= {gate:g}
             weight_credit = 1 within {wf:.0%} relative error, falling linearly to 0 at {wz:.0%}
             carrier_credit = 1 if the carrier symbol is right
  reward   = {ws:g} * separation + {wr:g} * report          (0 if nothing is submitted)
"""


def rules_text(cfg: Config) -> str:
    return RULES_TEXT.format(
        ih_lo=cfg.interferer_half_life_h[0], ih_hi=cfg.interferer_half_life_h[1],
        ore=cfg.ore_mass_g, loss=cfg.op_loss, maxwait=cfg.max_wait_h, maxmin=cfg.max_count_minutes,
        anoise=cfg.activity_rel_noise, bg=cfg.background_cpm, wrel=cfg.weigh_rel_noise,
        wabs=cfg.weigh_abs_noise_g, kth=cfg.known_line_threshold,
        uth_lo=cfg.unknown_line_threshold[0], uth_hi=cfg.unknown_line_threshold[1],
        snoise=cfg.spectrum_rel_noise, awn=cfg.atomic_weight_rel_noise,
        awmin=cfg.atomic_weight_min_mass_g, maxact=cfg.max_actions, units=cfg.reagent_units,
        pg=cfg.purity_goal, rg=cfg.recovery_goal, gate=cfg.report_gate_purity,
        wf=cfg.weight_tol_full, wz=cfg.weight_tol_zero, ws=cfg.w_separation, wr=cfg.w_report,
    )


def rules_payload(lab) -> dict:
    return {"rules": rules_text(lab.cfg), **lab.world.public(),
            "budgets": {"max_actions": lab.cfg.max_actions, "reagent_units_each": lab.cfg.reagent_units}}
