"""All difficulty knobs and scoring constants live here.

Results reported in the README name the config (by `name`) and the seeds used.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class Config:
    name: str = "default-v1"

    # --- world ---------------------------------------------------------
    n_common: int = 6               # known (non-trace) elements in the ore
    n_reagents: int = 5             # precipitating reagents available
    ore_mass_g: float = 1000.0
    target_mass_g: float = 0.30     # hidden long-lived trace element
    interferer_mass_g: float = 0.004  # hidden short-lived trace element
    carrier_fraction: tuple = (0.005, 0.010)   # ore mass fraction of the target's carrier
    common_fraction: tuple = (0.05, 0.30)    # other commons (normalised later)
    n_minor_common: int = 2                  # other commons as rare as the carrier (assay can't single it out)
    target_alpha: tuple = (5.0, 8.0)         # crystallisation enrichment of target vs carrier
    precip_high: tuple = (0.92, 0.99)        # "precipitates" fraction range
    precip_low: tuple = (0.01, 0.06)         # "stays in solution" fraction range
    follow_jitter: float = 0.03              # how closely trace elements follow their carrier

    # --- radioactivity (counts per minute per gram, detector included) --
    target_activity: float = 4.0e4
    target_half_life_h: float = 1.4e7
    interferer_initial_cpm: tuple = (2.5e4, 5.0e4)   # whole-ore activity at t=0
    interferer_half_life_h: tuple = (8.0, 20.0)
    parent_activity: float = 20.0           # known weakly radioactive common element
    background_cpm: float = 30.0
    activity_rel_noise: float = 0.03        # multiplicative noise (geometry etc.)

    # --- other instruments ---------------------------------------------
    weigh_abs_noise_g: float = 0.0005
    weigh_rel_noise: float = 0.002
    known_line_threshold: float = 1e-3       # mass fraction for known lines to show
    unknown_line_threshold: tuple = (0.01, 0.04)  # per-seed, for hidden elements
    spectrum_rel_noise: float = 0.05
    atomic_weight_rel_noise: float = 0.003
    atomic_weight_min_mass_g: float = 0.005

    # --- operations ------------------------------------------------------
    op_loss: float = 0.01                    # fraction of every element lost per wet operation
    max_actions: int = 100                   # all lab tool calls count, incl. measurements
    reagent_units: int = 3                   # uses per reagent
    max_wait_h: float = 240.0
    max_count_minutes: float = 60.0
    hours: tuple = (                         # simulated time cost of each operation
        ("dissolve", 0.5), ("precipitate", 1.0), ("crystallize", 4.0),
        ("combine", 0.25), ("measure_activity_base", 0.1), ("weigh", 0.25),
        ("spectrum", 0.5), ("atomic_weight", 1.0),
    )

    # --- scoring -----------------------------------------------------------
    purity_goal: float = 0.90
    recovery_goal: float = 0.30
    report_gate_purity: float = 0.50
    weight_tol_full: float = 0.02            # relative error for full credit
    weight_tol_zero: float = 0.10            # relative error for zero credit
    w_separation: float = 0.70
    w_report: float = 0.30

    def hours_for(self, op: str) -> float:
        return dict(self.hours)[op]

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Config()


def get_config(name: str = "default-v1", **overrides) -> Config:
    if name != DEFAULT.name:
        raise KeyError(f"unknown config {name!r}")
    return replace(DEFAULT, **overrides) if overrides else DEFAULT
