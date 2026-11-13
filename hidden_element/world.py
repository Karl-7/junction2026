"""Seeded world generation. Everything in `World` except `public()` is hidden from the agent."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .config import Config

# Invented symbols. No real element symbol starts with J or Q.
SYMBOL_POOL = ["Qa", "Qe", "Qi", "Qo", "Qu", "Qy", "Ja", "Je", "Jo", "Ju"]
TARGET = "target"
INTERFERER = "interferer"


@dataclass
class Element:
    key: str                    # symbol for known elements, TARGET/INTERFERER for hidden ones
    known: bool
    atomic_weight: float
    activity: float             # cpm per gram at t=0
    half_life_h: float          # math.inf for stable
    precip: list                # fraction precipitated per reagent index
    beta: float                 # crystallisation affinity (relative)
    lines: list                 # characteristic wavelengths, nm
    sensitivity: float          # line intensity per unit mass fraction
    line_threshold: float       # mass fraction needed for lines to appear

    def decay_const(self) -> float:
        return 0.0 if math.isinf(self.half_life_h) else math.log(2) / self.half_life_h


@dataclass
class World:
    seed: int
    elements: dict              # key -> Element
    reagents: list              # public reagent names
    ore: dict                   # key -> grams at t=0
    carrier: str                # known symbol the target follows
    interferer_carrier: str
    parent: str                 # known, weakly radioactive common element
    meta: dict = field(default_factory=dict)

    def public(self) -> dict:
        """What the agent is told: known elements' physics and the ore assay, nothing seed-chemical."""
        total = sum(self.ore.values())
        known = []
        for k, e in self.elements.items():
            if not e.known:
                continue
            known.append({
                "symbol": k,
                "atomic_weight": round(e.atomic_weight, 2),
                "spectral_lines_nm": [round(x, 1) for x in e.lines],
                "line_sensitivity": round(e.sensitivity, 3),
                "ore_mass_fraction": round(self.ore[k] / total, 4),
                "specific_activity_cpm_per_g": e.activity,
                "half_life": "stable" if math.isinf(e.half_life_h) else "very long",
            })
        known.sort(key=lambda d: d["symbol"])
        return {"known_elements": known, "reagents": list(self.reagents)}

    def truth(self) -> dict:
        return {
            "seed": self.seed,
            "carrier": self.carrier,
            "interferer_carrier": self.interferer_carrier,
            "parent": self.parent,
            "ore_g": self.ore,
            "elements": {k: {kk: (vv if not (isinstance(vv, float) and math.isinf(vv)) else "inf")
                             for kk, vv in e.__dict__.items()} for k, e in self.elements.items()},
            **self.meta,
        }


def _pick_lines(rng: random.Random, taken: list, n: int = 2) -> list:
    out = []
    while len(out) < n:
        x = round(rng.uniform(380.0, 700.0), 1)
        if all(abs(x - t) > 2.0 for t in taken + out):
            out.append(x)
    taken.extend(out)
    return sorted(out)


def _pick_weight(rng: random.Random, lo: float, hi: float, taken: list, gap: float) -> float:
    while True:
        w = round(rng.uniform(lo, hi), 2)
        if all(abs(w - t) >= gap for t in taken):
            taken.append(w)
            return w


def generate(seed: int, cfg: Config) -> World:
    rng = random.Random(f"world|{seed}")
    nc, nr = cfg.n_common, cfg.n_reagents

    # Binary precipitation patterns for common elements: every reagent must split the set,
    # and the carrier's pattern must be unique so the target group *can* be isolated.
    while True:
        pat = [[rng.random() < 0.5 for _ in range(nr)] for _ in range(nc)]
        cols_ok = all(0 < sum(pat[i][j] for i in range(nc)) < nc for j in range(nr))
        if not cols_ok:
            continue
        carrier_i = rng.randrange(nc)
        # carrier differs from every other common element in >= 2 reagents
        if any(i != carrier_i and sum(a != b for a, b in zip(pat[i], pat[carrier_i])) < 2
               for i in range(nc)):
            continue
        others = [i for i in range(nc) if i != carrier_i]
        rng.shuffle(others)
        icar_i, parent_i = others[0], others[1]
        break

    symbols = rng.sample(SYMBOL_POOL, nc)
    taken_lines: list = []
    taken_w: list = []

    def frac(high: bool) -> float:
        lo, hi = cfg.precip_high if high else cfg.precip_low
        return rng.uniform(lo, hi)

    elements: dict = {}
    for i, sym in enumerate(symbols):
        elements[sym] = Element(
            key=sym, known=True,
            atomic_weight=_pick_weight(rng, 20.0, 210.0, taken_w, 8.0),
            activity=cfg.parent_activity if i == parent_i else 0.0,
            half_life_h=1.0e10 if i == parent_i else math.inf,
            precip=[frac(b) for b in pat[i]],
            beta=math.exp(rng.uniform(math.log(0.5), math.log(2.0))),
            lines=_pick_lines(rng, taken_lines),
            sensitivity=round(rng.uniform(0.5, 2.0), 3),
            line_threshold=cfg.known_line_threshold,
        )
    carrier, icar, parent = symbols[carrier_i], symbols[icar_i], symbols[parent_i]

    def follow(base: list) -> list:
        return [min(0.995, max(0.005, f + rng.uniform(-cfg.follow_jitter, cfg.follow_jitter))) for f in base]

    car = elements[carrier]
    t_weight = _pick_weight(rng, max(car.atomic_weight + 30.0, 150.0), 260.0, taken_w, 8.0) \
        if car.atomic_weight + 30.0 < 260.0 else _pick_weight(rng, 230.0, 260.0, taken_w, 3.0)
    elements[TARGET] = Element(
        key=TARGET, known=False, atomic_weight=t_weight,
        activity=cfg.target_activity, half_life_h=cfg.target_half_life_h,
        precip=follow(car.precip), beta=car.beta * rng.uniform(*cfg.target_alpha),
        lines=_pick_lines(rng, taken_lines), sensitivity=round(rng.uniform(0.5, 2.0), 3),
        line_threshold=round(rng.uniform(*cfg.unknown_line_threshold), 4),
    )
    ic = elements[icar]
    i_half = rng.uniform(*cfg.interferer_half_life_h)
    i_cpm = rng.uniform(*cfg.interferer_initial_cpm)
    elements[INTERFERER] = Element(
        key=INTERFERER, known=False,
        atomic_weight=_pick_weight(rng, 60.0, 240.0, taken_w, 3.0),
        activity=i_cpm / cfg.interferer_mass_g, half_life_h=i_half,
        precip=follow(ic.precip), beta=ic.beta * rng.uniform(0.8, 1.25),
        lines=_pick_lines(rng, taken_lines), sensitivity=round(rng.uniform(0.5, 2.0), 3),
        line_threshold=round(rng.uniform(*cfg.unknown_line_threshold), 4),
    )

    # Ore assay.
    trace = cfg.target_mass_g + cfg.interferer_mass_g
    common_total = cfg.ore_mass_g - trace
    car_g = rng.uniform(*cfg.carrier_fraction) * cfg.ore_mass_g
    raw = {s: rng.uniform(*cfg.common_fraction) for s in symbols if s != carrier}
    scale = (common_total - car_g) / sum(raw.values())
    ore = {s: raw[s] * scale for s in raw}
    ore[carrier] = car_g
    ore[TARGET] = cfg.target_mass_g
    ore[INTERFERER] = cfg.interferer_mass_g

    reagents = [f"reagent_{chr(ord('A') + j)}" for j in range(nr)]
    return World(seed=seed, elements=elements, reagents=reagents, ore=ore,
                 carrier=carrier, interferer_carrier=icar, parent=parent,
                 meta={"config": cfg.name})
