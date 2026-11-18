"""Episode state and lab operations.

Masses are stored *undecayed* (as of t=0); the live mass of an element is
m0 * exp(-lambda * t). All operations act on undecayed masses, so mass balance
is exact: ore = sum(samples) + sum(recorded operation losses), per element.
Decay is the only other change and it is applied analytically on read.
"""
from __future__ import annotations

import math
import random

from .config import Config, DEFAULT
from .scoring import score_submission
from .world import TARGET, World, generate

FREE_TOOLS = {"rules", "list_samples"}


class LabError(Exception):
    pass


class Lab:
    def __init__(self, seed: int, cfg: Config = DEFAULT):
        self.cfg = cfg
        self.seed = seed
        self.world: World = generate(seed, cfg)
        self.t = 0.0
        self.actions = 0
        self.reagent_left = {r: cfg.reagent_units for r in self.world.reagents}
        self.samples: dict = {}          # id -> {"m": {key: g0}, "state": str, "origin": str}
        self.lost = {k: 0.0 for k in self.world.ore}
        self._next = 0
        self.done = False
        self.result: dict | None = None
        self.log: list = []
        self._new("solid", dict(self.world.ore), "ore", sid="ore")

    # ------------------------------------------------------------ helpers
    def _new(self, state: str, m: dict, origin: str, sid: str | None = None) -> str:
        if sid is None:
            self._next += 1
            sid = f"S{self._next}"
        self.samples[sid] = {"m": m, "state": state, "origin": origin}
        return sid

    def _rng(self, tag: str) -> random.Random:
        return random.Random(f"noise|{self.seed}|{self.actions}|{tag}")

    def _get(self, sid: str) -> dict:
        if sid not in self.samples:
            raise LabError(f"no sample {sid!r}; call list_samples")
        return self.samples[sid]

    def _live(self, m: dict) -> dict:
        els = self.world.elements
        return {k: g * math.exp(-els[k].decay_const() * self.t) for k, g in m.items()}

    def _apply_loss(self, m: dict) -> dict:
        f = self.cfg.op_loss
        out = {}
        for k, g in m.items():
            self.lost[k] += g * f
            out[k] = g * (1.0 - f)
        return out

    def _tick(self, op: str):
        self.t += self.cfg.hours_for(op)

    # ------------------------------------------------------------ dispatch
    def call(self, tool: str, reason: str | None = None, **args) -> dict:
        """Single entry point used by LLM agents, human players and scripted policies.

        `reason` is the one-sentence key reason the decision-maker gave for this call. It is
        recorded in the log and never passed to the operation.
        """
        entry = {"i": len(self.log), "tool": tool, "args": args, "reason": reason}
        try:
            if self.done and tool not in FREE_TOOLS:
                raise LabError("episode is over")
            fn = getattr(self, f"op_{tool}", None)
            if fn is None:
                raise LabError(f"unknown tool {tool!r}")
            if tool not in FREE_TOOLS:
                if self.actions >= self.cfg.max_actions:
                    raise LabError("action budget exhausted")
            out = fn(**args)
            if tool not in FREE_TOOLS:
                self.actions += 1
                out["actions_left"] = self.cfg.max_actions - self.actions
                out["time_h"] = round(self.t, 2)
                if self.actions >= self.cfg.max_actions and not self.done:
                    self._finish(None)
                    out["episode_over"] = "action budget exhausted without submission; score 0"
        except LabError as e:
            out = {"error": str(e)}
        except (TypeError, ValueError, OverflowError) as e:
            out = {"error": f"bad arguments: {e}"}
        entry["result"] = out
        self.log.append(entry)
        return out

    # ------------------------------------------------------------ free tools
    def op_rules(self) -> dict:
        from .rules import rules_payload
        return rules_payload(self)

    def op_list_samples(self) -> dict:
        return {"samples": [{"id": s, "state": v["state"], "origin": v["origin"]}
                            for s, v in self.samples.items()],
                "reagents_left": dict(self.reagent_left),
                "actions_left": self.cfg.max_actions - self.actions,
                "time_h": round(self.t, 2)}

    # ------------------------------------------------------------ wet chemistry
    def op_dissolve(self, sample_id: str) -> dict:
        s = self._get(sample_id)
        if s["state"] != "solid":
            raise LabError(f"{sample_id} is already in solution")
        s["m"] = self._apply_loss(s["m"])
        s["state"] = "solution"
        self._tick("dissolve")
        return {"sample_id": sample_id, "state": "solution"}

    def op_precipitate(self, sample_id: str, reagent: str) -> dict:
        s = self._get(sample_id)
        if s["state"] != "solution":
            raise LabError(f"{sample_id} must be dissolved first")
        if reagent not in self.reagent_left:
            raise LabError(f"unknown reagent {reagent!r}")
        if self.reagent_left[reagent] <= 0:
            raise LabError(f"no {reagent} left")
        j = self.world.reagents.index(reagent)
        m = self._apply_loss(s["m"])
        ppt, fil = {}, {}
        for k, g in m.items():
            f = self.world.elements[k].precip[j]
            ppt[k], fil[k] = g * f, g * (1.0 - f)
        self.reagent_left[reagent] -= 1
        del self.samples[sample_id]
        a = self._new("solid", ppt, f"precipitate of {sample_id} with {reagent}")
        b = self._new("solution", fil, f"filtrate of {sample_id} with {reagent}")
        self._tick("precipitate")
        return {"precipitate": a, "filtrate": b}

    def op_crystallize(self, sample_id: str, fraction: float) -> dict:
        s = self._get(sample_id)
        if s["state"] != "solution":
            raise LabError(f"{sample_id} must be dissolved first")
        fraction = float(fraction)
        if not (0.05 <= fraction <= 0.95):
            raise LabError("fraction must be between 0.05 and 0.95")
        m = self._apply_loss(s["m"])
        total = sum(m.values())
        beta = {k: self.world.elements[k].beta for k in m}

        def crystal_mass(lam: float) -> float:
            return sum(g * lam * beta[k] / (lam * beta[k] + 1.0) for k, g in m.items())

        lo, hi = 1e-12, 1e12
        for _ in range(200):           # deterministic bisection in log space
            mid = math.sqrt(lo * hi)
            if crystal_mass(mid) < fraction * total:
                lo = mid
            else:
                hi = mid
        lam = math.sqrt(lo * hi)
        cry, mot = {}, {}
        for k, g in m.items():
            p = lam * beta[k] / (lam * beta[k] + 1.0)
            cry[k], mot[k] = g * p, g * (1.0 - p)
        del self.samples[sample_id]
        a = self._new("solid", cry, f"crystals of {sample_id} ({fraction:.2f})")
        b = self._new("solution", mot, f"mother liquor of {sample_id}")
        self._tick("crystallize")
        return {"crystals": a, "mother_liquor": b}

    def op_combine(self, sample_ids: list) -> dict:
        ids = list(dict.fromkeys(sample_ids))
        if len(ids) < 2:
            raise LabError("combine needs at least two distinct samples")
        ss = [self._get(i) for i in ids]
        states = {s["state"] for s in ss}
        if len(states) != 1:
            raise LabError("cannot combine solids with solutions; dissolve first")
        m = {k: 0.0 for k in self.world.ore}
        for s in ss:
            for k, g in s["m"].items():
                m[k] += g
        for i in ids:
            del self.samples[i]
        sid = self._new(states.pop(), m, "combined " + "+".join(ids))
        self._tick("combine")
        return {"sample_id": sid}

    def op_wait(self, hours: float) -> dict:
        hours = float(hours)
        if not (0 < hours <= self.cfg.max_wait_h):
            raise LabError(f"hours must be in (0, {self.cfg.max_wait_h}]")
        self.t += hours
        return {"waited_h": hours}

    # ------------------------------------------------------------ instruments
    def op_measure_activity(self, sample_id: str, minutes: float = 1.0) -> dict:
        s = self._get(sample_id)
        minutes = float(minutes)
        if not (0.1 <= minutes <= self.cfg.max_count_minutes):
            raise LabError(f"minutes must be in [0.1, {self.cfg.max_count_minutes}]")
        live = self._live(s["m"])
        rate = sum(g * self.world.elements[k].activity for k, g in live.items())
        rng = self._rng("act")
        rate *= max(0.0, 1.0 + rng.gauss(0, self.cfg.activity_rel_noise))
        lam = (rate + self.cfg.background_cpm) * minutes
        counts = max(0, round(lam + rng.gauss(0, math.sqrt(lam))))
        self.t += self.cfg.hours_for("measure_activity_base") + minutes / 60.0
        return {"sample_id": sample_id, "counts": counts, "minutes": minutes,
                "gross_cpm": round(counts / minutes, 1)}

    def op_weigh(self, sample_id: str) -> dict:
        s = self._get(sample_id)
        mass = sum(self._live(s["m"]).values())
        rng = self._rng("w")
        noisy = mass * (1 + rng.gauss(0, self.cfg.weigh_rel_noise)) + rng.gauss(0, self.cfg.weigh_abs_noise_g)
        self._tick("weigh")
        return {"sample_id": sample_id, "mass_g": round(max(0.0, noisy), 4)}

    def op_spectrum(self, sample_id: str) -> dict:
        s = self._get(sample_id)
        live = self._live(s["m"])
        total = sum(live.values())
        rng = self._rng("spec")
        lines = []
        if total > 0:
            for k, g in live.items():
                e = self.world.elements[k]
                w = g / total
                if w >= e.line_threshold:
                    for x in e.lines:
                        inten = w * e.sensitivity * max(0.0, 1 + rng.gauss(0, self.cfg.spectrum_rel_noise))
                        lines.append({"nm": x, "intensity": round(inten, 4)})
        lines.sort(key=lambda d: d["nm"])
        self._tick("spectrum")
        return {"sample_id": sample_id, "lines": lines}

    def op_atomic_weight(self, sample_id: str) -> dict:
        s = self._get(sample_id)
        live = self._live(s["m"])
        total = sum(live.values())
        if total < self.cfg.atomic_weight_min_mass_g:
            raise LabError(f"need at least {self.cfg.atomic_weight_min_mass_g} g")
        aw = sum(g * self.world.elements[k].atomic_weight for k, g in live.items()) / total
        aw *= 1 + self._rng("aw").gauss(0, self.cfg.atomic_weight_rel_noise)
        self._tick("atomic_weight")
        return {"sample_id": sample_id, "apparent_atomic_weight": round(aw, 2)}

    # ------------------------------------------------------------ submission
    def op_submit(self, sample_id: str, atomic_weight: float, carrier: str) -> dict:
        self._get(sample_id)
        if not math.isfinite(float(atomic_weight)):
            raise LabError("atomic_weight must be a finite number")
        self._finish({"sample_id": sample_id, "atomic_weight": float(atomic_weight), "carrier": str(carrier)})
        return {"submitted": sample_id, "message": "episode over; score computed by the server"}

    def _finish(self, sub: dict | None):
        self.done = True
        if sub is None:
            self.result = {"reward": 0.0, "reason": "no submission"}
            return
        live = self._live(self.samples[sub["sample_id"]]["m"])
        target0 = self.world.ore[TARGET] * math.exp(-self.world.elements[TARGET].decay_const() * self.t)
        self.result = score_submission(self.cfg, self.world, live, target0, sub)

    def final_reward(self) -> float:
        return 0.0 if not self.result else self.result["reward"]

    def trajectory(self) -> dict:
        return {"seed": self.seed, "config": self.cfg.to_dict(), "actions": self.actions,
                "time_h": self.t, "log": self.log, "result": self.result, "truth": self.world.truth()}
