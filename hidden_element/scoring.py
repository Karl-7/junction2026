"""Server-side reward. Inputs are the hidden world and the submitted sample's true masses."""
from __future__ import annotations

from .config import Config
from .world import TARGET, World


def _clip01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def weight_credit(cfg: Config, reported: float, true: float) -> float:
    err = abs(reported - true) / true
    if err <= cfg.weight_tol_full:
        return 1.0
    return _clip01((cfg.weight_tol_zero - err) / (cfg.weight_tol_zero - cfg.weight_tol_full))


def score_submission(cfg: Config, world: World, live: dict, target_total: float, sub: dict) -> dict:
    total = sum(live.values())
    t = live.get(TARGET, 0.0)
    purity = t / total if total > 0 else 0.0
    recovery = t / target_total if target_total > 0 else 0.0
    p_score = _clip01(purity / cfg.purity_goal)
    r_score = _clip01(recovery / cfg.recovery_goal)
    separation = p_score * r_score

    w_credit = weight_credit(cfg, sub["atomic_weight"], world.elements[TARGET].atomic_weight)
    c_credit = 1.0 if sub["carrier"].strip() == world.carrier else 0.0
    gated = purity >= cfg.report_gate_purity
    report = 0.5 * w_credit + 0.5 * c_credit if gated else 0.0

    reward = cfg.w_separation * separation + cfg.w_report * report
    return {
        "reward": round(reward, 6),
        "purity": purity, "recovery": recovery,
        "purity_score": p_score, "recovery_score": r_score, "separation": separation,
        "report_gated_in": gated, "weight_credit": w_credit, "carrier_credit": c_credit,
        "report": report, "submitted": sub,
    }
