"""Baselines required by the judging protocol. All use only the public tool interface."""
from __future__ import annotations

import random


def do_nothing(env) -> dict:
    env.call("rules")
    return {}


def submit_ore(env) -> dict:
    """Submit the raw ore with a plausible-looking report."""
    info = env.call("rules")
    guess = info["known_elements"][0]["symbol"]
    return env.call("submit", sample_id="ore", atomic_weight=226.0, carrier=guess)


def one_step(env) -> dict:
    """Early stop: one separation, follow the stronger product, submit."""
    info = env.call("rules")
    env.call("dissolve", sample_id="ore")
    r = env.call("precipitate", sample_id="ore", reagent=info["reagents"][0])
    a = env.call("measure_activity", sample_id=r["precipitate"], minutes=2)["gross_cpm"]
    b = env.call("measure_activity", sample_id=r["filtrate"], minutes=2)["gross_cpm"]
    keep = r["precipitate"] if a >= b else r["filtrate"]
    aw = env.call("atomic_weight", sample_id=keep).get("apparent_atomic_weight", 200.0)
    return env.call("submit", sample_id=keep, atomic_weight=aw, carrier=info["known_elements"][0]["symbol"])


def random_ops(env, seed: int = 0, n_ops: int = 40) -> dict:
    """Random valid-looking operations, then submit a random sample."""
    rng = random.Random(f"random-baseline|{seed}")
    info = env.call("rules")
    tools = ["dissolve", "precipitate", "crystallize", "measure_activity", "weigh", "spectrum", "wait", "combine"]
    for _ in range(n_ops):
        samples = [s["id"] for s in env.call("list_samples")["samples"]]
        t = rng.choice(tools)
        sid = rng.choice(samples)
        args = {"sample_id": sid}
        if t == "precipitate":
            args["reagent"] = rng.choice(info["reagents"])
        elif t == "crystallize":
            args["fraction"] = round(rng.uniform(0.1, 0.9), 2)
        elif t == "wait":
            args = {"hours": round(rng.uniform(1, 100), 1)}
        elif t == "combine":
            args = {"sample_ids": rng.sample(samples, min(2, len(samples)))}
        env.call(t, **args)
    samples = [s["id"] for s in env.call("list_samples")["samples"]]
    return env.call("submit", sample_id=rng.choice(samples), atomic_weight=round(rng.uniform(150, 260), 1),
                    carrier=rng.choice(info["known_elements"])["symbol"])
