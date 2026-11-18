"""Run models on seeds, then rebuild the comparison site.

    python -m llm_bench.run --seeds 101 102 103                     # every enabled model in models.json
    python -m llm_bench.run --models claude-opus-5-5 gpt-5 --seeds 101
    python -m llm_bench.run --families Claude Gemini --seeds 101 102 --skip-existing
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from .agent import run_episode
from .config import RUNS_DIR, find_model, load_env, load_models

EST_REQUESTS = 60     # typical model requests per episode (one request can carry several tool calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="model ids or labels (default: all in models.json)")
    ap.add_argument("--families", nargs="*", help="only these families, e.g. Claude GPT Gemini")
    ap.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    ap.add_argument("--max-requests", type=int, default=200)
    ap.add_argument("--parallel", type=int, default=1, help="episodes run concurrently")
    ap.add_argument("--skip-existing", action="store_true", help="skip model/seed pairs that already have a finished run")
    ap.add_argument("--no-site", action="store_true")
    a = ap.parse_args()

    load_env()
    models = load_models()
    specs = [find_model(m, models) for m in a.models] if a.models else [m for m in models if m.enabled]
    if a.families:
        specs = [s for s in specs if s.family in a.families]
    jobs = []
    for s in specs:
        for seed in a.seeds:
            if a.skip_existing and (RUNS_DIR / s.slug / f"seed{seed}.json").exists():
                print(f"skip {s.id} seed {seed} (exists)")
                continue
            jobs.append((s, seed))

    per_model = {}
    for s, _ in jobs:
        per_model[s.id] = per_model.get(s.id, 0) + 1
    for s in specs:
        n = per_model.get(s.id, 0)
        if n and s.rpd and s.rpd < n * EST_REQUESTS:
            print(f"WARNING {s.label}: ~{n * EST_REQUESTS} requests needed for {n} episode(s) "
                  f"(~{EST_REQUESTS} each) but its daily limit is {s.rpd}. It will stop with a quota error "
                  f"part-way; use a paid tier or fewer seeds.", flush=True)
        if n and s.rpm:
            print(f"{s.label}: limited to {s.rpm:g} requests/min, so one episode takes about "
                  f"{EST_REQUESTS / s.rpm:.0f} min.", flush=True)

    fatal: dict = {}     # model id -> account-level error; its remaining episodes are skipped

    def one(job):
        s, seed = job
        if s.id in fatal:
            print(f"-- skip {s.label} seed {seed}: {fatal[s.id]}", flush=True)
            return None
        print(f"== {s.label} seed {seed}", flush=True)
        rec = run_episode(s, seed, max_requests=a.max_requests, verbose=a.parallel == 1)
        r = rec["result"] or {}
        if rec["status"].startswith("error"):
            print(f"!! {s.label} seed {seed} FAILED: {rec['status']}", flush=True)
            st = rec["status"]
            if any(k in st for k in ("HTTP 401", "HTTP 402", "HTTP 403", "quota exhausted", "is not set")):
                fatal[s.id] = "account / key problem, see the error above (fix it and re-run with --skip-existing)"
            return rec
        print(f"== {s.label} seed {seed}: {rec['status']} reward={r.get('reward', 0):.3f} "
              f"actions={rec['actions']} tokens={rec['usage']}", flush=True)
        return rec

    with ThreadPoolExecutor(max(1, a.parallel)) as ex:
        list(ex.map(one, jobs))
    if not a.no_site:
        from .site import build
        print("site:", build())


if __name__ == "__main__":
    main()
