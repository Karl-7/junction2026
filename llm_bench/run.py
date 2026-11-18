"""Run models on seeds, then rebuild the comparison site.

    python -m llm_bench.run --seeds 101 102 103                     # every model in models.json
    python -m llm_bench.run --models claude-opus-5-5 gpt-5 --seeds 101
    python -m llm_bench.run --families Claude Gemini --seeds 101 102 --skip-existing
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from .agent import run_episode
from .config import RUNS_DIR, find_model, load_env, load_models


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
    specs = [find_model(m, models) for m in a.models] if a.models else models
    if a.families:
        specs = [s for s in specs if s.family in a.families]
    jobs = []
    for s in specs:
        for seed in a.seeds:
            if a.skip_existing and (RUNS_DIR / s.slug / f"seed{seed}.json").exists():
                print(f"skip {s.id} seed {seed} (exists)")
                continue
            jobs.append((s, seed))

    def one(job):
        s, seed = job
        print(f"== {s.label} seed {seed}", flush=True)
        rec = run_episode(s, seed, max_requests=a.max_requests, verbose=a.parallel == 1)
        r = rec["result"] or {}
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
