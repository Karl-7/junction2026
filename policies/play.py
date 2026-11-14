"""One MCP tool call per invocation, for driving the lab by hand (or by a CLI agent).

    python -m policies.play --url http://127.0.0.1:8801/mcp rules
    python -m policies.play --url ... crystallize sample_id=S3 fraction=0.4
    python -m policies.play --url ... combine sample_ids=S4,S7
"""
from __future__ import annotations

import argparse
import json

from policies.mcp_env import McpEnv

LIST_ARGS = {"sample_ids"}
FLOAT_ARGS = {"fraction", "hours", "minutes", "atomic_weight"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("tool")
    ap.add_argument("args", nargs="*", help="key=value")
    a = ap.parse_args()
    kw = {}
    for item in a.args:
        k, _, v = item.partition("=")
        if k in LIST_ARGS:
            kw[k] = [x for x in v.split(",") if x]
        elif k in FLOAT_ARGS:
            try:
                kw[k] = float(v)
            except ValueError:
                kw[k] = v
        else:
            kw[k] = v
    env = McpEnv(a.url)
    try:
        print(json.dumps(env.call(a.tool, **kw), indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
