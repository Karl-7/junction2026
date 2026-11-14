"""Generate one Harbor task directory per seed under tasks/.

    python -m scripts.build_tasks --seeds 0 1 2

Layout per task:
  instruction.md, task.toml
  environment/Dockerfile            agent container ("main"): no simulation code, no seed
  environment/docker-compose.yaml   adds the "lab" sidecar holding the hidden state
  environment/lab/                  sidecar image: hidden_element package + MCP server
  tests/test.sh                     verifier: fetches the server-side reward with a secret token
  solution/                         oracle = the reference policy, driven over MCP
"""
from __future__ import annotations

import argparse
import secrets
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hidden_element.config import get_config  # noqa: E402
from hidden_element.rules import rules_text  # noqa: E402

MCP_PIN = "mcp==1.26.0"
PY_IMAGE = "python:3.12-slim-bookworm"

INSTRUCTION = """\
# Hidden Element

You are working in a simulated radiochemistry lab, reachable only through the MCP server
named `lab`. Its tools are your only way to act on the simulation; there is nothing to find
on this machine's filesystem.

Start by calling the `rules` tool: it returns the full rules, the scoring formula, budgets,
and the public data on the known elements and reagents. The rules are reproduced below.

You are scored on what you submit with the `submit` tool. The episode ends at `submit`, or
with reward 0 when the action budget runs out. There is no time pressure: only tool calls
advance the simulation.

{rules}
"""

TASK_TOML = """\
schema_version = "1.4"
artifacts = [
  {{ source = "/var/lab/trajectory.json", destination = "lab_trajectory.json", service = "lab" }},
]

[task]
name = "hidden-element/seed-{seed}"
version = "1.0.0"
description = "Isolate a hidden radioactive element by following the activity through chemical separations."
keywords = ["science", "simulation", "mcp", "radiochemistry"]

[metadata]
config = "{config}"
seed = {seed}

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 3600.0

[environment]
build_timeout_sec = 900.0
os = "linux"

[[environment.mcp_servers]]
name = "lab"
transport = "streamable-http"
url = "http://lab:8000/mcp"
"""

MAIN_DOCKERFILE = f"""\
FROM {PY_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \\
    && rm -rf /var/lib/apt/lists/*
# The MCP client library is only used by the oracle (reference policy); it holds no task data.
RUN pip install --no-cache-dir "{MCP_PIN}"
WORKDIR /app
"""

LAB_DOCKERFILE = f"""\
FROM {PY_IMAGE}
RUN pip install --no-cache-dir "{MCP_PIN}"
WORKDIR /srv
COPY hidden_element /srv/hidden_element
ENV LAB_LOG_DIR=/var/lab
EXPOSE 8000
CMD ["python", "-m", "hidden_element.server"]
"""

COMPOSE = """\
services:
  main:
    depends_on:
      lab:
        condition: service_healthy
  lab:
    build:
      context: ./lab
    environment:
      LAB_SEED: "{seed}"
      LAB_CONFIG: "{config}"
      GRADER_TOKEN: "{token}"
    expose:
      - "8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]
      interval: 2s
      timeout: 5s
      retries: 30
      start_period: 3s
"""

TEST_SH = """\
#!/bin/bash
# The reward is computed by the lab server from its hidden state; nothing the agent wrote is used.
set -u
mkdir -p /logs/verifier
if curl -sf -H "x-grader-token: {token}" http://lab:8000/_grade -o /logs/verifier/grade.json; then
  python3 -c "import json; d=json.load(open('/logs/verifier/grade.json')); print(d['reward'])" \\
    > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
cat /logs/verifier/reward.txt
"""

SOLVE_SH = """\
#!/bin/bash
# Reference policy: fixed procedure, uses only the same MCP tools as the agent.
set -e
cd /solution
python3 -m policies.mcp_env --url http://lab:8000/mcp --policy {policy}
"""


def build(seed: int, out: Path, config: str, policy: str = "reference"):
    cfg = get_config(config)
    token = secrets.token_hex(24)
    d = out / f"hidden-element-s{seed}"
    if d.exists():
        shutil.rmtree(d)
    (d / "environment" / "lab").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "solution" / "policies").mkdir(parents=True)

    (d / "instruction.md").write_text(INSTRUCTION.format(rules=rules_text(cfg)))
    (d / "task.toml").write_text(TASK_TOML.format(seed=seed, config=cfg.name))
    (d / "environment" / "Dockerfile").write_text(MAIN_DOCKERFILE)
    (d / "environment" / "docker-compose.yaml").write_text(COMPOSE.format(seed=seed, config=cfg.name, token=token))
    (d / "environment" / "lab" / "Dockerfile").write_text(LAB_DOCKERFILE)
    shutil.copytree(ROOT / "hidden_element", d / "environment" / "lab" / "hidden_element",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (d / "tests" / "test.sh").write_text(TEST_SH.format(token=token))
    (d / "solution" / "solve.sh").write_text(SOLVE_SH.format(policy=policy))
    for f in ("__init__.py", "reference.py", "baselines.py", "mcp_env.py"):
        shutil.copy(ROOT / "policies" / f, d / "solution" / "policies" / f)
    for f in (d / "tests" / "test.sh", d / "solution" / "solve.sh"):
        f.chmod(0o755)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=str(ROOT / "tasks"))
    ap.add_argument("--config", default="default-v0")
    ap.add_argument("--oracle-policy", default="reference", choices=["reference", "one_step", "submit_ore"],
                    help="what solution/solve.sh runs (one_step = the early-stop baseline)")
    a = ap.parse_args()
    for s in a.seeds:
        print(build(s, Path(a.out), a.config, a.oracle_policy))


if __name__ == "__main__":
    main()
