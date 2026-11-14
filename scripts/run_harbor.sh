#!/usr/bin/env bash
# One command for the full judging protocol on the given seeds:
#   reference (oracle), early-stop baseline, do-nothing baseline (nop), and N agent runs.
#
#   MODEL=anthropic/claude-sonnet-4-5 SEEDS="0 1 2" AGENT_RUNS=3 scripts/run_harbor.sh
#
# Needs: docker, harbor (pip install harbor), and the model provider's API key in the env.
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2"}
AGENT_RUNS=${AGENT_RUNS:-3}
MODEL=${MODEL:?set MODEL, e.g. anthropic/claude-sonnet-4-5}
JOBS=${JOBS:-jobs}
STAMP=$(date +%Y%m%d-%H%M%S)

python3 -m scripts.build_tasks --seeds $SEEDS --out tasks/reference
python3 -m scripts.build_tasks --seeds $SEEDS --out tasks/early-stop --oracle-policy one_step

for s in $SEEDS; do
  t=hidden-element-s$s
  harbor run -p tasks/reference/$t  -a oracle -o "$JOBS" --job-name "$STAMP-s$s-reference"
  harbor run -p tasks/early-stop/$t -a oracle -o "$JOBS" --job-name "$STAMP-s$s-early-stop"
  harbor run -p tasks/reference/$t  -a nop    -o "$JOBS" --job-name "$STAMP-s$s-do-nothing"
  harbor run -p tasks/reference/$t  -a opencode -m "$MODEL" -k "$AGENT_RUNS" -o "$JOBS" --job-name "$STAMP-s$s-agent"
done
python3 -m scripts.summarize_jobs "$JOBS" --prefix "$STAMP"
