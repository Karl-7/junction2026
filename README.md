# Reasoning Like Curie

**How close is LLM reasoning to human reasoning during a scientific discovery?**

Claude, GPT and Gemini models each work through the same simulated radiochemistry problem: isolate an
unknown radioactive element from an ore, much as Marie Curie did. They have to separate the ore chemically,
follow the radioactivity, concentrate the element by fractional crystallisation and report what they found.
Every tool call must carry a one-sentence **key reason** for the decision. The site then shows, step by step,
what each model did and why, next to a human doing the same seed. It also scores each model on two axes:

* **Task performance:** reward in [0, 1], computed from the true contents of the submitted sample.
* **Human-likeness:** how similar its decisions, order of work, tool use and stated reasons are to the
  human runs on the same seed.

Each seed generates new hidden chemistry. Recalling facts about real elements does not help (the
environment uses no real element names), so a model has to reason from the evidence it collects.

## Quick start

```bash
cp .env.example .env          # then fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
python -m llm_bench.human protocol --seeds 101 102 103      # human expert reference (already committed)
python -m llm_bench.run --seeds 101 102 103                  # every model in llm_bench/models.json
open docs/index.html                                         # comparison; docs/runs.html = decisions & reasons
```

Only Python 3.10+ is needed. API calls use the standard library, so there are no SDK dependencies (`pytest` is used for tests).

Useful variants:

```bash
python -m llm_bench.run --models claude-opus-5-5 gpt-5 gemini-3.5-flash --seeds 101
python -m llm_bench.run --families Claude --seeds 101 102 103 --skip-existing --parallel 3
python -m llm_bench.site                                     # rebuild docs/data.js from results/runs
python -m scripts.show_trajectory results/runs/<model>/seed101.json --truth
```

To change which models are compared, edit [`llm_bench/models.json`](llm_bench/models.json). In that file, `id` is the API model
name and `extra` is merged into every request body, for example to switch on thinking or set the reasoning effort. A model id
that isn't listed is also accepted on the command line if it starts with `claude`, `gpt`, `o` or `gemini`.
OpenAI calls use the Responses API by default. Set `OPENAI_API_MODE=chat` to use Chat Completions or an
OpenAI-compatible endpoint (`OPENAI_BASE_URL`).

## Human references

* **Human expert protocol** (`policies/reference.py`, agent id `human-expert-protocol`): a fixed procedure modelled
  on the Curies' method. It waits out the short-lived interferer, follows the activity through every
  precipitation, then runs fractional crystallisation with mother-liquor recycling. A human expert's one-sentence
  reason is attached to every step.
* **Real people** can play in the terminal, and the run is saved as an additional human reference:

  ```bash
  python -m llm_bench.human play --seed 101 --name alice
  ```

  Commands look like tool calls (`precipitate sample_id=ore reagent=reagent_A`). After each command you are asked for your key reason.

## What is measured (`llm_bench/metrics.py`)

| Metric | Meaning |
|---|---|
| reward, purity, recovery, carrier ✓, weight error | task outcome (server-side scoring in `hidden_element/scoring.py`) |
| **decisions** | agreement with the human on 7 key decisions: wait out the interferer, compare both products of a separation, repeat fractional crystallisation, recycle mother liquors, check the spectrum, don't stop early, submit |
| **sequence** | 1 − normalised edit distance between phase sequences (separation → crystallisation → recycling → analysis …) |
| **tool mix** | 1 − Jensen–Shannon distance between the distributions of lab tools used |
| **concepts** | cosine similarity of which scientific concepts the reasons invoke (background, decay, specific activity, carrier, recovery, purity, noise, …) |
| **human-likeness** | mean of decisions, sequence, tool mix and concepts |
| reasons given, words / reason | how well the model followed the one-sentence-reason requirement |

## Layout

```
hidden_element/   the simulated lab: world generation, operations, rules text, scoring (hidden state stays here)
policies/         human expert protocol (reference.py) and trivial baselines used in tests
llm_bench/        providers (Anthropic / OpenAI / Gemini), agent loop, human play, metrics, site data
results/runs/     one JSON per agent and seed: every step with tool, args, reason, result; outcome; hidden truth
results/raw/      raw API transcripts (git-ignored)
docs/             static site: index.html (comparison), runs.html (decisions & reasons, side by side)
tests/            environment tests (determinism, exploits, leaks) and pipeline tests (no network)
```

The model sees only the system prompt, the task and rules text (`llm_bench/tools.py`) and the result of each tool call.
The hidden truth is written to the run file only after the episode ends.

```bash
python -m pytest -q
```
