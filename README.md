# Reasoning Like Curie

**Are the reasons behind an LLM's decisions actually right?**

We give language models (currently five Gemini models) a lab puzzle: find and purify an unknown radioactive
element hidden in an ore, the way Marie Curie did. The models work through tools (dissolve, precipitate,
crystallise, measure, …). **Every time a model acts, it must say why in one sentence.**

Most benchmarks only score the final answer. Here the simulator knows the hidden truth at every step, so we
can also grade the *reasoning process*: was each decision right, did the model follow its own evidence, were
the facts it stated true, and did it actually do what it said?

![Overview: Gemini models (and optionally Claude, GPT, human players and a Curie-style script) solve the same lab task; every action carries a one-sentence reason; runs are scored on task reward and on grounded reasoning checked against the hidden truth](docs/images/overview.svg)

## The puzzle

Each run generates new hidden chemistry, so memorised facts don't help. The model has to reason from its own
measurements. A careful chemist goes through these steps:

![The five steps of the Curie-style approach: wait for the short-lived element to decay, separate chemically while following the radioactivity, crystallise, recycle the mother liquor, then submit](docs/images/episode.svg)

A fixed script that follows these steps is included as an **expert baseline** (`policies/reference.py`).
It is a reference for performance, not ground truth: the ground truth is the simulator's hidden state.

## The metric: Grounded Reasoning Score (GRS)

The run is replayed on the deterministic simulator and every step is checked against the hidden state
(`llm_bench/grounded.py`). No human labels are needed.

| Metric | Question it answers |
|---|---|
| **Decision accuracy** | After each split, did the agent carry on with the product that really holds the target? |
| **Evidence consistency** | When it measured both products first, did it follow its own measurements? |
| **Claim truth** | Are the facts stated in its reasons true at that moment? ("the interferer has decayed", "X is the carrier", "purity is ~N%", "the activity went into the filtrate") |
| **Faithfulness** | When a reason says which product to follow or discard, does the call act on that product? |
| **Prediction accuracy** *(optional)* | Optional `expect_*` fields (where the target will go, which product gets enriched, expected count rate), checked against what happened. How often a model predicts is reported too. |

**GRS** is the mean of the first four (those that could be checked in a run). Wrong decisions are also split
by cause, which is often the most telling part:

* **misled by evidence**: it followed its measurements, but they were misleading (e.g. it didn't wait for the
  short-lived interferer to decay);
* **against its own evidence**: its measurements pointed the right way and it went the other way;
* **guessed**: it never measured both products.

Also reported: task reward (purity, recovery, atomic weight, carrier), a strategy checklist, which concepts the
reasons mention, and, once real people have played, similarity to human players.

## The website

`docs/index.html` compares all models (reward vs. GRS, leaderboard, why decisions went wrong, strategy).
`docs/runs.html` shows every decision with its one-sentence reason and the checks against the hidden truth,
for two agents side by side.

![The decisions-and-reasons page, showing the Curie-style baseline's run step by step with the reason for each action](docs/images/reasons-viewer.png)

*Screenshot: the Curie-style baseline on seed 101. Model runs appear here once you run the models.*

## Quick start

```bash
cp .env.example .env                       # put your GEMINI_API_KEY in .env
python -m llm_bench.gemini_models flash    # check the exact model ids your key can use
python -m llm_bench.run --seeds 101        # every enabled model in llm_bench/models.json
open docs/index.html
```

Only Python 3.10+ is needed. API calls use the standard library, so there are no SDK dependencies (`pytest` is used for tests).

**Models.** [`llm_bench/models.json`](llm_bench/models.json) enables five Gemini models: 2.5 Flash Lite, 2.5 Flash,
3 Flash, 3.1 Flash Lite and 3.5 Flash Lite. Claude and GPT entries are there but disabled (`"enabled": false`).
If an id is wrong for your key, replace it with one printed by `python -m llm_bench.gemini_models`.

**Rate limits.** One episode takes roughly 60 model requests. Set each model's `rpm` and `rpd` (requests per minute
and per day) to your API tier's limits. Requests are then spaced automatically, retries are printed, and the run
warns up front when a daily limit is too low to finish an episode. On the free tier, models limited to 20 requests
per day cannot finish one episode.

More commands:

```bash
python -m llm_bench.run --models gemini-2.5-flash --seeds 101 102 103 --skip-existing
python -m llm_bench.human protocol --seeds 101 102 103     # re-record the Curie-style baseline
python -m llm_bench.human play --seed 101 --name alice     # play yourself; your runs become human references
python -m llm_bench.site                                   # rebuild docs/data.js from results/runs
python -m scripts.show_trajectory results/runs/<model>/seed101.json --truth
python -m pytest -q
```

## Layout

```
hidden_element/   the simulated lab: world generation, operations, rules text, scoring (hidden state stays here)
policies/         Curie-style expert baseline (reference.py) and trivial baselines used in tests
llm_bench/        providers (Gemini / Anthropic / OpenAI), agent loop, human play, grounded metrics, site data
results/runs/     one JSON per agent and seed: every step with tool, args, reason, prediction, result; outcome; hidden truth
results/raw/      raw API transcripts (git-ignored)
docs/             static site: index.html (comparison), runs.html (decisions & reasons, side by side)
tests/            environment tests (determinism, exploits, leaks), metric tests and pipeline tests (no network)
```

The model sees only the system prompt, the task and rules text (`llm_bench/tools.py`) and the result of each tool call.
The hidden truth is written to the run file only after the episode ends.
