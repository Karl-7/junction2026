"""Provider-neutral tool definitions and the task prompt given to every model.

Every tool takes a required `reason` argument: one sentence giving the key reason for the
decision. It is recorded with the call and stripped before the call reaches the lab.
"""
from __future__ import annotations

from hidden_element.config import Config
from hidden_element.rules import rules_text

REASON = {
    "type": "string",
    "description": "One sentence (at most ~25 words): the key reason you are making this call now.",
}


def _s(d=""):
    return {"type": "string", "description": d}


def _n(d=""):
    return {"type": "number", "description": d}


def _e(values, d=""):
    return {"type": "string", "enum": values, "description": d}


def _tool(name: str, desc: str, props: dict | None = None, required: list | None = None) -> dict:
    props = props or {}
    return {
        "name": name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": {"reason": REASON, **props},
            "required": ["reason", *(required if required is not None else list(props))],
        },
    }


TOOLS = [
    _tool("rules", "Free. Full rules, scoring formula, budgets, public data on known elements and reagents."),
    _tool("list_samples", "Free. Current sample ids and states, reagent units left, actions left, clock."),
    _tool("dissolve", "Dissolve a solid sample (same id).", {"sample_id": _s()}),
    _tool("precipitate", "Add a reagent to a solution and filter into a solid precipitate and a filtrate solution.",
          {"sample_id": _s(), "reagent": _s("e.g. reagent_A"),
           "expect_target_in": _e(["precipitate", "filtrate", "unsure"],
                                  "Optional: where you expect most of the target to end up.")},
          ["sample_id", "reagent"]),
    _tool("crystallize", "Fractional crystallisation of a solution; crystals hold `fraction` (0.05-0.95) of the mass.",
          {"sample_id": _s(), "fraction": _n(),
           "expect_enriched_in": _e(["crystals", "mother_liquor", "unsure"],
                                    "Optional: which product you expect to be richer in the target.")},
          ["sample_id", "fraction"]),
    _tool("combine", "Merge two or more samples of the same state.",
          {"sample_ids": {"type": "array", "items": {"type": "string"}}}),
    _tool("wait", "Let simulated time pass (hours); radioactive elements decay.", {"hours": _n()}),
    _tool("measure_activity", "Count radioactivity for `minutes` (0.1-60). Subtract the background.",
          {"sample_id": _s(), "minutes": _n(),
           "expect_net_cpm": _n("Optional: the background-subtracted count rate you expect.")},
          ["sample_id", "minutes"]),
    _tool("weigh", "Mass of a sample in grams (noisy).", {"sample_id": _s()}),
    _tool("spectrum", "Emission lines (nm, intensity) of elements above their detection limit.", {"sample_id": _s()}),
    _tool("atomic_weight", "Apparent (mass-fraction-weighted) atomic weight of the whole sample.", {"sample_id": _s()}),
    _tool("submit", "End the episode: submit a sample, the target's atomic weight and the carrier's symbol.",
          {"sample_id": _s(), "atomic_weight": _n(), "carrier": _s()}),
]
TOOL_NAMES = {t["name"] for t in TOOLS}
PREDICTION_FIELDS = {"expect_target_in", "expect_enriched_in", "expect_net_cpm"}

SYSTEM_PROMPT = """\
You are a scientist working in a simulated radiochemistry lab. You act only through the tools
provided; there is nothing else to inspect.

Every tool call MUST include the argument `reason`: ONE sentence stating the key reason for this
decision (what you expect to learn or achieve, and why now). Be concrete; do not restate the tool name.
You may make several independent tool calls in one turn. Some tools also have optional `expect_*` fields where you can record what you expect to happen
before seeing the result.
"""

TASK_PROMPT = """\
# Hidden Element

You are scored on what you submit with the `submit` tool. The episode ends at `submit`, or with
reward 0 when the action budget runs out. There is no time pressure: only tool calls advance the
simulation. The full rules follow (the `rules` tool returns the same text plus the public data on
known elements and reagents).

{rules}

Begin. Use the tools, give a one-sentence `reason` with every call, and finish with `submit`.
"""


def task_prompt(cfg: Config) -> str:
    return TASK_PROMPT.format(rules=rules_text(cfg))
