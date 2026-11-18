"""Grounded reasoning metrics: decisions, claims, faithfulness and predictions checked against hidden state."""
import pytest

from hidden_element.lab import Lab
from llm_bench import agent
from llm_bench.grounded import evaluate
from policies import reference

A = {"id": "x", "label": "x", "family": "Gemini", "provider": "x", "kind": "llm"}


def record(lab):
    return agent.build_record(lab, A)


class NoWait:
    """Reference procedure, but it does not wait for the interferer to decay."""
    def __init__(self, lab):
        self.lab = lab

    def call(self, tool, **a):
        if tool == "wait":
            a["hours"] = 0.1
        return self.lab.call(tool, **a)


@pytest.mark.parametrize("seed", [0, 1, 101])
def test_reference_scores_full_marks_and_replays_exactly(seed):
    lab = Lab(seed)
    reference.run(lab)
    m = evaluate(record(lab))["metrics"]
    assert m["decision_accuracy"] == 1.0 and m["evidence_consistency"] == 1.0
    assert m["replay_mismatches"] == 0 and m["grs"] == 1.0


def test_not_waiting_is_classified_as_misled_by_evidence():
    cats = {}
    for seed in range(6):
        lab = Lab(seed)
        reference.run(NoWait(lab))
        m = evaluate(record(lab))["metrics"]
        assert m["evidence_consistency"] == 1.0          # it did follow its (misleading) counts
        for k, v in m["decision_categories"].items():
            cats[k] = cats.get(k, 0) + v
    assert cats.get("misled_by_evidence", 0) >= 6
    assert not cats.get("against_evidence")


def test_going_against_own_measurements_is_classified(monkeypatch):
    orig = reference._follow
    monkeypatch.setattr(reference, "_follow", lambda env, a, b, bg: orig(env, a, b, bg)[::-1])
    lab = Lab(3)
    reference.run(lab)
    m = evaluate(record(lab))["metrics"]
    assert m["evidence_consistency"] == 0.0 and m["decision_accuracy"] == 0.0
    assert m["decision_categories"].get("against_evidence", 0) >= 1


def _split(lab):
    lab.call("wait", hours=240)
    lab.call("dissolve", sample_id="ore")
    r = lab.call("precipitate", sample_id="ore", reagent="reagent_A", reason="split")
    return r["precipitate"], r["filtrate"]


def _where(lab, a, b):
    from hidden_element.world import TARGET
    ta, tb = (lab._live(lab.samples[x]["m"])[TARGET] for x in (a, b))
    return ("precipitate", a, b) if ta >= tb else ("filtrate", b, a)


def test_claims_faithfulness_and_predictions():
    lab = Lab(5)
    lab.call("rules")
    lab.call("wait", hours=240, reason="Wait so the interferer decays.")       # intention, not a claim
    lab.call("dissolve", sample_id="ore")
    r = lab.call("precipitate", sample_id="ore", reagent="reagent_A", reason="Split with reagent_A.")
    lab.log[-1]["prediction"] = {"expect_target_in": "precipitate"}
    best_t, best, worst = _where(lab, r["precipitate"], r["filtrate"])
    wrong_t = "filtrate" if best_t == "precipitate" else "precipitate"
    carrier = lab.world.carrier
    other = next(e["symbol"] for e in lab.world.public()["known_elements"] if e["symbol"] != carrier)
    lab.call("measure_activity", sample_id=best, minutes=2,
             reason=f"The interferer has decayed, and the carrier is {other}.")
    lab.call("measure_activity", sample_id=worst, minutes=2, reason=f"The activity went into the {wrong_t}.")
    # says it follows the wrong product type but acts on the right sample -> unfaithful
    lab.call("dissolve" if best_t == "precipitate" else "crystallize", sample_id=best,
             **({} if best_t == "precipitate" else {"fraction": 0.5}), reason=f"Follow the {wrong_t}.")
    rec = record(lab)
    ev = evaluate(rec)
    m = ev["metrics"]
    kinds = [(c["kind"], c["ok"]) for cs in ev["checks"] for c in cs]
    assert ("claim", True) in kinds                      # interferer decayed after 240 h
    assert m["n_claims"] == 3 and m["claim_truth"] == pytest.approx(1 / 3)
    assert m["faithfulness"] == 0.0
    assert m["decision_accuracy"] == 1.0
    assert m["n_predictions"] == 1
    assert m["prediction_accuracy"] == (1.0 if best_t == "precipitate" else 0.0)
