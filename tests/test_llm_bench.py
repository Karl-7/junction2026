"""The LLM comparison pipeline: agent loop, reason recording, metrics and site data (no network)."""
import json

from hidden_element.lab import Lab
from llm_bench import agent, metrics, site
from llm_bench.config import ModelSpec, find_model, load_models
from llm_bench.human import PROTOCOL_AGENT
from llm_bench.providers import ToolCall, Turn
from llm_bench.tools import TOOLS
from policies import reference


def protocol_run(seed):
    lab = Lab(seed)
    reference.run(lab)
    return agent.build_record(lab, PROTOCOL_AGENT, status="submitted")


class ReplayProvider:
    """Replays a list of (tool, args) calls as model turns, one call per turn."""

    def __init__(self, calls, drop_reason_every=0, chatter_first=False):
        self.calls, self.k, self.drop, self.chatter = list(calls), 0, drop_reason_every, chatter_first
        self.results = []

    def start(self, system, user):
        assert "reason" in system and "Hidden Element" in user

    def step(self):
        if self.chatter:
            self.chatter = False
            return Turn([], text="Let me think first.", usage={"input_tokens": 10, "output_tokens": 5})
        tool, args, reason = self.calls[self.k]
        self.k += 1
        a = dict(args)
        if not (self.drop and self.k % self.drop == 0):
            a["reason"] = reason
        return Turn([ToolCall(f"c{self.k}", tool, a)], usage={"input_tokens": 100, "output_tokens": 20})

    def add_results(self, results):
        self.results += results

    def add_user(self, text):
        self.results.append(("user", text))


def test_every_tool_requires_reason():
    for t in TOOLS:
        assert t["parameters"]["required"][0] == "reason"
        assert "reason" in t["parameters"]["properties"]


def test_lab_records_reason_and_strips_it():
    lab = Lab(0)
    out = lab.call("dissolve", sample_id="ore", reason="needs to be in solution")
    assert "error" not in out
    assert lab.log[-1]["reason"] == "needs to be in solution"
    assert "reason" not in lab.log[-1]["args"]


def test_agent_loop_replays_protocol(tmp_path):
    ref = protocol_run(101)
    calls = [(s["tool"], s["args"], s["reason"]) for s in ref["steps"]]
    p = ReplayProvider(calls, chatter_first=True)
    spec = ModelSpec(id="fake-model", label="Fake", provider="fake", family="Claude")
    rec = agent.run_episode(spec, 101, provider=p, runs_dir=tmp_path / "runs", raw_dir=tmp_path / "raw", verbose=False)
    assert rec["status"] == "submitted"
    assert rec["result"]["reward"] == ref["result"]["reward"]
    assert [s["reason"] for s in rec["steps"]] == [s["reason"] for s in ref["steps"]]
    assert rec["usage"]["requests"] == len(calls) + 1
    assert ("user", agent.NUDGE) in p.results
    assert (tmp_path / "runs" / "fake-model" / "seed101.json").exists()


def test_missing_reason_is_counted_and_flagged(tmp_path):
    ref = protocol_run(102)
    calls = [(s["tool"], s["args"], s["reason"]) for s in ref["steps"]]
    p = ReplayProvider(calls, drop_reason_every=4)
    spec = ModelSpec(id="fake2", label="Fake2", provider="fake", family="GPT")
    rec = agent.run_episode(spec, 102, provider=p, runs_dir=tmp_path, raw_dir=tmp_path / "raw", verbose=False)
    assert rec["missing_reasons"] == len(calls) // 4
    assert sum(1 for s in rec["steps"] if not s["reason"]) == len(calls) // 4
    assert any("note" in r for c, r in p.results if c != "user")
    assert metrics.reason_stats(rec)["coverage"] < 1


def test_bad_tool_and_bad_args_do_not_crash(tmp_path):
    calls = [("teleport", {}, "try"), ("crystallize", {"sample_id": "ore", "fraction": "lots"}, "x"),
             ("submit", {"sample_id": "ore", "atomic_weight": 200, "carrier": "Qa"}, "give up")]
    p = ReplayProvider(calls)
    rec = agent.run_episode(ModelSpec("f3", "F3", "fake", "Gemini"), 0, provider=p, runs_dir=tmp_path,
                            raw_dir=tmp_path / "raw", verbose=False)
    assert rec["status"] == "submitted"
    assert "error" in p.results[0][1] and "error" in p.results[1][1]


def test_similarity_identity_and_shortcut():
    a, b = protocol_run(101), protocol_run(101)
    b["agent"] = {**b["agent"], "id": "copy"}
    assert metrics.similarity(a, b)["human_likeness"] > 0.999
    assert all(metrics.strategy(a).values())
    lab = Lab(101)
    lab.call("submit", sample_id="ore", atomic_weight=200.0, carrier="Qa", reason="guess")
    short = agent.build_record(lab, {"id": "s", "label": "s", "family": "GPT", "provider": "x", "kind": "llm"})
    sim = metrics.similarity(short, a)
    assert sim["human_likeness"] < 0.4
    assert not metrics.strategy(short)["recycled_mother_liquor"]


def test_phases_cover_steps_and_detect_recycling():
    r = protocol_run(103)
    ph = metrics.label_phases(r)
    assert len(ph) == len(r["steps"]) and set(ph) <= set(metrics.PHASES)
    assert "recycling" in ph and ph[-1] == "submit"


def test_site_data(tmp_path):
    runs = tmp_path / "runs"
    for s in (101, 102):
        agent.save_record(protocol_run(s), runs)
        ref = protocol_run(s)
        llm = {**ref, "agent": {"id": "m", "label": "M", "family": "Gemini", "provider": "gemini", "kind": "llm"},
               "steps": ref["steps"][:40] + ref["steps"][-1:]}
        agent.save_record(llm, runs)
        human = {**ref, "agent": {"id": "human-a", "label": "A", "family": "Human", "provider": "human", "kind": "human"}}
        agent.save_record(human, runs)
    d = site.collect(runs)
    ids = {a["id"]: a for a in d["agents"]}
    assert d["has_humans"]
    assert ids["m"]["similarity"]["human_likeness"] < 1
    assert ids["human-a"]["similarity"] is None            # no other human to compare with
    assert ids["curie-style-baseline"]["grounded"]["grs"] == 1.0
    assert all("phase" in s and "reason" in s for r in d["runs"] for s in r["steps"])
    out = site.build(runs, tmp_path / "docs")
    assert json.loads(out.read_text()[len("window.BENCH = "):-2])["runs"]


def test_model_registry():
    ms = load_models()
    assert {m.family for m in ms} >= {"Claude", "GPT", "Gemini"}
    assert {m.family for m in ms if m.enabled} == {"Gemini", "DeepSeek"}
    assert find_model("deepseek-reasoner").provider == "deepseek"
    assert find_model("claude-opus-5-5").provider == "anthropic"
    assert find_model("gpt-9-turbo").provider == "openai"
    assert find_model("gemini-9").family == "Gemini"


def test_resume_after_quota_error_replays_then_continues(tmp_path, monkeypatch):
    """Quota runs out mid-episode; a resumed run replays the recorded responses (no API calls)
    and then continues live, ending exactly where an uninterrupted run would."""
    from llm_bench import providers as P
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    ref = protocol_run(101)
    script = [{"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": s["tool"], "args": {**s["args"], "reason": s["reason"]}}}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2}} for s in ref["steps"]]
    calls = {"n": 0}

    def fake_post(url, headers, body, log, **k):
        i = calls["n"]
        if quota["left"] <= 0:
            raise P.ProviderError("quota exhausted (HTTP 429): PerDay")
        quota["left"] -= 1
        calls["n"] += 1
        # the conversation sent must contain every earlier model turn (rebuilt exactly on resume)
        assert sum(1 for c in body["contents"] if c["role"] == "model") == i
        return script[i]

    monkeypatch.setattr(P, "post_json", fake_post)
    spec = ModelSpec(id="gemini-x", label="GX", provider="gemini", family="Gemini")
    kw = dict(runs_dir=tmp_path / "runs", raw_dir=tmp_path / "raw", verbose=False)
    quota = {"left": 30}
    first = agent.run_episode(spec, 101, **kw)
    assert first["status"].startswith("error") and len(first["steps"]) == 30
    quota["left"] = 10_000
    second = agent.run_episode(spec, 101, resume=True, **kw)
    assert second["status"] == "submitted"
    assert calls["n"] == len(script)                      # the 30 recorded ones were not requested again
    assert second["result"]["reward"] == ref["result"]["reward"]
    assert [s["reason"] for s in second["steps"]] == [s["reason"] for s in ref["steps"]]
