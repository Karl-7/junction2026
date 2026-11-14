import json, math, os, random, re, subprocess, sys, pathlib
import pytest
from hidden_element.config import DEFAULT as CFG
from hidden_element.lab import Lab
from hidden_element import world as W
from hidden_element.rules import rules_text
from policies import baselines, reference

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = [0, 1, 2, 3, 7, 42]
REAL_SYMBOLS = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir
Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og""".split())
REAL_NAMES = ["radium", "barium", "uranium", "polonium", "bismuth", "thorium", "actinium", "lead", "calcium",
              "strontium", "radon", "francium", "protactinium", "thallium", "potassium", "sodium", "iron",
              "hydrogen", "helium", "oxygen", "silicon", "copper", "zinc", "silver", "gold", "plutonium"]


def scripted(lab):
    """A fixed sequence touching every wet op; returns nothing."""
    lab.call("dissolve", sample_id="ore")
    r = lab.call("precipitate", sample_id="ore", reagent="reagent_A")
    p, f = r["precipitate"], r["filtrate"]
    lab.call("weigh", sample_id=p); lab.call("spectrum", sample_id=f)
    lab.call("measure_activity", sample_id=p, minutes=2)
    lab.call("dissolve", sample_id=p)
    c = lab.call("crystallize", sample_id=p, fraction=0.4)
    lab.call("atomic_weight", sample_id=f)
    lab.call("wait", hours=50)
    r2 = lab.call("crystallize", sample_id=f, fraction=0.5)
    lab.call("dissolve", sample_id=c["crystals"])
    m = lab.call("combine", sample_ids=[c["crystals"], c["mother_liquor"]])
    lab.call("combine", sample_ids=[m["sample_id"], r2["mother_liquor"]])
    lab.call("list_samples")


def total_mass(lab):
    return sum(sum(s["m"].values()) for s in lab.samples.values())


def test_mass_conservation():
    for seed in SEEDS:
        lab = Lab(seed); scripted(lab)
        for k, g in lab.world.ore.items():
            tot = sum(s["m"][k] for s in lab.samples.values()) + lab.lost[k]
            assert tot == pytest.approx(g, rel=1e-12), (seed, k)


def test_loss_fraction_per_wet_op():
    lab = Lab(5)
    ops = [("dissolve", dict(sample_id="ore"))]
    for tool, args in ops:
        before = {k: lab.lost[k] for k in lab.lost}; m0 = dict(lab.samples["ore"]["m"])
        lab.call(tool, **args)
        for k in m0:
            assert lab.lost[k] - before[k] == pytest.approx(CFG.op_loss * m0[k], rel=1e-12)
    for tool, args in [("precipitate", dict(sample_id="ore", reagent="reagent_B"))]:
        m0 = dict(lab.samples["ore"]["m"]); l0 = dict(lab.lost)
        r = lab.call(tool, **args)
        for k in m0:
            assert lab.lost[k] - l0[k] == pytest.approx(CFG.op_loss * m0[k], rel=1e-12)
            got = lab.samples[r["precipitate"]]["m"][k] + lab.samples[r["filtrate"]]["m"][k]
            assert got == pytest.approx((1 - CFG.op_loss) * m0[k], rel=1e-12)
    sid = r["filtrate"]; m0 = dict(lab.samples[sid]["m"]); l0 = dict(lab.lost)
    lab.call("crystallize", sample_id=sid, fraction=0.5)
    for k in m0:
        assert lab.lost[k] - l0[k] == pytest.approx(CFG.op_loss * m0[k], rel=1e-12)
    # non-wet ops lose nothing
    l1 = dict(lab.lost); lab.call("wait", hours=5); lab.call("weigh", sample_id=r["precipitate"])
    assert lab.lost == l1


def run_actions(seed, acts):
    lab = Lab(seed)
    for t, a in acts:
        lab.call(t, **a)
    return lab


def test_determinism_and_seed_difference():
    acts = [("dissolve", dict(sample_id="ore")), ("precipitate", dict(sample_id="ore", reagent="reagent_C")),
            ("weigh", dict(sample_id="S1")), ("spectrum", dict(sample_id="S2")),
            ("measure_activity", dict(sample_id="S1", minutes=3))]
    a, b = run_actions(3, acts), run_actions(3, acts)
    assert json.dumps(a.log, sort_keys=True) == json.dumps(b.log, sort_keys=True)
    assert json.dumps(a.trajectory(), sort_keys=True) == json.dumps(b.trajectory(), sort_keys=True)
    ta, tb = Lab(3).world.truth(), Lab(4).world.truth()
    assert json.dumps(ta["elements"], sort_keys=True) != json.dumps(tb["elements"], sort_keys=True)
    assert ta["ore_g"] != tb["ore_g"]


def ref_run(seed):
    lab = Lab(seed); reference.run(lab); return lab


def test_reference_bitwise_repeatable():
    for seed in SEEDS[:4]:
        a, b = ref_run(seed), ref_run(seed)
        assert json.dumps(a.trajectory(), sort_keys=True) == json.dumps(b.trajectory(), sort_keys=True)
        assert a.final_reward() == b.final_reward()


def test_hashseed_independence():
    code = ("import json,hashlib\nfrom hidden_element.lab import Lab\nfrom policies import reference\n"
            "out=[]\nfor s in (0,1,2):\n    l=Lab(s); reference.run(l); out.append(json.dumps(l.trajectory(),sort_keys=True))\n"
            "print(hashlib.sha256('|'.join(out).encode()).hexdigest())\n")
    outs = set()
    for hs in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(ROOT))
        outs.add(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                                text=True, check=True).stdout.strip())
    assert len(outs) == 1


def test_reward_bounds_and_trivial():
    for seed in SEEDS:
        for pol in (reference.run, baselines.submit_ore, baselines.one_step,
                    lambda e: baselines.random_ops(e, seed=seed)):
            lab = Lab(seed); pol(lab)
            assert 0.0 <= lab.final_reward() <= 1.0
        lab = Lab(seed); baselines.do_nothing(lab)
        assert lab.final_reward() == 0.0 and not lab.done


def test_random_episodes_bounded():
    for i in range(200):
        lab = Lab(i % 20); baselines.random_ops(lab, seed=i, n_ops=random.Random(i).randint(0, 60))
        assert 0.0 <= lab.final_reward() <= 1.0
        assert math.isfinite(total_mass(lab))


def test_budget_exhaustion_scores_zero():
    lab = Lab(0)
    for _ in range(CFG.max_actions):
        out = lab.call("weigh", sample_id="ore")
    assert lab.done and lab.final_reward() == 0.0 and "episode_over" in out
    assert lab.actions == CFG.max_actions
    assert "error" in lab.call("weigh", sample_id="ore")
    assert "error" in lab.call("submit", sample_id="ore", atomic_weight=200.0, carrier="Qa")
    assert lab.final_reward() == 0.0


def test_no_real_element_names_or_symbols():
    pat = re.compile(r"\b(" + "|".join(REAL_NAMES) + r")\b", re.I)
    assert not pat.search(rules_text(CFG))
    files = [p for d in ("hidden_element", "policies") for p in (ROOT / d).rglob("*.py")]
    assert files
    for p in files:
        m = pat.search(p.read_text())
        assert not m, (p, m.group(0))
    assert not (set(W.SYMBOL_POOL) & REAL_SYMBOLS)
    assert len(REAL_SYMBOLS) == 118


def numbers_in(text):
    return set(re.findall(r"(?<![\w.])-?\d+\.\d+(?![\w.])", text))


def test_rules_payload_no_leak():
    pub_keys = {"symbol", "atomic_weight", "spectral_lines_nm", "line_sensitivity", "ore_mass_fraction",
                "specific_activity_cpm_per_g", "half_life"}
    for seed in SEEDS:
        lab = Lab(seed); p = lab.call("rules")
        assert set(p) == {"rules", "known_elements", "reagents", "budgets"}
        for e in p["known_elements"]:
            assert set(e) == pub_keys
        syms = [e["symbol"] for e in p["known_elements"]]
        assert syms == sorted(syms)  # no positional hint about the carrier
        text = json.dumps(p)
        known_w = {f"{e['atomic_weight']}" for e in p["known_elements"]}
        for key in (W.TARGET, W.INTERFERER):
            e = lab.world.elements[key]
            for v in (f"{round(e.atomic_weight, 2)}", f"{e.atomic_weight}"):
                if v not in known_w:
                    assert v not in numbers_in(text), (seed, key)
            assert repr(e.beta) not in text
        for el in lab.world.elements.values():
            for f in el.precip:
                assert len(repr(f)) < 8 or repr(f) not in text  # skip clamped round values
        assert W.TARGET not in json.dumps(p["known_elements"]) and W.INTERFERER not in json.dumps(p["known_elements"])


def test_errors_cost_no_actions():
    lab = Lab(1)
    r = lab.call("dissolve", sample_id="ore"); assert lab.actions == 1
    pr = lab.call("precipitate", sample_id="ore", reagent="reagent_A")
    n = lab.actions
    bad = [("weigh", dict(sample_id="nope")), ("dissolve", dict(sample_id="nope")),
           ("crystallize", dict(sample_id=pr["filtrate"], fraction=0.99)),
           ("crystallize", dict(sample_id=pr["filtrate"], fraction=0.0)),
           ("precipitate", dict(sample_id=pr["precipitate"], reagent="reagent_B")),
           ("precipitate", dict(sample_id=pr["filtrate"], reagent="reagent_Z")),
           ("dissolve", dict(sample_id=pr["filtrate"])), ("combine", dict(sample_ids=[pr["filtrate"]])),
           ("combine", dict(sample_ids=[pr["filtrate"], pr["precipitate"]])),
           ("wait", dict(hours=-1)), ("wait", dict(hours=1e6)), ("measure_activity", dict(sample_id="ore", minutes=0)),
           ("bogus_tool", {}), ("weigh", {})]
    t0 = lab.t
    for t, a in bad:
        out = lab.call(t, **a)
        assert "error" in out, (t, a)
        assert lab.actions == n and lab.t == t0
    assert lab.final_reward() == 0.0 and not lab.done


def test_reagent_budget_enforced():
    lab = Lab(2); lab.call("dissolve", sample_id="ore")
    cur = "ore"
    for i in range(CFG.reagent_units):
        cur = lab.call("precipitate", sample_id=cur, reagent="reagent_A")["filtrate"]
    assert lab.reagent_left["reagent_A"] == 0
    n = lab.actions
    out = lab.call("precipitate", sample_id=cur, reagent="reagent_A")
    assert "error" in out and lab.actions == n
    assert "error" not in lab.call("precipitate", sample_id=cur, reagent="reagent_B")
