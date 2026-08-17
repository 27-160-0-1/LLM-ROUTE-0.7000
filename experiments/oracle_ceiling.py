"""Oracle ceiling: EV with perfect score/cost knowledge under tier budgets."""

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes

policy = load_bundled_policy()
inputs = load_input(HERE / "data/combined/inputs.json")
outcomes = load_outcomes(HERE / "data/combined/outcomes.json")
episodes = inputs.episodes
n = len(episodes)
index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}


def true_cost(eid, mid):
    o = index[(eid, mid)]
    r = policy.models[mid]
    unit = Decimal(policy.token_unit)
    return float(r.fixed_cost + Decimal(o.input_tokens) * r.input_token_rate / unit
                 + Decimal(o.output_tokens) * r.output_token_rate / unit)


true_s = np.array([[float(index[(e.episode_id, m)].score) for m in MODEL_IDS] for e in episodes])
true_c = np.array([[true_cost(e.episode_id, m) for m in MODEL_IDS] for e in episodes])


def allocate(ps, pc, mult):
    lt = pc[:, 0].sum(); cap = lt * mult

    def choose(pen):
        u = ps - pen * pc / lt
        pick = np.argmax(u + np.array([2e-12, 1e-12, 0.0]), axis=1)
        return pick, pc[np.arange(len(pick)), pick].sum()

    pick, tot = choose(0.0)
    if tot > cap:
        lo, hi = 0.0, 1.0
        pick, tot = choose(hi)
        while tot > cap and hi < 2**60:
            lo, hi = hi, hi * 2
            pick, tot = choose(hi)
        for _ in range(60):
            mid = (lo + hi) / 2
            c2, t2 = choose(mid)
            if t2 <= cap:
                hi, pick, tot = mid, c2, t2
            else:
                lo = mid
    if tot > cap:
        pick = np.zeros(len(ps), dtype=int)
    return pick


rng2 = np.random.default_rng(7)
samples = [rng2.integers(0, n, size=880) for _ in range(400)]
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}

tot_ev = 0.0
for tier, mult in MULTS.items():
    evs = []
    for sample in samples:
        p = allocate(true_s[sample], true_c[sample], mult)
        r = np.arange(len(sample))
        ratio = true_c[sample][r, p].sum() / true_c[sample][:, 0].sum()
        evs.append(0.0 if ratio > mult else true_s[sample][r, p].mean())
    ev = float(np.mean(evs))
    full = allocate(true_s, true_c, mult)
    full_sc = float(true_s[np.arange(n), full].mean())
    tot_ev += W[tier] * ev
    print(f"[oracle] {tier}: bootstrap EV {ev:.4f} | full-set score {full_sc:.4f}", flush=True)
print(f"[oracle] weighted EV ceiling {tot_ev:.4f}", flush=True)

means = true_s.mean(axis=0)
print(f"[oracle] all-light {means[0]:.4f} all-mid {means[1]:.4f} all-think {means[2]:.4f}", flush=True)
best_any = float(true_s.max(axis=1).mean())
print(f"[oracle] unconstrained best-of-3 {best_any:.4f}", flush=True)
