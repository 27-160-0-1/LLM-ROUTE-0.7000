"""E42 diagnostics: compare dumped OOF predictions of two runs (e.g. none vs parse).

python e42_diag.py reports/e42/preds_none_m0.5_s7.npz reports/e42/preds_parse_m0.5_s7.npz
Prints per-model bias of meta log-cost / score, linear-scale cost sum ratios, per-family bias,
and re-runs the tier evaluation with an EXTENDED safety grid to separate 'miscalibration the
safety factor can absorb' from 'genuinely worse ranking'.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ossp_router import similarity
from ossp_router.heuristic import episode_text
from ossp_router.protocol import load_input

ROOT = Path(__file__).resolve().parent.parent
A = np.load(sys.argv[1]); B = np.load(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
targets = A["targets"]; true_s = A["true_s"]; true_c = A["true_c"]
n = len(targets)
eps = load_input(ROOT / "data/combined/inputs.json").episodes
fams = np.array([similarity.classify_family(episode_text(e)) for e in eps])
MODELS = ["light", "mid", "think"]

for name, D in (("A", A), ("B", B)):
    m = D["meta_all"]
    print(f"== {name} {sys.argv[1 if name == 'A' else 2]}")
    for j, mo in enumerate(MODELS):
        ps, pl = m[:, j], m[:, 3 + j]
        print(f"  {mo:6s} score bias {np.mean(ps - targets[:, j]):+.4f} rmse {np.sqrt(np.mean((ps-targets[:, j])**2)):.4f} | "
              f"logcost bias {np.mean(pl - targets[:, 3+j]):+.4f} rmse {np.sqrt(np.mean((pl-targets[:, 3+j])**2)):.4f} | "
              f"sum exp(pred)/sum true {np.exp(np.clip(pl,-50,50)).sum()/true_c[:, j].sum():.3f}")
print("== per-family light/think logcost bias (A -> B)")
for f in sorted(set(fams)):
    i = fams == f
    a0 = np.mean(A["meta_all"][i, 3] - targets[i, 3]); b0 = np.mean(B["meta_all"][i, 3] - targets[i, 3])
    a2 = np.mean(A["meta_all"][i, 5] - targets[i, 5]); b2 = np.mean(B["meta_all"][i, 5] - targets[i, 5])
    ra = np.exp(A["meta_all"][i, 3]).sum() / true_c[i, 0].sum(); rb = np.exp(B["meta_all"][i, 3]).sum() / true_c[i, 0].sum()
    print(f"  {f:16s} n={i.sum():4d} light {a0:+.3f}->{b0:+.3f} (lin ratio {ra:.2f}->{rb:.2f}) | think {a2:+.3f}->{b2:+.3f}")


def allocate(ps, pc, mult, safety):
    lt = pc[:, 0].sum(); cap = lt * max(1.0, mult * safety)

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
        for _ in range(40):
            mid = (lo + hi) / 2
            c2, t2 = choose(mid)
            if t2 <= cap:
                hi, pick, tot = mid, c2, t2
            else:
                lo = mid
    if tot > cap:
        pick = np.zeros(len(ps), dtype=int)
    return pick


TIER_BLENDS = {"fast": 0.6, "balanced": 0.3, "premium": 0.45}
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
GRID = np.arange(0.70, 1.001, 0.01)
rng2 = np.random.default_rng(seed)
samples = [rng2.integers(0, n, size=880) for _ in range(200)]
for name, D in (("A", A), ("B", B)):
    meta_all = D["meta_all"]; rank_gain = D["rank_gain"]; prod_all = D["prod_all"]
    mixed = 0.75 * meta_all[:, 6:8] + 0.25 * rank_gain
    meta = meta_all[:, :6].copy()
    recon = np.column_stack([meta[:, 0], meta[:, 0] + mixed[:, 0], meta[:, 0] + mixed[:, 0] + mixed[:, 1]])
    meta[:, :3] = 0.5 * meta[:, :3] + 0.5 * recon
    tot = 0.0; parts = []
    for tier, mult in MULTS.items():
        st = (1 - TIER_BLENDS[tier]) * prod_all + TIER_BLENDS[tier] * meta
        ps = np.clip(st[:, :3], 0, 1); pc = np.exp(np.clip(st[:, 3:], -50, 50))
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12)); pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        best = None
        for s in GRID:
            evs = []; busts = 0
            for sample in samples:
                p = allocate(ps[sample], pc[sample], mult, s)
                r = np.arange(len(sample))
                ratio = true_c[sample][r, p].sum() / true_c[sample][:, 0].sum()
                if ratio > mult:
                    busts += 1; evs.append(0.0)
                else:
                    evs.append(true_s[sample][r, p].mean())
            ev = float(np.mean(evs))
            if best is None or ev > best[0]:
                best = (ev, s, busts / len(samples))
        tot += W[tier] * best[0]
        parts.append(f"{tier} {best[0]:.4f}@{best[1]:.2f} bust {best[2]:.2f}")
    print(f"== {name} extended-grid weighted EV {tot:.4f} | " + "; ".join(parts))
