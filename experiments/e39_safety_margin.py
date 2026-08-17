"""E39: safety-margin optimisation for the deployed router.

The safety ratio s_tier scales the allocator's predicted-cost budget
(cap = light_total * multiplier * s).  Deployed values 0.98/0.89/0.88 were
picked on one bootstrap seed.  This run:

  1. Fine grid (0.005) x 3 seeds x 400 bootstraps of size 880  -> EV curve,
     bust probability, and the seed-averaged optimum per tier.
  2. Distribution-shift stress: re-weight bootstrap sampling so the private
     set is (a) harder (more math/code/aime), (b) longer-output (top think-
     cost quintile over-sampled), (c) smaller batch (N=440) and larger
     (N=1760) -> how the optimum moves and how much EV a conservative
     ratio costs ("insurance premium" table).
  3. Recommendation: the ratio maximising the WORST-CASE-weighted EV
     across the nominal + stress scenarios, and the EV cost of adopting it.

Uses the deployed OOF predictions (same nested CV as E27/E38).
"""

import math
import sys
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from ossp_router import learned_router, legacy_hash_regex, similarity
from ossp_router.heuristic import episode_text
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes

similarity.NEIGHBORS = 16

policy = load_bundled_policy()
inputs = load_input(ROOT / "data/combined/inputs.json")
outcomes = load_outcomes(ROOT / "data/combined/outcomes.json")
artifact = learned_router.load_artifact(ROOT / "src/ossp_router/resources/learned-router.v1.json")
legacy_artifact = legacy_hash_regex.load_artifact(ROOT / "src/ossp_router/resources/hash-regex-public.v1.json")

LEGACY_W, FAM_W, CONF_SCALE, GAIN_ALPHA = 0.75, 0.3, 0.4, 0.5
TIER_BLENDS = {"fast": 0.6, "balanced": 0.3, "premium": 0.45}
GBM_PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=30,
                  l2_regularization=3.0, early_stopping=True, validation_fraction=0.15, random_state=11)
ORD_T = (0.25, 0.5, 0.75, 1.0)
RANK_BETA = 0.25
LUT_NODES = 65
DEPLOYED = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}

episodes = inputs.episodes
n = len(episodes)
texts = [episode_text(e) for e in episodes]
index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}


def true_cost(eid, mid):
    o = index[(eid, mid)]
    r = policy.models[mid]
    unit = Decimal(policy.token_unit)
    return float(r.fixed_cost + Decimal(o.input_tokens) * r.input_token_rate / unit
                 + Decimal(o.output_tokens) * r.output_token_rate / unit)


true_s = np.array([[float(index[(e.episode_id, m)].score) for m in MODEL_IDS] for e in episodes])
true_c = np.array([[true_cost(e.episode_id, m) for m in MODEL_IDS] for e in episodes])
targets = np.hstack([true_s, np.log(true_c)])
delta_targets = np.column_stack([targets[:, 1] - targets[:, 0], targets[:, 2] - targets[:, 1]])
full_targets = np.hstack([targets, delta_targets])

print("[e39] shared blocks", flush=True)
from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

dense_rows, legacy_rows, fam_names = [], [], []
srows, scols, svals = [], [], []
for ri, episode in enumerate(episodes):
    dense = learned_router.raw_dense_features(episode)
    dense_rows.append(dense)
    items = learned_router.feature_items(
        episode, word_hash_bins=artifact.word_hash_bins, char_hash_bins=artifact.char_hash_bins,
        dense_mean=artifact.dense_mean, dense_scale=artifact.dense_scale, raw_dense=dense)
    for c, v in items.items():
        srows.append(ri); scols.append(c); svals.append(v)
    ls, lc = legacy_hash_regex.predict_episode(episode, legacy_artifact)
    legacy_rows.append([ls[m] for m in MODEL_IDS] + [math.log(lc[m]) for m in MODEL_IDS])
    fam_names.append(similarity.classify_family(texts[ri]))
dense_rows = np.asarray(dense_rows); legacy_rows = np.asarray(legacy_rows)
dim = len(learned_router.DENSE_FEATURE_NAMES) + artifact.word_hash_bins + artifact.char_hash_bins
X_sparse = sparse.csr_matrix((svals, (srows, scols)), shape=(n, dim))
FAMILIES = list(similarity.FAMILY_NAMES)
fam_idx = np.array([FAMILIES.index(f) for f in fam_names])
fam_onehot = np.zeros((n, len(FAMILIES)))
fam_onehot[np.arange(n), fam_idx] = 1.0

rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)
prod_all = np.zeros((n, 6)); meta_all = np.zeros((n, 8)); ord_score = np.zeros((n, 3)); rank_gain = np.zeros((n, 2))

for fold in range(5):
    t0 = time.perf_counter()
    hold = fold_of == fold
    fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
    ridge = Ridge(alpha=30.0, solver="sparse_cg").fit(X_sparse[fit_idx], targets[fit_idx])
    linear_hold = ridge.predict(X_sparse[hold_idx]); linear_hold[:, :3] = np.clip(linear_hold[:, :3], 0, 1)
    inner_oof = np.zeros((len(fit_idx), 6))
    inner_fold = np.random.default_rng(fold).integers(0, 5, size=len(fit_idx))
    for inner in range(5):
        ih = inner_fold == inner
        m = Ridge(alpha=30.0, solver="sparse_cg").fit(X_sparse[fit_idx[~ih]], targets[fit_idx[~ih]])
        inner_oof[ih] = m.predict(X_sparse[fit_idx[ih]])
    inner_oof[:, :3] = np.clip(inner_oof[:, :3], 0, 1)
    fit_texts = [texts[i] for i in fit_idx]
    freqs, total = similarity.document_frequencies(fit_texts)
    idf = similarity.idf_table(freqs, total)
    fit_vectors = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS) for t in fit_texts]
    knn_index = similarity.KnnIndex(fit_vectors, targets[fit_idx].tolist())
    gmean = targets[fit_idx].mean(axis=0)

    def kq(text, exclude=None):
        q = similarity.tfidf_vector(text, idf)
        if not q:
            return np.concatenate([gmean, [0.0]])
        scores = {}
        for g, v in q.items():
            for d, s in knn_index.postings.get(g, ()):
                if d == exclude:
                    continue
                scores[d] = scores.get(d, 0.0) + v * s
        if not scores:
            return np.concatenate([gmean, [0.0]])
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:similarity.NEIGHBORS]
        tot = sum(s for _d, s in ranked)
        row = np.zeros(6)
        for d, s in ranked:
            row += (s / tot) * targets[fit_idx][d]
        return np.concatenate([row, [ranked[0][1]]])

    knn_fit = np.array([kq(t, exclude=i) for i, t in enumerate(fit_texts)])
    knn_hold = np.array([kq(texts[i]) for i in hold_idx])
    fam_mean = {}
    by_family = defaultdict(list)
    for i in fit_idx:
        by_family[fam_names[i]].append(targets[i])
    fglobal = targets[fit_idx].mean(axis=0)
    for name in FAMILIES:
        rows = by_family.get(name, [])
        fam_mean[name] = np.mean(rows, axis=0) if len(rows) >= 8 else fglobal
    X_fit = np.hstack([dense_rows[fit_idx], fam_onehot[fit_idx], legacy_rows[fit_idx], inner_oof, knn_fit])
    X_hold = np.hstack([dense_rows[hold_idx], fam_onehot[hold_idx], legacy_rows[hold_idx], linear_hold, knn_hold])
    for hidx in range(8):
        meta_all[hold_idx, hidx] = HistGradientBoostingRegressor(**GBM_PARAMS).fit(X_fit, full_targets[fit_idx][:, hidx]).predict(X_hold)
    for mi in range(3):
        acc = np.zeros(len(hold_idx))
        for thr in ORD_T:
            y = (true_s[fit_idx, mi] >= thr).astype(int)
            if y.min() == y.max():
                acc += float(y.min()); continue
            acc += HistGradientBoostingClassifier(**GBM_PARAMS).fit(X_fit, y).predict_proba(X_hold)[:, 1]
        ord_score[hold_idx, mi] = acc / len(ORD_T)
    pc_hold = np.exp(np.clip(meta_all[hold_idx, 3:6], -50, 50))
    grid = np.linspace(0, 1, LUT_NODES)
    for g, (a, b) in enumerate([(0, 1), (1, 2)]):
        ds = true_s[:, b] - true_s[:, a]; dc_raw = true_c[:, b] - true_c[:, a]
        floor = max(float(np.quantile(dc_raw[fit_idx], 0.05)), 1e-9)
        eff = ds / np.maximum(dc_raw, floor)
        r_fit = rankdata(eff[fit_idx], method="average") / max(len(fit_idx) - 1, 1)
        q = np.quantile(eff[fit_idx], grid)
        dchat = np.maximum(pc_hold[:, b] - pc_hold[:, a], floor)
        m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(X_fit, r_fit)
        rank_gain[hold_idx, g] = np.interp(np.clip(m.predict(X_hold), 0, 1), grid, q) * dchat
    prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
    fam_rows_hold = np.array([fam_mean[fam_names[i]] for i in hold_idx])
    prod = (1 - FAM_W) * prod + FAM_W * fam_rows_hold
    conf = np.clip(knn_hold[:, 6], 0, 1)[:, None] * CONF_SCALE
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0, 1)
    prod_all[hold_idx] = prod
    print(f"[e39] fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)

meta = meta_all[:, :6].copy(); meta[:, :3] = ord_score
d1 = (1 - RANK_BETA) * meta_all[:, 6] + RANK_BETA * rank_gain[:, 0]
d2 = (1 - RANK_BETA) * meta_all[:, 7] + RANK_BETA * rank_gain[:, 1]
recon = np.column_stack([meta[:, 0], meta[:, 0] + d1, meta[:, 0] + d1 + d2])
meta[:, :3] = (1 - GAIN_ALPHA) * meta[:, :3] + GAIN_ALPHA * recon
PS, PC = {}, {}
for tier in TIER_BLENDS:
    st = (1 - TIER_BLENDS[tier]) * prod_all + TIER_BLENDS[tier] * meta
    PS[tier] = np.clip(st[:, :3], 0, 1)
    pc = np.exp(np.clip(st[:, 3:], -50, 50))
    pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12)); pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
    PC[tier] = pc


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


MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
GRID = {"fast": np.arange(0.90, 1.0001, 0.005), "balanced": np.arange(0.78, 0.98, 0.005), "premium": np.arange(0.76, 0.98, 0.005)}


def curve(tier, samples):
    """Return arrays over GRID[tier]: mean EV, bust prob, mean conditional score."""
    ps, pc, mult = PS[tier], PC[tier], MULTS[tier]
    evs, busts, conds = [], [], []
    for s in GRID[tier]:
        ev_l, b, c_l = [], 0, []
        for sample in samples:
            p = allocate(ps[sample], pc[sample], mult, s)
            r = np.arange(len(sample))
            ratio = true_c[sample][r, p].sum() / true_c[sample][:, 0].sum()
            sc = true_s[sample][r, p].mean()
            if ratio > mult:
                b += 1; ev_l.append(0.0)
            else:
                ev_l.append(sc)
            c_l.append(sc)
        evs.append(np.mean(ev_l)); busts.append(b / len(samples)); conds.append(np.mean(c_l))
    return np.array(evs), np.array(busts), np.array(conds)


def make_samples(seed, size=880, weights=None, R=400):
    rng_ = np.random.default_rng(seed)
    p = None if weights is None else weights / weights.sum()
    return [rng_.choice(n, size=size, replace=True, p=p) for _ in range(R)]


# ---- 1. nominal, 3 seeds
print("[e39] === nominal (880, 3 seeds) ===", flush=True)
nominal = {}
for tier in MULTS:
    ev_acc = np.zeros(len(GRID[tier])); bust_acc = np.zeros(len(GRID[tier])); cond_acc = np.zeros(len(GRID[tier]))
    per_seed_opt = []
    for seed in (7, 17, 23):
        ev, bust, cond = curve(tier, make_samples(seed))
        ev_acc += ev / 3; bust_acc += bust / 3; cond_acc += cond / 3
        per_seed_opt.append(GRID[tier][int(np.argmax(ev))])
    nominal[tier] = (ev_acc, bust_acc, cond_acc)
    i_opt = int(np.argmax(ev_acc)); i_dep = int(np.argmin(np.abs(GRID[tier] - DEPLOYED[tier])))
    print(f"[e39] {tier:8s} seed-avg optimum s={GRID[tier][i_opt]:.3f} EV {ev_acc[i_opt]:.4f} bust {bust_acc[i_opt]:.3f} | deployed s={DEPLOYED[tier]:.3f} EV {ev_acc[i_dep]:.4f} bust {bust_acc[i_dep]:.3f} | per-seed opt {[round(x,3) for x in per_seed_opt]}", flush=True)
    # print the curve compactly around the optimum
    lo = max(0, i_opt - 6); hi = min(len(GRID[tier]), i_opt + 7)
    for i in range(lo, hi):
        print(f"[e39]    s={GRID[tier][i]:.3f}  EV {ev_acc[i]:.4f}  bust {bust_acc[i]:.3f}  score|pass {cond_acc[i]:.4f}", flush=True)

# ---- 2. stress scenarios
print("[e39] === stress scenarios ===", flush=True)
hard_fams = {"aime", "hrmcr", "dmmath", "code", "gsm8k_or_other"}
w_hard = np.array([2.0 if fam_names[i] in hard_fams else 1.0 for i in range(n)])
think_q = np.quantile(true_c[:, 2], 0.8)
w_long = np.array([2.0 if true_c[i, 2] >= think_q else 1.0 for i in range(n)])
scenarios = {
    "harder(math/code x2)": dict(weights=w_hard),
    "longer-think(top20% x2)": dict(weights=w_long),
    "small batch N=440": dict(size=440),
    "large batch N=1760": dict(size=1760),
}
stress = {tier: {} for tier in MULTS}
for name, kw in scenarios.items():
    samples = make_samples(7, **kw)
    for tier in MULTS:
        ev, bust, cond = curve(tier, samples)
        stress[tier][name] = (ev, bust)
        i_opt = int(np.argmax(ev)); i_dep = int(np.argmin(np.abs(GRID[tier] - DEPLOYED[tier])))
        print(f"[e39] {name:24s} {tier:8s} opt s={GRID[tier][i_opt]:.3f} EV {ev[i_opt]:.4f} bust {bust[i_opt]:.3f} | deployed EV {ev[i_dep]:.4f} bust {bust[i_dep]:.3f}", flush=True)

# ---- 3. robust recommendation: maximise min over {nominal, stress} of (EV - EV_opt_scenario), i.e. minimal regret
print("[e39] === robust choice (min-regret across nominal + stress) and insurance table ===", flush=True)
recommend = {}
for tier in MULTS:
    g = GRID[tier]
    curves = [nominal[tier][0]] + [stress[tier][k][0] for k in scenarios]
    busts_ = [nominal[tier][1]] + [stress[tier][k][1] for k in scenarios]
    regret = np.max([c.max() - c for c in curves], axis=0)   # worst-case regret per s
    i_rob = int(np.argmin(regret))
    i_nom = int(np.argmax(nominal[tier][0]))
    i_dep = int(np.argmin(np.abs(g - DEPLOYED[tier])))
    recommend[tier] = g[i_rob]
    print(f"[e39] {tier:8s} nominal-opt s={g[i_nom]:.3f} | robust(min-regret) s={g[i_rob]:.3f} | deployed s={g[i_dep]:.3f}", flush=True)
    print(f"[e39]    nominal EV: opt {nominal[tier][0][i_nom]:.4f} robust {nominal[tier][0][i_rob]:.4f} deployed {nominal[tier][0][i_dep]:.4f}", flush=True)
    print(f"[e39]    worst-case regret: opt {regret[i_nom]:.4f} robust {regret[i_rob]:.4f} deployed {regret[i_dep]:.4f}", flush=True)
    print(f"[e39]    max bust across scenarios: opt {max(b[i_nom] for b in busts_):.3f} robust {max(b[i_rob] for b in busts_):.3f} deployed {max(b[i_dep] for b in busts_):.3f}", flush=True)
    # insurance premium table: EV cost (nominal) vs worst-case bust for a few conservative steps
    print(f"[e39]    insurance table ({tier}):", flush=True)
    for step in (0.0, -0.01, -0.02, -0.03, -0.05):
        s = g[i_nom] + step
        i = int(np.argmin(np.abs(g - s)))
        print(f"[e39]      s={g[i]:.3f}  nominal EV {nominal[tier][0][i]:.4f} ({nominal[tier][0][i]-nominal[tier][0][i_nom]:+.4f})  worst bust {max(b[i] for b in busts_):.3f}  worst regret {regret[i]:.4f}", flush=True)

tot_dep = sum(W[t] * nominal[t][0][int(np.argmin(np.abs(GRID[t] - DEPLOYED[t])))] for t in MULTS)
tot_nom = sum(W[t] * nominal[t][0].max() for t in MULTS)
tot_rob = sum(W[t] * nominal[t][0][int(np.argmin(np.abs(GRID[t] - recommend[t])))] for t in MULTS)
print(f"[e39] weighted nominal EV: deployed {tot_dep:.4f} | nominal-opt {tot_nom:.4f} | robust {tot_rob:.4f}", flush=True)
print(f"[e39] RECOMMEND robust safety = {recommend}", flush=True)
print("[e39] DONE", flush=True)
