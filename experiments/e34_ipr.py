"""E34: IPR-style structure transplanted onto the deployed pipeline.

Source: Sivasubramanian et al., "IPR: Intelligent Prompt Routing with
User-Controlled Quality-Cost Trade-offs" (arXiv 2509.06274).  Three ideas
that survive our constraints (stdlib runtime, 2,640 labels):

  (A) threshold selection: pick the CHEAPEST model whose predicted quality
      >= theta, theta re-solved per batch so realized cost fits the budget
      (IPR Sec. 2.3/2.4) -- replaces the Lagrangian utility allocator.
  (B) pairwise heads: P(s1 > s0), P(s2 > s1) classifiers so cross-model
      ORDERING is trained directly (IPR Sec. 2.2 ranking loss); folded into
      the gain reconstruction as expected-sign-weighted deltas.
  (C) per-head temperature scaling of the ordinal probabilities, fitted on
      inner-OOF logits (IPR Sec. 3 calibration ablation).

Everything is evaluated with the same nested-CV + 880-bootstrap EV harness
as the deployed configuration so numbers are comparable to E27's 0.6982.
"""

import math
import sys
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from ossp_router import learned_router, legacy_hash_regex, similarity
from ossp_router.heuristic import episode_text
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes

similarity.NEIGHBORS = 16

policy = load_bundled_policy()
inputs = load_input(HERE / "data/combined/inputs.json")
outcomes = load_outcomes(HERE / "data/combined/outcomes.json")
artifact = learned_router.load_artifact(HERE / "src/ossp_router/resources/learned-router.v1.json")
legacy_artifact = legacy_hash_regex.load_artifact(HERE / "src/ossp_router/resources/hash-regex-public.v1.json")

LEGACY_W, FAM_W, CONF_SCALE, GAIN_ALPHA = 0.75, 0.3, 0.4, 0.5
TIER_BLENDS = {"fast": 0.6, "balanced": 0.3, "premium": 0.45}
GBM_PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=30,
                  l2_regularization=3.0, early_stopping=True, validation_fraction=0.15,
                  random_state=11)
ORD_T = (0.25, 0.5, 0.75, 1.0)
RANK_BETA = 0.25
LUT_NODES = 65
PAIR_GAMMAS = (0.0, 0.25, 0.5)

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

print("[e34] shared blocks", flush=True)
from scipy import sparse
from scipy.stats import rankdata

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
fam_onehot = np.zeros((n, len(FAMILIES)))
for i, name in enumerate(fam_names):
    fam_onehot[i, FAMILIES.index(name)] = 1.0

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def fit_temperature(logits, y):
    """1-D temperature by grid on NLL (IPR: per-head temperature scaling)."""
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.5, 3.0, 26):
        p = np.clip(sigmoid(logits / t), 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = t, nll
    return best_t


rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)
prod_all = np.zeros((n, 6))
meta_all = np.zeros((n, 8))
ord_raw = np.zeros((n, 3, len(ORD_T)))     # ordinal logits
ord_temp = np.ones((n, 3, len(ORD_T)))     # per-fold temperatures broadcast to rows
rank_gain = np.zeros((n, 2))
pair_prob = np.zeros((n, 2))               # P(s_{g+1} > s_g)
pair_mag = np.zeros((n, 2))                # E[|delta| | delta != 0]

for fold in range(5):
    t0 = time.perf_counter()
    hold = fold_of == fold
    fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
    ridge = Ridge(alpha=30.0, solver="sparse_cg")
    ridge.fit(X_sparse[fit_idx], targets[fit_idx])
    linear_hold = ridge.predict(X_sparse[hold_idx])
    linear_hold[:, :3] = np.clip(linear_hold[:, :3], 0.0, 1.0)
    inner_oof = np.zeros((len(fit_idx), 6))
    inner_fold = np.random.default_rng(fold).integers(0, 5, size=len(fit_idx))
    for inner in range(5):
        ih = inner_fold == inner
        m = Ridge(alpha=30.0, solver="sparse_cg")
        m.fit(X_sparse[fit_idx[~ih]], targets[fit_idx[~ih]])
        inner_oof[ih] = m.predict(X_sparse[fit_idx[ih]])
    inner_oof[:, :3] = np.clip(inner_oof[:, :3], 0.0, 1.0)
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
        m = HistGradientBoostingRegressor(**GBM_PARAMS)
        m.fit(X_fit, full_targets[fit_idx][:, hidx])
        meta_all[hold_idx, hidx] = m.predict(X_hold)

    # ordinal heads + (C) temperature from an inner split of the fit set
    cal_mask = np.random.default_rng(100 + fold).random(len(fit_idx)) < 0.2
    for mi in range(3):
        for ti, thr in enumerate(ORD_T):
            y = (true_s[fit_idx, mi] >= thr).astype(int)
            if y.min() == y.max():
                p = min(max(float(y.min()), 1e-6), 1 - 1e-6)
                ord_raw[hold_idx, mi, ti] = math.log(p / (1 - p))
                continue
            clf = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X_fit, y)
            ord_raw[hold_idx, mi, ti] = clf.decision_function(X_hold)
            if cal_mask.sum() > 30 and y[cal_mask].min() != y[cal_mask].max():
                clf_c = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X_fit[~cal_mask], y[~cal_mask])
                temp = fit_temperature(clf_c.decision_function(X_fit[cal_mask]), y[cal_mask])
                ord_temp[hold_idx, mi, ti] = temp

    # E27 rank heads (deployed)
    pc_hold = np.exp(np.clip(meta_all[hold_idx, 3:6], -50.0, 50.0))
    grid = np.linspace(0.0, 1.0, LUT_NODES)
    for g, (a, b) in enumerate([(0, 1), (1, 2)]):
        ds = true_s[:, b] - true_s[:, a]
        dc_raw = true_c[:, b] - true_c[:, a]
        floor = max(float(np.quantile(dc_raw[fit_idx], 0.05)), 1e-9)
        eff = ds / np.maximum(dc_raw, floor)
        r_fit = rankdata(eff[fit_idx], method="average") / max(len(fit_idx) - 1, 1)
        q = np.quantile(eff[fit_idx], grid)
        dchat = np.maximum(pc_hold[:, b] - pc_hold[:, a], floor)
        m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(X_fit, r_fit)
        rank_gain[hold_idx, g] = np.interp(np.clip(m.predict(X_hold), 0.0, 1.0), grid, q) * dchat
        # (B) pairwise heads
        d = ds[fit_idx]
        y_pos = (d > 0).astype(int)
        clf = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X_fit, y_pos)
        pair_prob[hold_idx, g] = clf.predict_proba(X_hold)[:, 1]
        nz = d != 0
        if nz.sum() >= 50:
            reg = HistGradientBoostingRegressor(**GBM_PARAMS).fit(X_fit[nz], np.abs(d[nz]))
            pair_mag[hold_idx, g] = np.maximum(reg.predict(X_hold), 0.0)
        else:
            pair_mag[hold_idx, g] = np.abs(d[nz]).mean() if nz.any() else 0.0

    prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
    fam_rows_hold = np.array([fam_mean[fam_names[i]] for i in hold_idx])
    prod = (1 - FAM_W) * prod + FAM_W * fam_rows_hold
    conf = np.clip(knn_hold[:, 6], 0.0, 1.0)[:, None] * CONF_SCALE
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
    prod_all[hold_idx] = prod
    print(f"[e34] fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------- allocators
def allocate_lagrange(ps, pc, mult, safety):
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


def allocate_threshold(ps, pc, mult, safety):
    """IPR Sec 2.3: cheapest model with quality >= theta; theta bisected so
    the batch fits the (safety-scaled) budget.  Fallback argmax when none
    qualifies.  Higher theta -> more expensive."""
    lt = pc[:, 0].sum(); cap = lt * max(1.0, mult * safety)
    ps_ord = ps.copy()  # enforce monotone quality for cheapest-first semantics
    ps_ord[:, 1] = np.maximum(ps_ord[:, 1], ps_ord[:, 0])
    ps_ord[:, 2] = np.maximum(ps_ord[:, 2], ps_ord[:, 1])
    argmax_pick = np.argmax(ps + np.array([2e-12, 1e-12, 0.0]), axis=1)

    def choose(theta):
        ok = ps_ord >= theta
        pick = np.where(ok[:, 0], 0, np.where(ok[:, 1], 1, np.where(ok[:, 2], 2, argmax_pick)))
        return pick, pc[np.arange(len(pick)), pick].sum()

    # theta in [0,1]; cost is nondecreasing in theta -> bisect the largest feasible theta
    lo, hi = 0.0, 1.0
    pick_lo, tot_lo = choose(lo)
    if tot_lo > cap:
        return np.zeros(len(ps), dtype=int)
    best = pick_lo
    for _ in range(40):
        mid = (lo + hi) / 2
        p, t = choose(mid)
        if t <= cap:
            lo, best = mid, p
        else:
            hi = mid
    return best


def allocate_hybrid(ps, pc, mult, safety):
    """Threshold pass first (IPR), then spend leftover budget by Lagrangian
    marginal utility among the remaining upgrades."""
    lt = pc[:, 0].sum(); cap = lt * max(1.0, mult * safety)
    base = allocate_threshold(ps, pc, mult, safety * 0.9)  # reserve 10% headroom
    spent = pc[np.arange(len(base)), base].sum()
    if spent >= cap:
        return base
    # candidate single-step upgrades from current pick
    nxt = np.minimum(base + 1, 2)
    gain = ps[np.arange(len(ps)), nxt] - ps[np.arange(len(ps)), base]
    dcost = pc[np.arange(len(ps)), nxt] - pc[np.arange(len(ps)), base]
    eff = np.where((dcost > 0) & (nxt > base), gain / np.maximum(dcost, 1e-12), -np.inf)
    order = np.argsort(-eff)
    pick = base.copy()
    for i in order:
        if eff[i] <= 0:
            break
        if spent + dcost[i] <= cap:
            pick[i] = nxt[i]; spent += dcost[i]
    return pick


ALLOCATORS = {"lagrange": allocate_lagrange, "threshold": allocate_threshold, "hybrid": allocate_hybrid}

rng2 = np.random.default_rng(7)
samples = [rng2.integers(0, n, size=880) for _ in range(400)]
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
GRIDS = {"fast": np.arange(0.92, 1.0, 0.01), "balanced": np.arange(0.82, 0.94, 0.01),
         "premium": np.arange(0.80, 0.93, 0.01)}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}


def build_meta(use_temp, gamma):
    logits = ord_raw / (ord_temp if use_temp else 1.0)
    ord_scores = sigmoid(logits).mean(axis=2)  # (n,3)
    meta = meta_all[:, :6].copy()
    meta[:, :3] = ord_scores
    d1 = (1 - RANK_BETA) * meta_all[:, 6] + RANK_BETA * rank_gain[:, 0]
    d2 = (1 - RANK_BETA) * meta_all[:, 7] + RANK_BETA * rank_gain[:, 1]
    if gamma > 0:
        # (B) expected signed delta from pairwise heads: (2P-1)*E|d|
        pd1 = (2 * pair_prob[:, 0] - 1) * pair_mag[:, 0]
        pd2 = (2 * pair_prob[:, 1] - 1) * pair_mag[:, 1]
        d1 = (1 - gamma) * d1 + gamma * pd1
        d2 = (1 - gamma) * d2 + gamma * pd2
    recon = np.column_stack([meta[:, 0], meta[:, 0] + d1, meta[:, 0] + d1 + d2])
    meta[:, :3] = (1 - GAIN_ALPHA) * meta[:, :3] + GAIN_ALPHA * recon
    return meta


def evaluate(label, meta, allocator):
    tot_ev = tot_sc = 0.0
    parts = []
    for tier, mult in MULTS.items():
        stacked = (1 - TIER_BLENDS[tier]) * prod_all + TIER_BLENDS[tier] * meta
        ps = np.clip(stacked[:, :3], 0, 1)
        pc = np.exp(np.clip(stacked[:, 3:], -50, 50))
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        best = None
        for s in GRIDS[tier]:
            evs = []
            for sample in samples:
                p = allocator(ps[sample], pc[sample], mult, s)
                r = np.arange(len(sample))
                ratio = true_c[sample][r, p].sum() / true_c[sample][:, 0].sum()
                evs.append(0.0 if ratio > mult else true_s[sample][r, p].mean())
            ev = float(np.mean(evs))
            if best is None or ev > best[0]:
                pick = allocator(ps, pc, mult, s)
                best = (ev, s, float(true_s[np.arange(n), pick].mean()))
        tot_ev += W[tier] * best[0]; tot_sc += W[tier] * best[2]
        parts.append(f"{tier} {best[0]:.4f}@{best[1]:.2f}")
    print(f"[e34] {label}: weighted EV {tot_ev:.4f} score {tot_sc:.4f} | {'; '.join(parts)}", flush=True)
    return tot_ev


print(f"[e34] mean temperatures per (model,thr): {np.round(ord_temp.mean(axis=0), 2).tolist()}", flush=True)
results = {}
# deployed reference
results["deployed"] = evaluate("deployed(lagrange,ord,rank0.25)", build_meta(False, 0.0), allocate_lagrange)
# (A) allocators on deployed predictions
for name in ("threshold", "hybrid"):
    results[f"A:{name}"] = evaluate(f"A:{name} allocator", build_meta(False, 0.0), ALLOCATORS[name])
# (C) temperature scaling
results["C:temp"] = evaluate("C:temperature-scaled ordinal", build_meta(True, 0.0), allocate_lagrange)
# (B) pairwise heads
for gamma in PAIR_GAMMAS[1:]:
    results[f"B:gamma{gamma}"] = evaluate(f"B:pairwise gamma={gamma}", build_meta(False, gamma), allocate_lagrange)
# combos
results["B+C:gamma0.25"] = evaluate("B+C pairwise0.25 + temp", build_meta(True, 0.25), allocate_lagrange)
results["A+B+C:hybrid"] = evaluate("A+B+C hybrid + pairwise0.25 + temp", build_meta(True, 0.25), allocate_hybrid)
best = max(results.items(), key=lambda kv: kv[1])
print(f"[e34] BEST {best[0]} EV {best[1]:.4f} vs deployed {results['deployed']:.4f} ({best[1]-results['deployed']:+.4f})", flush=True)
print("[e34] DONE", flush=True)
