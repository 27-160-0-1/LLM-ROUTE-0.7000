"""E28: supervised token-presence features for the meta trees.

The meta GBM sees vocabulary only through ridge/kNN scalars, so it cannot
express "keyword X AND long prompt -> light fails" interactions.  Per outer
fold, select tokens by Welch t against d1 (=s1-s0) and s0 with a 5-split
stability filter (kept if in the top-K on >=4/5 inner subsets), append their
presence columns to the 58-dim meta input, refit the 8 heads.  K=0 is the
baseline consistency check.  Selection is fold-pure; token lists are dumped
for the first fold for manual inspection.
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
KS = (0, 24, 48, 96)
MIN_DF = 30
STABLE_MIN = 4

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
d1_all = delta_targets[:, 0]
s0_all = true_s[:, 0]

print("[e28] shared blocks", flush=True)
from scipy import sparse

token_sets = [frozenset(learned_router._normalized_tokens(t)) for t in texts]
df_global = defaultdict(int)
for ts in token_sets:
    for tok in ts:
        df_global[tok] += 1
vocab = sorted(tok for tok, c in df_global.items() if c >= 20)
V = len(vocab)
tok_col = {tok: j for j, tok in enumerate(vocab)}
P = np.zeros((n, V), dtype=np.float64)
for i, ts in enumerate(token_sets):
    for tok in ts:
        j = tok_col.get(tok)
        if j is not None:
            P[i, j] = 1.0
print(f"[e28] candidate vocab {V} tokens (global df>=20)", flush=True)


def welch_t(Pm, y):
    n_rows = Pm.shape[0]
    n1 = Pm.sum(axis=0)
    n0 = n_rows - n1
    s1 = Pm.T @ y
    q1 = Pm.T @ (y * y)
    ytot, qtot = y.sum(), (y * y).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = s1 / n1
        m0 = (ytot - s1) / n0
        v1 = np.maximum(q1 / n1 - m1 * m1, 0.0) * n1 / np.maximum(n1 - 1, 1)
        v0 = np.maximum((qtot - q1) / n0 - m0 * m0, 0.0) * n0 / np.maximum(n0 - 1, 1)
        t = (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0 + 1e-12)
    t[(n1 < 5) | (n0 < 5)] = 0.0
    return np.nan_to_num(t)


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

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)
prod_all = np.zeros((n, 6))
meta_by_k = {k: np.zeros((n, 8)) for k in KS}

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

    df_fit = P[fit_idx].sum(axis=0)
    eligible = df_fit >= MIN_DF
    sel_by_k = {}
    for K in KS:
        if K == 0:
            sel_by_k[K] = np.array([], dtype=int)
            continue
        count_d1 = np.zeros(V, dtype=int)
        count_s0 = np.zeros(V, dtype=int)
        for inner in range(5):
            sub = fit_idx[inner_fold != inner]
            t_d1 = np.abs(welch_t(P[sub], d1_all[sub]))
            t_s0 = np.abs(welch_t(P[sub], s0_all[sub]))
            t_d1[~eligible] = 0.0
            t_s0[~eligible] = 0.0
            count_d1[np.argsort(t_d1)[-K:]] += 1
            count_s0[np.argsort(t_s0)[-K:]] += 1
        sel = np.where((count_d1 >= STABLE_MIN) | (count_s0 >= STABLE_MIN))[0]
        sel_by_k[K] = sel
        if fold == 0:
            names = [vocab[j] for j in sel[:60]]
            print(f"[e28] fold0 K={K}: {len(sel)} stable tokens: {names}", flush=True)
    for K in KS:
        sel = sel_by_k[K]
        if len(sel):
            Xf = np.hstack([X_fit, P[fit_idx][:, sel]])
            Xh = np.hstack([X_hold, P[hold_idx][:, sel]])
        else:
            Xf, Xh = X_fit, X_hold
        for hidx in range(8):
            m = HistGradientBoostingRegressor(**GBM_PARAMS)
            m.fit(Xf, full_targets[fit_idx][:, hidx])
            meta_by_k[K][hold_idx, hidx] = m.predict(Xh)
    prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
    fam_rows_hold = np.array([fam_mean[fam_names[i]] for i in hold_idx])
    prod = (1 - FAM_W) * prod + FAM_W * fam_rows_hold
    conf = np.clip(knn_hold[:, 6], 0.0, 1.0)[:, None] * CONF_SCALE
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
    prod_all[hold_idx] = prod
    print(f"[e28] fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)


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


rng2 = np.random.default_rng(7)
samples = [rng2.integers(0, n, size=880) for _ in range(400)]
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
GRIDS = {"fast": np.arange(0.92, 1.0, 0.01), "balanced": np.arange(0.82, 0.94, 0.01),
         "premium": np.arange(0.80, 0.93, 0.01)}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}

for K in KS:
    meta_all = meta_by_k[K]
    meta = meta_all[:, :6].copy()
    recon = np.column_stack([meta[:, 0], meta[:, 0] + meta_all[:, 6],
                             meta[:, 0] + meta_all[:, 6] + meta_all[:, 7]])
    meta[:, :3] = (1 - GAIN_ALPHA) * meta[:, :3] + GAIN_ALPHA * recon
    tot_ev = tot_sc = 0.0
    per_tier = []
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
                p = allocate(ps[sample], pc[sample], mult, s)
                r = np.arange(len(sample))
                ratio = true_c[sample][r, p].sum() / true_c[sample][:, 0].sum()
                evs.append(0.0 if ratio > mult else true_s[sample][r, p].mean())
            ev = float(np.mean(evs))
            if best is None or ev > best[0]:
                pick = allocate(ps, pc, mult, s)
                best = (ev, s, float(true_s[np.arange(n), pick].mean()))
        tot_ev += W[tier] * best[0]; tot_sc += W[tier] * best[2]
        per_tier.append(f"{tier} {best[0]:.4f}@{best[1]:.2f}")
    print(f"[e28] K={K}: weighted EV {tot_ev:.4f} score {tot_sc:.4f} | {'; '.join(per_tier)}", flush=True)
print("[e28] DONE", flush=True)
