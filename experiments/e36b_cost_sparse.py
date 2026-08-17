"""E36b: cost heads on the SPARSE feature space (contrast to E36's dense meta path).

E36 refines the cost heads inside the 58-dim dense meta space.  Here the cost
predictor sees the full 16,414-dim hashed n-gram + dense features directly:

  S1  sparse ridge on log cost per model (already inside the ensemble at 25%
      via the linear row) -> raised to a dedicated cost source with weight w
  S2  sparse ridge on log OUTPUT tokens per model (the 87% part), input tokens
      reconstructed from a separate ridge, cost = rate_in*in + rate_out*out
  S3  sparse-to-dense stack: OOF sparse-ridge log-cost preds appended to the
      meta features, cost head refit (dense GBM sees the sparse signal)
  S4  sparse quantile: ridge on log cost with upper-tail target = log(cost) +
      alpha*|resid| (cheap tail-aware sparse variant)

Cost side only; the score side is the deployed configuration.  Same nested
CV + 880-bootstrap EV harness, extended safety grids as E36.
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
SPARSE_W = (0.25, 0.5, 1.0)
UNIT = float(policy.token_unit)
RATES = {m: (float(policy.models[m].input_token_rate), float(policy.models[m].output_token_rate)) for m in MODEL_IDS}

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
in_tok = np.array([[index[(e.episode_id, m)].input_tokens for m in MODEL_IDS] for e in episodes], dtype=float)
out_tok = np.array([[index[(e.episode_id, m)].output_tokens for m in MODEL_IDS] for e in episodes], dtype=float)
targets = np.hstack([true_s, np.log(true_c)])
delta_targets = np.column_stack([targets[:, 1] - targets[:, 0], targets[:, 2] - targets[:, 1]])
full_targets = np.hstack([targets, delta_targets])

print("[e36b] shared blocks", flush=True)
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


def ridge_fit(Xf, y):
    return Ridge(alpha=30.0, solver="sparse_cg").fit(Xf, y)


rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)
prod_all = np.zeros((n, 6))
meta_all = np.zeros((n, 8))
ord_score = np.zeros((n, 3))
rank_gain = np.zeros((n, 2))
s1_cost = np.zeros((n, 3))     # sparse ridge log cost (hold)
s2_cost = np.zeros((n, 3))     # sparse out-token ridge + in-token ridge reconstruction
s3_cost = np.zeros((n, 3))     # dense GBM cost head with sparse OOF preds stacked
s4_cost = np.zeros((n, 3))     # sparse tail-aware ridge

for fold in range(5):
    t0 = time.perf_counter()
    hold = fold_of == fold
    fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
    ridge = ridge_fit(X_sparse[fit_idx], targets[fit_idx])
    linear_hold = ridge.predict(X_sparse[hold_idx])
    linear_hold[:, :3] = np.clip(linear_hold[:, :3], 0.0, 1.0)
    inner_oof = np.zeros((len(fit_idx), 6))
    inner_fold = np.random.default_rng(fold).integers(0, 5, size=len(fit_idx))
    inner_cost_oof = np.zeros((len(fit_idx), 3))
    for inner in range(5):
        ih = inner_fold == inner
        m = ridge_fit(X_sparse[fit_idx[~ih]], targets[fit_idx[~ih]])
        inner_oof[ih] = m.predict(X_sparse[fit_idx[ih]])
        inner_cost_oof[ih] = inner_oof[ih, 3:6]
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
        m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(X_fit, full_targets[fit_idx][:, hidx])
        meta_all[hold_idx, hidx] = m.predict(X_hold)
    for mi in range(3):
        acc = np.zeros(len(hold_idx))
        for thr in ORD_T:
            y = (true_s[fit_idx, mi] >= thr).astype(int)
            if y.min() == y.max():
                acc += float(y.min()); continue
            acc += HistGradientBoostingClassifier(**GBM_PARAMS).fit(X_fit, y).predict_proba(X_hold)[:, 1]
        ord_score[hold_idx, mi] = acc / len(ORD_T)
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

    # ---- sparse cost variants
    # S1: sparse ridge log-cost (this is linear_hold[:,3:6] but kept explicit)
    s1_cost[hold_idx] = linear_hold[:, 3:6]
    # S2: sparse ridge on log1p(out_tok) and log1p(in_tok) -> reconstruct
    m_out = ridge_fit(X_sparse[fit_idx], np.log1p(out_tok[fit_idx]))
    m_in = ridge_fit(X_sparse[fit_idx], np.log1p(in_tok[fit_idx]))
    o_hat = np.expm1(np.maximum(m_out.predict(X_sparse[hold_idx]), 0.0))
    i_hat = np.expm1(np.maximum(m_in.predict(X_sparse[hold_idx]), 0.0))
    for mi, mid in enumerate(MODEL_IDS):
        s2_cost[hold_idx, mi] = np.log(np.maximum((RATES[mid][0] * i_hat[:, mi] + RATES[mid][1] * o_hat[:, mi]) / UNIT, 1e-9))
    # S3: dense GBM cost head with sparse OOF log-cost stacked (fit: inner OOF; hold: full-fit ridge)
    Xs_fit = np.hstack([X_fit, inner_cost_oof])
    Xs_hold = np.hstack([X_hold, linear_hold[:, 3:6]])
    for mi in range(3):
        m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(Xs_fit, np.log(true_c[fit_idx, mi]))
        s3_cost[hold_idx, mi] = m.predict(Xs_hold)
    # S4: sparse tail-aware ridge — target log cost + 0.5*|inner resid| (fold-pure)
    resid_abs = np.abs(np.log(true_c[fit_idx]) - inner_cost_oof)
    m_tail = ridge_fit(X_sparse[fit_idx], np.log(true_c[fit_idx]) + 0.5 * resid_abs)
    s4_cost[hold_idx] = m_tail.predict(X_sparse[hold_idx])

    prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
    fam_rows_hold = np.array([fam_mean[fam_names[i]] for i in hold_idx])
    prod = (1 - FAM_W) * prod + FAM_W * fam_rows_hold
    conf = np.clip(knn_hold[:, 6], 0.0, 1.0)[:, None] * CONF_SCALE
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
    prod_all[hold_idx] = prod
    print(f"[e36b] fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)

for mi, m in enumerate(MODEL_IDS):
    rmse = lambda pred: np.sqrt(np.mean((np.log(true_c[:, mi]) - pred) ** 2))
    print(f"[e36b] log-RMSE {m:12s}: dense-meta {rmse(meta_all[:, 3+mi]):.3f} | S1 sparse ridge {rmse(s1_cost[:, mi]):.3f} | S2 out+in recon {rmse(s2_cost[:, mi]):.3f} | S3 stack {rmse(s3_cost[:, mi]):.3f} | S4 tail {rmse(s4_cost[:, mi]):.3f}", flush=True)


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
GRIDS = {"fast": np.arange(0.90, 1.001, 0.01), "balanced": np.arange(0.80, 0.98, 0.01),
         "premium": np.arange(0.78, 0.98, 0.01)}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}

meta_score = meta_all[:, :6].copy()
meta_score[:, :3] = ord_score
d1 = (1 - RANK_BETA) * meta_all[:, 6] + RANK_BETA * rank_gain[:, 0]
d2 = (1 - RANK_BETA) * meta_all[:, 7] + RANK_BETA * rank_gain[:, 1]
recon = np.column_stack([meta_score[:, 0], meta_score[:, 0] + d1, meta_score[:, 0] + d1 + d2])
meta_score[:, :3] = (1 - GAIN_ALPHA) * meta_score[:, :3] + GAIN_ALPHA * recon


def evaluate(label, meta_logcost, sparse_logcost=None, w=0.0):
    """Final log cost = (1-w)*blend(prod, meta) + w*sparse_logcost (w=0 -> deployed path)."""
    meta = meta_score.copy()
    meta[:, 3:] = meta_logcost
    tot_ev = tot_sc = 0.0
    parts = []
    for tier, mult in MULTS.items():
        stacked = (1 - TIER_BLENDS[tier]) * prod_all + TIER_BLENDS[tier] * meta
        ps = np.clip(stacked[:, :3], 0, 1)
        lc = stacked[:, 3:]
        if sparse_logcost is not None and w > 0:
            lc = (1 - w) * lc + w * sparse_logcost
        pc = np.exp(np.clip(lc, -50, 50))
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
        parts.append(f"{tier} {best[0]:.4f}@{best[1]:.2f}")
    print(f"[e36b] {label}: weighted EV {tot_ev:.4f} score {tot_sc:.4f} | {'; '.join(parts)}", flush=True)
    return tot_ev


results = {}
results["deployed"] = evaluate("deployed (dense meta cost)", meta_all[:, 3:6])
for w in SPARSE_W:
    results[f"S1:w{w}"] = evaluate(f"S1 sparse ridge cost w={w}", meta_all[:, 3:6], s1_cost, w)
for w in SPARSE_W:
    results[f"S2:w{w}"] = evaluate(f"S2 sparse out+in recon w={w}", meta_all[:, 3:6], s2_cost, w)
results["S3"] = evaluate("S3 sparse->dense stacked cost head", s3_cost)
for w in (0.25, 0.5):
    results[f"S4:w{w}"] = evaluate(f"S4 sparse tail-aware ridge w={w}", meta_all[:, 3:6], s4_cost, w)
best = max(results.items(), key=lambda kv: kv[1])
print(f"[e36b] BEST {best[0]} EV {best[1]:.4f} vs deployed {results['deployed']:.4f} ({best[1]-results['deployed']:+.4f})", flush=True)
print("[e36b] DONE", flush=True)
