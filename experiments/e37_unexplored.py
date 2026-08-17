"""E37: three unexplored directions from the design-space map.

  MF  matrix factorization (collaborative filtering) over the episode x model
      score matrix.  Episode latent vector = ridge from prompt features onto
      fold-train item factors (so unseen prompts get a vector); model factors
      learned per fold.  MF score = <u_episode, v_model> + biases.  Blended
      into the meta score row at weight mu.  Different generalization axis
      from kNN (global low-rank structure vs local neighbors).
  ILP exact batch allocation via scipy.milp (HiGHS) vs our Lagrangian
      bisection, on identical predictions -> measures the relaxation gap.
      Not deployable as-is (stdlib), so this is a ceiling measurement.
  CP  conformal cost bound: per-model split-conformal upper quantile of the
      log-cost residual (fold-pure), used as allocation cost c_hat*exp(q_a)
      with coverage a in grid, replacing/augmenting the global safety ratio.

Score/cost side otherwise = deployed configuration.  Same nested CV +
880-bootstrap EV harness.
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
MF_RANK = 8
MF_MUS = (0.0, 0.15, 0.3, 0.5)
CP_ALPHAS = (0.5, 0.6, 0.7, 0.8)

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

print("[e37] shared blocks", flush=True)
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


def fit_mf(S, rank, iters=60, lam=0.5, seed=0):
    """ALS on a fully-observed (n_fit x 3) score matrix with biases:
    S ~ mu + b_i + b_j + U V^T.  Returns U, V, b_i, b_j, mu."""
    rng_ = np.random.default_rng(seed)
    n_i, n_j = S.shape
    mu = S.mean()
    b_i = S.mean(axis=1) - mu
    b_j = S.mean(axis=0) - mu
    U = rng_.normal(0, 0.1, (n_i, rank))
    V = rng_.normal(0, 0.1, (n_j, rank))
    I = np.eye(rank)
    for _ in range(iters):
        R = S - mu - b_i[:, None] - b_j[None, :]
        # U step
        A = V.T @ V + lam * I
        U = np.linalg.solve(A, (R @ V).T).T
        # V step
        A = U.T @ U + lam * I
        V = np.linalg.solve(A, (R.T @ U).T).T
        # biases
        b_i = (S - mu - b_j[None, :] - U @ V.T).mean(axis=1)
        b_j = (S - mu - b_i[:, None] - U @ V.T).mean(axis=0)
    return U, V, b_i, b_j, mu


rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)
prod_all = np.zeros((n, 6))
meta_all = np.zeros((n, 8))
ord_score = np.zeros((n, 3))
rank_gain = np.zeros((n, 2))
mf_score = np.zeros((n, 3))
cp_q = np.zeros((n, 3, len(CP_ALPHAS)))   # conformal upper log-resid quantiles broadcast

for fold in range(5):
    t0 = time.perf_counter()
    hold = fold_of == fold
    fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
    ridge = ridge_fit(X_sparse[fit_idx], targets[fit_idx])
    linear_hold = ridge.predict(X_sparse[hold_idx])
    linear_hold[:, :3] = np.clip(linear_hold[:, :3], 0.0, 1.0)
    inner_oof = np.zeros((len(fit_idx), 6))
    inner_fold = np.random.default_rng(fold).integers(0, 5, size=len(fit_idx))
    for inner in range(5):
        ih = inner_fold == inner
        m = ridge_fit(X_sparse[fit_idx[~ih]], targets[fit_idx[~ih]])
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

    # ---- MF: factorize fit-train score matrix, map prompts -> user factors via ridge on sparse feats
    U, V, b_i, b_j, mu = fit_mf(true_s[fit_idx], MF_RANK, seed=fold)
    # inner OOF for the factor regressor is unnecessary here because MF preds are only blended
    # into hold predictions (no stacking on fit); hold user factors come from a ridge fit on fit rows.
    fac_model = ridge_fit(X_sparse[fit_idx], np.hstack([U, b_i[:, None]]))
    fac_hold = fac_model.predict(X_sparse[hold_idx])
    U_hold, b_hold = fac_hold[:, :MF_RANK], fac_hold[:, MF_RANK]
    mf_score[hold_idx] = np.clip(mu + b_hold[:, None] + b_j[None, :] + U_hold @ V.T, 0.0, 1.0)

    # ---- CP: split-conformal upper quantiles of log-cost residual (inner OOF of the meta cost head)
    inner_cost = np.zeros((len(fit_idx), 3))
    for inner in range(5):
        ih = inner_fold == inner
        Xi_fit = X_fit[~ih]; Xi_hold = X_fit[ih]
        for mi in range(3):
            m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(Xi_fit, np.log(true_c[fit_idx[~ih], mi]))
            inner_cost[ih, mi] = m.predict(Xi_hold)
    resid = np.log(true_c[fit_idx]) - inner_cost
    for mi in range(3):
        for ai, a in enumerate(CP_ALPHAS):
            cp_q[hold_idx, mi, ai] = np.quantile(resid[:, mi], a)

    prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
    fam_rows_hold = np.array([fam_mean[fam_names[i]] for i in hold_idx])
    prod = (1 - FAM_W) * prod + FAM_W * fam_rows_hold
    conf = np.clip(knn_hold[:, 6], 0.0, 1.0)[:, None] * CONF_SCALE
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
    prod_all[hold_idx] = prod
    print(f"[e37] fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)

# MF diagnostics
for mi, m in enumerate(MODEL_IDS):
    r_mf = np.corrcoef(mf_score[:, mi], true_s[:, mi])[0, 1]
    r_ord = np.corrcoef(ord_score[:, mi], true_s[:, mi])[0, 1]
    r_cross = np.corrcoef(mf_score[:, mi], ord_score[:, mi])[0, 1]
    print(f"[e37] MF {m:12s}: corr(MF,true) {r_mf:.3f} | corr(ordinal,true) {r_ord:.3f} | corr(MF,ordinal) {r_cross:.3f}", flush=True)
print(f"[e37] CP quantiles (think) per alpha {CP_ALPHAS}: {np.round(cp_q[:, 2, :].mean(axis=0), 3).tolist()}", flush=True)


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


from scipy.optimize import Bounds, LinearConstraint, milp


def allocate_ilp(ps, pc, mult, safety):
    """Exact 0/1 assignment: max sum ps[i,k] x[i,k]  s.t. sum_k x[i,k]=1, sum pc[i,k]x[i,k] <= cap."""
    m_, k_ = ps.shape
    lt = pc[:, 0].sum(); cap = lt * max(1.0, mult * safety)
    c = -ps.ravel()
    A_eq = np.zeros((m_, m_ * k_))
    for i in range(m_):
        A_eq[i, i * k_:(i + 1) * k_] = 1.0
    A_bud = pc.ravel()[None, :]
    cons = [LinearConstraint(A_eq, 1, 1), LinearConstraint(A_bud, -np.inf, cap)]
    res = milp(c, constraints=cons, integrality=np.ones(m_ * k_), bounds=Bounds(0, 1),
               options={"time_limit": 20.0})
    if res.x is None:
        return allocate(ps, pc, mult, safety)
    return np.argmax(res.x.reshape(m_, k_), axis=1)


rng2 = np.random.default_rng(7)
samples = [rng2.integers(0, n, size=880) for _ in range(400)]
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
GRIDS = {"fast": np.arange(0.92, 1.0, 0.01), "balanced": np.arange(0.82, 0.94, 0.01),
         "premium": np.arange(0.80, 0.93, 0.01)}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}


def build_meta(mu):
    meta = meta_all[:, :6].copy()
    base = ord_score.copy()
    if mu > 0:
        base = (1 - mu) * base + mu * mf_score
    meta[:, :3] = base
    d1 = (1 - RANK_BETA) * meta_all[:, 6] + RANK_BETA * rank_gain[:, 0]
    d2 = (1 - RANK_BETA) * meta_all[:, 7] + RANK_BETA * rank_gain[:, 1]
    recon = np.column_stack([meta[:, 0], meta[:, 0] + d1, meta[:, 0] + d1 + d2])
    meta[:, :3] = (1 - GAIN_ALPHA) * meta[:, :3] + GAIN_ALPHA * recon
    return meta


def evaluate(label, meta, allocator=allocate, cost_mult=None, grids=GRIDS, n_samples=400):
    tot_ev = tot_sc = 0.0
    parts = []
    use = samples[:n_samples]
    for tier, mult in MULTS.items():
        stacked = (1 - TIER_BLENDS[tier]) * prod_all + TIER_BLENDS[tier] * meta
        ps = np.clip(stacked[:, :3], 0, 1)
        pc = np.exp(np.clip(stacked[:, 3:], -50, 50))
        if cost_mult is not None:
            pc = pc * cost_mult
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        best = None
        for s in grids[tier]:
            evs = []
            for sample in use:
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
    print(f"[e37] {label}: weighted EV {tot_ev:.4f} score {tot_sc:.4f} | {'; '.join(parts)}", flush=True)
    return tot_ev


results = {}
results["deployed"] = evaluate("deployed", build_meta(0.0))
# MF
for mu in MF_MUS[1:]:
    results[f"MF:mu{mu}"] = evaluate(f"MF rank{MF_RANK} mu={mu}", build_meta(mu))
# CP: conformal upper bound as allocation cost (safety grid widened upward since CP already inflates)
CP_GRIDS = {"fast": np.arange(0.92, 1.001, 0.01), "balanced": np.arange(0.82, 1.001, 0.02),
            "premium": np.arange(0.80, 1.001, 0.02)}
for ai, a in enumerate(CP_ALPHAS):
    results[f"CP:a{a}"] = evaluate(f"CP conformal alpha={a}", build_meta(0.0), cost_mult=np.exp(cp_q[:, :, ai]), grids=CP_GRIDS)
# ILP gap: exact vs Lagrangian on the deployed predictions, reduced bootstrap (ILP is slow)
t0 = time.perf_counter()
ev_lag_small = evaluate("Lagrangian (100 samples, for ILP comparison)", build_meta(0.0), n_samples=100)
ev_ilp_small = evaluate("ILP exact (100 samples)", build_meta(0.0), allocator=allocate_ilp, n_samples=100)
print(f"[e37] ILP relaxation gap: {ev_ilp_small - ev_lag_small:+.4f} (ILP wall {time.perf_counter()-t0:.0f}s)", flush=True)
results["ILP(100)"] = ev_ilp_small
best = max(((k, v) for k, v in results.items() if not k.startswith("ILP")), key=lambda kv: kv[1])
print(f"[e37] BEST {best[0]} EV {best[1]:.4f} vs deployed {results['deployed']:.4f} ({best[1]-results['deployed']:+.4f})", flush=True)
print("[e37] DONE", flush=True)
