"""E36: cost-prediction refinement, driven by diag_cost.py findings.

Findings: think cost is 87% output tokens; output length is uncorrelated
with prompt length (r=0.008) but anti-correlated with the model's own score
(r=-0.33); heavy tail (median 1.6k, max 130k tokens); exp(E[log c]) under-
predicts the batch total by 33% (pred/true 0.671); residual mass sits in
aime/hrmcr/dmmath/code (0.6-0.95) vs belebele/truthfulqa (0.36-0.41).

Three axes, alone and combined, on the deployed pipeline (ordinal + gain
alpha .5 + rank beta .25):

  T  tail-aware heads: quantile GBM (tau .5/.75/.9) on log cost per model;
     allocation cost = exp(q50 + kappa*(q90-q50)), kappa grid.  Unlike E32
     (symmetric |resid|), this learns the UPPER tail directly.
  F  family x model smearing: E[exp(resid)] per (family, model) from inner
     OOF residuals, replacing the implicit global safety absorption.
  P  score-conditioned cost: stack ordinal P(s>=.5) per model into the cost
     head features (hard problem -> long think output).

Safety grids extended upward (0.98->1.00 fast, 0.89->0.97 balanced,
0.88->0.97 premium) so a better cost model can convert into higher safety.
Same nested CV + 880-bootstrap EV harness; deployed reference printed first.
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
KAPPAS = (0.0, 0.25, 0.5, 1.0)

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

print("[e36] shared blocks", flush=True)
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
fam_idx = np.array([FAMILIES.index(f) for f in fam_names])
fam_onehot = np.zeros((n, len(FAMILIES)))
fam_onehot[np.arange(n), fam_idx] = 1.0

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)
prod_all = np.zeros((n, 6))
meta_all = np.zeros((n, 8))       # 6 heads + 2 gain
ord_score = np.zeros((n, 3))
rank_gain = np.zeros((n, 2))
q_cost = np.zeros((n, 3, 3))      # T: quantiles .5/.75/.9 of log cost
p_cost = np.zeros((n, 3))         # P: score-conditioned log cost head
smear_fm = np.ones((n, 3))        # F: per (family, model) E[exp(resid)] broadcast to hold rows
smear_glob = np.ones((n, 3))      # global Duan for reference

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

    # deployed heads
    for hidx in range(8):
        m = HistGradientBoostingRegressor(**GBM_PARAMS)
        m.fit(X_fit, full_targets[fit_idx][:, hidx])
        meta_all[hold_idx, hidx] = m.predict(X_hold)
    # ordinal + inner-OOF ordinal probs for P-axis stacking (fold-pure)
    ord_fit_inner = np.zeros((len(fit_idx), 3))
    for mi in range(3):
        acc = np.zeros(len(hold_idx))
        for thr in ORD_T:
            y = (true_s[fit_idx, mi] >= thr).astype(int)
            if y.min() == y.max():
                acc += float(y.min()); continue
            clf = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X_fit, y)
            acc += clf.predict_proba(X_hold)[:, 1]
        ord_score[hold_idx, mi] = acc / len(ORD_T)
        # inner OOF P(s>=.5) for stacking into cost head
        y5 = (true_s[fit_idx, mi] >= 0.5).astype(int)
        for inner in range(5):
            ih = inner_fold == inner
            if y5[~ih].min() == y5[~ih].max():
                ord_fit_inner[ih, mi] = y5[~ih].mean(); continue
            c = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X_fit[~ih], y5[~ih])
            ord_fit_inner[ih, mi] = c.predict_proba(X_fit[ih])[:, 1]
    # rank heads (deployed)
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

    # --- T: quantile heads on log cost
    for mi in range(3):
        y = np.log(true_c[fit_idx, mi])
        for qi, tau in enumerate((0.5, 0.75, 0.9)):
            m = HistGradientBoostingRegressor(loss="quantile", quantile=tau, **GBM_PARAMS).fit(X_fit, y)
            q_cost[hold_idx, mi, qi] = m.predict(X_hold)
    # --- P: score-conditioned cost head (features + inner-OOF ordinal P(s>=.5) x3)
    Xp_fit = np.hstack([X_fit, ord_fit_inner])
    ord_hold_p5 = np.zeros((len(hold_idx), 3))
    for mi in range(3):
        y5 = (true_s[fit_idx, mi] >= 0.5).astype(int)
        c = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X_fit, y5)
        ord_hold_p5[:, mi] = c.predict_proba(X_hold)[:, 1]
    Xp_hold = np.hstack([X_hold, ord_hold_p5])
    for mi in range(3):
        m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(Xp_fit, np.log(true_c[fit_idx, mi]))
        p_cost[hold_idx, mi] = m.predict(Xp_hold)
    # --- F: smearing from inner-OOF residuals of the deployed cost head, per (family, model)
    inner_cost = np.zeros((len(fit_idx), 3))
    for inner in range(5):
        ih = inner_fold == inner
        for mi in range(3):
            m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(X_fit[~ih], np.log(true_c[fit_idx[~ih], mi]))
            inner_cost[ih, mi] = m.predict(X_fit[ih])
    resid = np.log(true_c[fit_idx]) - inner_cost
    for mi in range(3):
        smear_glob[hold_idx, mi] = np.mean(np.exp(resid[:, mi]))
        for fi in range(len(FAMILIES)):
            mask_fit = fam_idx[fit_idx] == fi
            mask_hold = fam_idx[hold_idx] == fi
            if mask_fit.sum() >= 20:
                smear_fm[hold_idx[mask_hold], mi] = np.mean(np.exp(resid[mask_fit, mi]))
            else:
                smear_fm[hold_idx[mask_hold], mi] = smear_glob[hold_idx[0], mi]

    prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
    fam_rows_hold = np.array([fam_mean[fam_names[i]] for i in hold_idx])
    prod = (1 - FAM_W) * prod + FAM_W * fam_rows_hold
    conf = np.clip(knn_hold[:, 6], 0.0, 1.0)[:, None] * CONF_SCALE
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
    prod_all[hold_idx] = prod
    print(f"[e36] fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)

# --- diagnostics on the new cost predictors (batch-total calibration matters for budget)
rng_d = np.random.default_rng(7)
def total_ratio(pred_cost, mi):
    r = []
    for _ in range(200):
        s = rng_d.integers(0, n, size=880)
        r.append(pred_cost[s, mi].sum() / true_c[s, mi].sum())
    return np.mean(r), np.std(r)
base_cost = np.exp(meta_all[:, 3:6])
for mi, m in enumerate(MODEL_IDS):
    b = total_ratio(base_cost, mi)
    t50 = total_ratio(np.exp(q_cost[:, :, 0]), mi)
    t90 = total_ratio(np.exp(q_cost[:, :, 2]), mi)
    fs = total_ratio(base_cost * smear_fm, mi)
    gs = total_ratio(base_cost * smear_glob, mi)
    pp = total_ratio(np.exp(p_cost), mi)
    print(f"[e36] batch pred/true {m:12s}: base {b[0]:.3f} | q50 {t50[0]:.3f} q90 {t90[0]:.3f} | fam-smear {fs[0]:.3f} glob-smear {gs[0]:.3f} | score-cond {pp[0]:.3f}", flush=True)
    rmse = lambda pred: np.sqrt(np.mean((np.log(true_c[:, mi]) - pred) ** 2))
    print(f"[e36]   log-RMSE {m:12s}: base {rmse(meta_all[:, 3+mi]):.3f} | q50 {rmse(q_cost[:, mi, 0]):.3f} | score-cond {rmse(p_cost[:, mi]):.3f}", flush=True)


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

# deployed score side (fixed for all variants)
meta_score = meta_all[:, :6].copy()
meta_score[:, :3] = ord_score
d1 = (1 - RANK_BETA) * meta_all[:, 6] + RANK_BETA * rank_gain[:, 0]
d2 = (1 - RANK_BETA) * meta_all[:, 7] + RANK_BETA * rank_gain[:, 1]
recon = np.column_stack([meta_score[:, 0], meta_score[:, 0] + d1, meta_score[:, 0] + d1 + d2])
meta_score[:, :3] = (1 - GAIN_ALPHA) * meta_score[:, :3] + GAIN_ALPHA * recon


def evaluate(label, meta_logcost, post_mult=None):
    """meta_logcost: (n,3) log-cost from the meta side; post_mult: (n,3) multiplier applied
    to the final blended cost (smearing / tail inflation) -- allocation only, budget uses truth."""
    meta = meta_score.copy()
    meta[:, 3:] = meta_logcost
    tot_ev = tot_sc = 0.0
    parts = []
    for tier, mult in MULTS.items():
        stacked = (1 - TIER_BLENDS[tier]) * prod_all + TIER_BLENDS[tier] * meta
        ps = np.clip(stacked[:, :3], 0, 1)
        pc = np.exp(np.clip(stacked[:, 3:], -50, 50))
        if post_mult is not None:
            pc = pc * post_mult
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
    print(f"[e36] {label}: weighted EV {tot_ev:.4f} score {tot_sc:.4f} | {'; '.join(parts)}", flush=True)
    return tot_ev


results = {}
results["deployed"] = evaluate("deployed", meta_all[:, 3:6])
# T: tail-aware
for kappa in KAPPAS:
    lc = q_cost[:, :, 0] + kappa * (q_cost[:, :, 2] - q_cost[:, :, 0])
    results[f"T:k{kappa}"] = evaluate(f"T quantile q50+{kappa}*(q90-q50)", lc)
# F: smearing (allocation-side multiplier)
results["F:glob"] = evaluate("F global Duan smear", meta_all[:, 3:6], smear_glob)
results["F:fam"] = evaluate("F family x model smear", meta_all[:, 3:6], smear_fm)
# P: score-conditioned
results["P"] = evaluate("P score-conditioned cost head", p_cost)
# combos
results["P+F:fam"] = evaluate("P + family smear", p_cost, smear_fm)
best_k = max(KAPPAS, key=lambda k: results[f"T:k{k}"])
lc = q_cost[:, :, 0] + best_k * (q_cost[:, :, 2] - q_cost[:, :, 0])
results["T+F:fam"] = evaluate(f"T(k{best_k}) + family smear", lc, smear_fm)
results["T+P avg"] = evaluate(f"avg(T k{best_k}, P)", 0.5 * (lc + p_cost))
best = max(results.items(), key=lambda kv: kv[1])
print(f"[e36] BEST {best[0]} EV {best[1]:.4f} vs deployed {results['deployed']:.4f} ({best[1]-results['deployed']:+.4f})", flush=True)
print("[e36] DONE", flush=True)
