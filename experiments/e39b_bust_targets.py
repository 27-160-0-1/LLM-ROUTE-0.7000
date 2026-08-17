"""E39b: what do you gain by ACCEPTING budget-bust risk?

For each tier, sweep the safety ratio upward until the nominal bust
probability reaches ~5% / 10% / 20%, and report at each target:
  - safety ratio, bust probability
  - score IF the batch passes (what a lucky run scores)
  - expected score EV = score * (1 - bust)   (what you get on average)
  - the same numbers under the worst stress scenario (longer-think for
    premium/balanced, small batch for fast) so the risk is not understated.
Then the weighted final for four policies: deployed / 5% / 10% / 20%.
Reuses the deployed OOF predictions (same CV as E39).
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
ORD_T = (0.25, 0.5, 0.75, 1.0); RANK_BETA = 0.25; LUT_NODES = 65
DEPLOYED = {"fast": 0.98, "balanced": 0.89, "premium": 0.88}
episodes = inputs.episodes; n = len(episodes); texts = [episode_text(e) for e in episodes]
index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}


def true_cost(eid, mid):
    o = index[(eid, mid)]; r = policy.models[mid]; unit = Decimal(policy.token_unit)
    return float(r.fixed_cost + Decimal(o.input_tokens) * r.input_token_rate / unit + Decimal(o.output_tokens) * r.output_token_rate / unit)


true_s = np.array([[float(index[(e.episode_id, m)].score) for m in MODEL_IDS] for e in episodes])
true_c = np.array([[true_cost(e.episode_id, m) for m in MODEL_IDS] for e in episodes])
targets = np.hstack([true_s, np.log(true_c)])
delta_targets = np.column_stack([targets[:, 1] - targets[:, 0], targets[:, 2] - targets[:, 1]])
full_targets = np.hstack([targets, delta_targets])
print("[e39b] shared blocks", flush=True)
from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
dense_rows, legacy_rows, fam_names = [], [], []; srows, scols, svals = [], [], []
for ri, episode in enumerate(episodes):
    dense = learned_router.raw_dense_features(episode); dense_rows.append(dense)
    items = learned_router.feature_items(episode, word_hash_bins=artifact.word_hash_bins, char_hash_bins=artifact.char_hash_bins,
                                         dense_mean=artifact.dense_mean, dense_scale=artifact.dense_scale, raw_dense=dense)
    for c, v in items.items():
        srows.append(ri); scols.append(c); svals.append(v)
    ls, lc = legacy_hash_regex.predict_episode(episode, legacy_artifact)
    legacy_rows.append([ls[m] for m in MODEL_IDS] + [math.log(lc[m]) for m in MODEL_IDS])
    fam_names.append(similarity.classify_family(texts[ri]))
dense_rows = np.asarray(dense_rows); legacy_rows = np.asarray(legacy_rows)
dim = len(learned_router.DENSE_FEATURE_NAMES) + artifact.word_hash_bins + artifact.char_hash_bins
X_sparse = sparse.csr_matrix((svals, (srows, scols)), shape=(n, dim))
FAMILIES = list(similarity.FAMILY_NAMES); fam_idx = np.array([FAMILIES.index(f) for f in fam_names])
fam_onehot = np.zeros((n, len(FAMILIES))); fam_onehot[np.arange(n), fam_idx] = 1.0
rng = np.random.default_rng(123); fold_of = rng.integers(0, 5, size=n)
prod_all = np.zeros((n, 6)); meta_all = np.zeros((n, 8)); ord_score = np.zeros((n, 3)); rank_gain = np.zeros((n, 2))
for fold in range(5):
    t0 = time.perf_counter(); hold = fold_of == fold
    fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
    ridge = Ridge(alpha=30.0, solver="sparse_cg").fit(X_sparse[fit_idx], targets[fit_idx])
    linear_hold = ridge.predict(X_sparse[hold_idx]); linear_hold[:, :3] = np.clip(linear_hold[:, :3], 0, 1)
    inner_oof = np.zeros((len(fit_idx), 6)); inner_fold = np.random.default_rng(fold).integers(0, 5, size=len(fit_idx))
    for inner in range(5):
        ih = inner_fold == inner
        inner_oof[ih] = Ridge(alpha=30.0, solver="sparse_cg").fit(X_sparse[fit_idx[~ih]], targets[fit_idx[~ih]]).predict(X_sparse[fit_idx[ih]])
    inner_oof[:, :3] = np.clip(inner_oof[:, :3], 0, 1)
    fit_texts = [texts[i] for i in fit_idx]; freqs, total = similarity.document_frequencies(fit_texts); idf = similarity.idf_table(freqs, total)
    fit_vectors = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS) for t in fit_texts]
    knn_index = similarity.KnnIndex(fit_vectors, targets[fit_idx].tolist()); gmean = targets[fit_idx].mean(axis=0)
    def kq(text, exclude=None):
        q = similarity.tfidf_vector(text, idf)
        if not q: return np.concatenate([gmean, [0.0]])
        scores = {}
        for g, v in q.items():
            for d, s in knn_index.postings.get(g, ()):
                if d == exclude: continue
                scores[d] = scores.get(d, 0.0) + v * s
        if not scores: return np.concatenate([gmean, [0.0]])
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:similarity.NEIGHBORS]
        tot = sum(s for _d, s in ranked); row = np.zeros(6)
        for d, s in ranked: row += (s / tot) * targets[fit_idx][d]
        return np.concatenate([row, [ranked[0][1]]])
    knn_fit = np.array([kq(t, exclude=i) for i, t in enumerate(fit_texts)]); knn_hold = np.array([kq(texts[i]) for i in hold_idx])
    fam_mean = {}; by_family = defaultdict(list)
    for i in fit_idx: by_family[fam_names[i]].append(targets[i])
    fglobal = targets[fit_idx].mean(axis=0)
    for name in FAMILIES:
        rows = by_family.get(name, []); fam_mean[name] = np.mean(rows, axis=0) if len(rows) >= 8 else fglobal
    X_fit = np.hstack([dense_rows[fit_idx], fam_onehot[fit_idx], legacy_rows[fit_idx], inner_oof, knn_fit])
    X_hold = np.hstack([dense_rows[hold_idx], fam_onehot[hold_idx], legacy_rows[hold_idx], linear_hold, knn_hold])
    for hidx in range(8):
        meta_all[hold_idx, hidx] = HistGradientBoostingRegressor(**GBM_PARAMS).fit(X_fit, full_targets[fit_idx][:, hidx]).predict(X_hold)
    for mi in range(3):
        acc = np.zeros(len(hold_idx))
        for thr in ORD_T:
            y = (true_s[fit_idx, mi] >= thr).astype(int)
            if y.min() == y.max(): acc += float(y.min()); continue
            acc += HistGradientBoostingClassifier(**GBM_PARAMS).fit(X_fit, y).predict_proba(X_hold)[:, 1]
        ord_score[hold_idx, mi] = acc / len(ORD_T)
    pc_hold = np.exp(np.clip(meta_all[hold_idx, 3:6], -50, 50)); grid = np.linspace(0, 1, LUT_NODES)
    for g, (a, b) in enumerate([(0, 1), (1, 2)]):
        ds = true_s[:, b] - true_s[:, a]; dc_raw = true_c[:, b] - true_c[:, a]
        floor = max(float(np.quantile(dc_raw[fit_idx], 0.05)), 1e-9); eff = ds / np.maximum(dc_raw, floor)
        r_fit = rankdata(eff[fit_idx], method="average") / max(len(fit_idx) - 1, 1); q = np.quantile(eff[fit_idx], grid)
        dchat = np.maximum(pc_hold[:, b] - pc_hold[:, a], floor)
        rank_gain[hold_idx, g] = np.interp(np.clip(HistGradientBoostingRegressor(**GBM_PARAMS).fit(X_fit, r_fit).predict(X_hold), 0, 1), grid, q) * dchat
    prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
    prod = (1 - FAM_W) * prod + FAM_W * np.array([fam_mean[fam_names[i]] for i in hold_idx])
    conf = np.clip(knn_hold[:, 6], 0, 1)[:, None] * CONF_SCALE
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]; prod[:, :3] = np.clip(prod[:, :3], 0, 1); prod_all[hold_idx] = prod
    print(f"[e39b] fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)

meta = meta_all[:, :6].copy(); meta[:, :3] = ord_score
d1 = (1 - RANK_BETA) * meta_all[:, 6] + RANK_BETA * rank_gain[:, 0]; d2 = (1 - RANK_BETA) * meta_all[:, 7] + RANK_BETA * rank_gain[:, 1]
recon = np.column_stack([meta[:, 0], meta[:, 0] + d1, meta[:, 0] + d1 + d2])
meta[:, :3] = (1 - GAIN_ALPHA) * meta[:, :3] + GAIN_ALPHA * recon
PS, PC = {}, {}
for tier in TIER_BLENDS:
    st = (1 - TIER_BLENDS[tier]) * prod_all + TIER_BLENDS[tier] * meta
    PS[tier] = np.clip(st[:, :3], 0, 1); pc = np.exp(np.clip(st[:, 3:], -50, 50))
    pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12)); pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12)); PC[tier] = pc


def allocate(ps, pc, mult, safety):
    lt = pc[:, 0].sum(); cap = lt * max(1.0, mult * safety)
    def choose(pen):
        u = ps - pen * pc / lt; pick = np.argmax(u + np.array([2e-12, 1e-12, 0.0]), axis=1)
        return pick, pc[np.arange(len(pick)), pick].sum()
    pick, tot = choose(0.0)
    if tot > cap:
        lo, hi = 0.0, 1.0; pick, tot = choose(hi)
        while tot > cap and hi < 2**60: lo, hi = hi, hi * 2; pick, tot = choose(hi)
        for _ in range(40):
            mid = (lo + hi) / 2; c2, t2 = choose(mid)
            if t2 <= cap: hi, pick, tot = mid, c2, t2
            else: lo = mid
    if tot > cap: pick = np.zeros(len(ps), dtype=int)
    return pick


MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}; W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
GRID = {"fast": np.arange(0.95, 1.10, 0.005), "balanced": np.arange(0.85, 1.05, 0.005), "premium": np.arange(0.84, 1.02, 0.005)}


def curve(tier, samples):
    ps, pc, mult = PS[tier], PC[tier], MULTS[tier]; out = []
    for s in GRID[tier]:
        b = 0; sc_l = []
        for sample in samples:
            p = allocate(ps[sample], pc[sample], mult, s); r = np.arange(len(sample))
            if true_c[sample][r, p].sum() / true_c[sample][:, 0].sum() > mult: b += 1
            sc_l.append(true_s[sample][r, p].mean())
        bust = b / len(samples); sc = float(np.mean(sc_l)); out.append((s, bust, sc, sc * (1 - bust)))
    return out


def make_samples(seed, size=880, weights=None, R=400):
    rng_ = np.random.default_rng(seed); p = None if weights is None else weights / weights.sum()
    return [rng_.choice(n, size=size, replace=True, p=p) for _ in range(R)]


think_q = np.quantile(true_c[:, 2], 0.8)
w_long = np.array([2.0 if true_c[i, 2] >= think_q else 1.0 for i in range(n)])
nominal = {t: curve(t, make_samples(7)) for t in MULTS}
stress = {"fast": curve("fast", make_samples(7, size=440)),
          "balanced": curve("balanced", make_samples(7, weights=w_long)),
          "premium": curve("premium", make_samples(7, weights=w_long))}
STRESS_NAME = {"fast": "small batch 440", "balanced": "longer-think", "premium": "longer-think"}


def at_bust(cv, target):
    """first grid point whose bust >= target (or last)."""
    for row in cv:
        if row[1] >= target: return row
    return cv[-1]


def at_s(cv, s):
    return min(cv, key=lambda r: abs(r[0] - s))


print("[e39b] === per-tier: deployed vs bust targets (nominal) ===", flush=True)
policies = {"deployed": {}, "5%": {}, "10%": {}, "20%": {}}
for tier in MULTS:
    dep = at_s(nominal[tier], DEPLOYED[tier]); policies["deployed"][tier] = dep
    print(f"[e39b] {tier:8s} deployed s={dep[0]:.3f} bust {dep[1]:.3f} score|pass {dep[2]:.4f} EV {dep[3]:.4f}", flush=True)
    for tgt, key in ((0.05, "5%"), (0.10, "10%"), (0.20, "20%")):
        row = at_bust(nominal[tier], tgt); policies[key][tier] = row
        st = at_s(stress[tier], row[0])
        print(f"[e39b] {tier:8s} target {key:>3s}: s={row[0]:.3f} bust {row[1]:.3f} score|pass {row[2]:.4f} EV {row[3]:.4f}  | under {STRESS_NAME[tier]}: bust {st[1]:.3f} EV {st[3]:.4f}", flush=True)
print("[e39b] === weighted final under each policy ===", flush=True)
for key, rows in policies.items():
    ev = sum(W[t] * rows[t][3] for t in MULTS); sc = sum(W[t] * rows[t][2] for t in MULTS)
    p_all_pass = np.prod([1 - rows[t][1] for t in MULTS])
    ev_stress = sum(W[t] * at_s(stress[t], rows[t][0])[3] for t in MULTS)
    print(f"[e39b] {key:>8s}: safety {[round(rows[t][0],3) for t in MULTS]} | 통과시 점수 {sc:.4f} | 기대점수(명목) {ev:.4f} | 3tier 모두 통과확률 {p_all_pass:.3f} | 기대점수(스트레스) {ev_stress:.4f}", flush=True)
print("[e39b] DONE", flush=True)
