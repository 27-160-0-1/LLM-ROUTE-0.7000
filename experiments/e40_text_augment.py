"""E40: label-preserving text augmentation of the public training set.

Question: if every fold-train episode is augmented with K perturbed copies
(word dropout / span deletion / cropping / char typos) that inherit the
parent's score+cost labels, how far does the deployed pipeline's CV EV move?

Augmentation is applied *only* to fold-train rows (never hold-out), and copies
are grouped with their parent for the inner OOF and kNN self-exclusion so no
copy leaks its parent's label into a prediction for that parent.

Usage: python e40_text_augment.py K WHERE [STRENGTH] [SEED]
  K        copies per episode (0 = baseline)
  WHERE    ridge | ridge_gbm | all   (all = ridge + gbm + knn index)
  STRENGTH light | mid | heavy       (default mid)
  SEED     bootstrap seed (default 7)
Same nested CV + 880x400 bootstrap harness as E27/E39, beta=0.25 rank head.
"""

import math
import random
import re
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
from ossp_router.protocol import MODEL_IDS, Episode, load_bundled_policy, load_input, load_outcomes

K = int(sys.argv[1]) if len(sys.argv) > 1 else 0
WHERE = sys.argv[2] if len(sys.argv) > 2 else "ridge"
STRENGTH = sys.argv[3] if len(sys.argv) > 3 else "mid"
BOOT_SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
TAG = f"[e40 K={K} {WHERE} {STRENGTH} s{BOOT_SEED}]"
AUG_RIDGE = K > 0
AUG_GBM = K > 0 and WHERE in ("ridge_gbm", "all")
AUG_KNN = K > 0 and WHERE == "all"

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
RANK_BETA = 0.25
LUT_NODES = 65

STRENGTHS = {
    "light": dict(wdrop=0.05, spans=(1, 2), span_len=(2, 5), crop=(0.85, 1.0), typo=0.005),
    "mid": dict(wdrop=0.10, spans=(1, 3), span_len=(3, 8), crop=(0.75, 0.95), typo=0.01),
    "heavy": dict(wdrop=0.20, spans=(2, 5), span_len=(4, 12), crop=(0.6, 0.9), typo=0.02),
}
S = STRENGTHS[STRENGTH]
_TOK = re.compile(r"\S+|\s+")


def augment(text, rng):
    """One random label-preserving perturbation composed of 1-2 ops."""
    parts = _TOK.findall(text)
    words = [i for i, p in enumerate(parts) if not p.isspace()]
    if len(words) < 8:
        return text
    ops = rng.sample(["wdrop", "spandel", "crop", "typo"], k=rng.choice([1, 1, 2]))
    keep = set(range(len(parts)))
    if "wdrop" in ops:
        for i in words:
            if rng.random() < S["wdrop"]:
                keep.discard(i)
    if "spandel" in ops:
        for _ in range(rng.randint(*S["spans"])):
            L = rng.randint(*S["span_len"])
            if len(words) > L + 1:
                st = rng.randrange(0, len(words) - L)
                for i in words[st:st + L]:
                    keep.discard(i)
    if "crop" in ops:
        frac = rng.uniform(*S["crop"])
        L = max(8, int(len(words) * frac))
        st = rng.randrange(0, len(words) - L + 1)
        lo, hi = words[st], words[st + L - 1]
        keep = {i for i in keep if lo <= i <= hi}
    out = "".join(p for i, p in enumerate(parts) if i in keep)
    if "typo" in ops:
        chars = list(out)
        for i in range(len(chars) - 1):
            if chars[i].isalnum() and rng.random() < S["typo"]:
                r = rng.random()
                if r < 0.4:
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
                elif r < 0.7:
                    chars[i] = ""
                else:
                    chars[i] = chars[i] * 2
        out = "".join(chars)
    return out if out.strip() else text


episodes = list(inputs.episodes)
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
targets = np.hstack([true_s, np.log(true_c)])
delta_targets = np.column_stack([targets[:, 1] - targets[:, 0], targets[:, 2] - targets[:, 1]])
full_targets = np.hstack([targets, delta_targets])

# ---- build the row pool: originals (0..n-1) followed by K copies of each ----
print(f"{TAG} building augmented pool", flush=True)
aug_rng = random.Random(4040)
pool_eps = list(episodes)
parent = list(range(n))
for k in range(K):
    for i, e in enumerate(episodes):
        pool_eps.append(Episode(episode_id=f"{e.episode_id}#aug{k}", prompt=augment(episode_text(e), aug_rng)))
        parent.append(i)
parent = np.asarray(parent)
N = len(pool_eps)
texts = [episode_text(e) for e in pool_eps]

from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

t0 = time.perf_counter()
dense_rows, legacy_rows, fam_names = [], [], []
srows, scols, svals = [], [], []
for ri, episode in enumerate(pool_eps):
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
X_sparse = sparse.csr_matrix((svals, (srows, scols)), shape=(N, dim))
FAMILIES = list(similarity.FAMILY_NAMES)
fam_onehot = np.zeros((N, len(FAMILIES)))
for i, name in enumerate(fam_names):
    fam_onehot[i, FAMILIES.index(name)] = 1.0
T_pool = targets[parent]           # labels inherited from parent
FT_pool = full_targets[parent]
print(f"{TAG} features for {N} rows in {time.perf_counter()-t0:.0f}s", flush=True)

rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)          # per ORIGINAL episode (identical to E27)
prod_all = np.zeros((n, 6))
meta_all = np.zeros((n, 8))
rank_gain = np.zeros((n, 2))

for fold in range(5):
    t0 = time.perf_counter()
    hold = fold_of == fold
    fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
    fit_set = set(fit_idx.tolist())
    # pool rows whose parent is in fit: originals + (optionally) copies
    fit_pool = np.array([r for r in range(N) if parent[r] in fit_set and (r < n or AUG_RIDGE)])
    ridge = Ridge(alpha=30.0, solver="sparse_cg")
    ridge.fit(X_sparse[fit_pool], T_pool[fit_pool])
    linear_hold = ridge.predict(X_sparse[hold_idx])
    linear_hold[:, :3] = np.clip(linear_hold[:, :3], 0.0, 1.0)

    # GBM training rows: originals of fit (+ copies if AUG_GBM)
    gbm_rows = np.array([r for r in range(N) if parent[r] in fit_set and (r < n or AUG_GBM)])
    # inner OOF: inner fold assigned per original (same RNG as E27), copies inherit
    inner_fold_orig = np.random.default_rng(fold).integers(0, 5, size=len(fit_idx))
    inner_of_parent = {int(p): int(f) for p, f in zip(fit_idx, inner_fold_orig)}
    inner_fold_pool = np.array([inner_of_parent[int(parent[r])] for r in fit_pool])
    inner_fold_gbm = np.array([inner_of_parent[int(parent[r])] for r in gbm_rows])
    inner_oof = np.zeros((len(gbm_rows), 6))
    for inner in range(5):
        tr = fit_pool[inner_fold_pool != inner]
        te_mask = inner_fold_gbm == inner
        m = Ridge(alpha=30.0, solver="sparse_cg")
        m.fit(X_sparse[tr], T_pool[tr])
        inner_oof[te_mask] = m.predict(X_sparse[gbm_rows[te_mask]])
    inner_oof[:, :3] = np.clip(inner_oof[:, :3], 0.0, 1.0)

    # kNN index: originals of fit (+ copies if AUG_KNN); exclude parent group on self-query
    knn_rows = np.array([r for r in range(N) if parent[r] in fit_set and (r < n or AUG_KNN)])
    knn_texts = [texts[r] for r in knn_rows]
    freqs, total = similarity.document_frequencies(knn_texts)
    idf = similarity.idf_table(freqs, total)
    knn_vectors = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS) for t in knn_texts]
    knn_index = similarity.KnnIndex(knn_vectors, T_pool[knn_rows].tolist())
    knn_parent = parent[knn_rows]
    gmean = T_pool[knn_rows].mean(axis=0)

    def kq(text, exclude_parent=None):
        q = similarity.tfidf_vector(text, idf)
        if not q:
            return np.concatenate([gmean, [0.0]])
        scores = {}
        for g, v in q.items():
            for d, s in knn_index.postings.get(g, ()):
                if exclude_parent is not None and knn_parent[d] == exclude_parent:
                    continue
                scores[d] = scores.get(d, 0.0) + v * s
        if not scores:
            return np.concatenate([gmean, [0.0]])
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:similarity.NEIGHBORS]
        tot = sum(s for _d, s in ranked)
        row = np.zeros(6)
        for d, s in ranked:
            row += (s / tot) * T_pool[knn_rows[d]]
        return np.concatenate([row, [ranked[0][1]]])

    knn_fit = np.array([kq(texts[r], exclude_parent=int(parent[r])) for r in gbm_rows])
    knn_hold = np.array([kq(texts[i]) for i in hold_idx])
    fam_mean = {}
    by_family = defaultdict(list)
    for i in fit_idx:
        by_family[fam_names[i]].append(targets[i])
    fglobal = targets[fit_idx].mean(axis=0)
    for name in FAMILIES:
        rows = by_family.get(name, [])
        fam_mean[name] = np.mean(rows, axis=0) if len(rows) >= 8 else fglobal
    X_fit = np.hstack([dense_rows[gbm_rows], fam_onehot[gbm_rows], legacy_rows[gbm_rows], inner_oof, knn_fit])
    X_hold = np.hstack([dense_rows[hold_idx], fam_onehot[hold_idx], legacy_rows[hold_idx], linear_hold, knn_hold])
    Y_fit = FT_pool[gbm_rows]
    for hidx in range(8):
        m = HistGradientBoostingRegressor(**GBM_PARAMS)
        m.fit(X_fit, Y_fit[:, hidx])
        meta_all[hold_idx, hidx] = m.predict(X_hold)
    pc_hold = np.exp(np.clip(meta_all[hold_idx, 3:6], -50.0, 50.0))
    grid = np.linspace(0.0, 1.0, LUT_NODES)
    gp = parent[gbm_rows]
    for g, (a, b) in enumerate([(0, 1), (1, 2)]):
        ds = true_s[:, b] - true_s[:, a]
        dc_raw = true_c[:, b] - true_c[:, a]
        floor = max(float(np.quantile(dc_raw[fit_idx], 0.05)), 1e-9)
        dc = np.maximum(dc_raw, floor)
        eff = ds / dc
        r_orig = rankdata(eff[fit_idx], method="average") / max(len(fit_idx) - 1, 1)
        r_of = {int(p): r for p, r in zip(fit_idx, r_orig)}
        r_fit = np.array([r_of[int(p)] for p in gp])
        q = np.quantile(eff[fit_idx], grid)
        dchat = np.maximum(pc_hold[:, b] - pc_hold[:, a], floor)
        m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(X_fit, r_fit)
        eff_hat = np.interp(np.clip(m.predict(X_hold), 0.0, 1.0), grid, q)
        rank_gain[hold_idx, g] = eff_hat * dchat
    prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
    fam_rows_hold = np.array([fam_mean[fam_names[i]] for i in hold_idx])
    prod = (1 - FAM_W) * prod + FAM_W * fam_rows_hold
    conf = np.clip(knn_hold[:, 6], 0.0, 1.0)[:, None] * CONF_SCALE
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
    prod_all[hold_idx] = prod
    print(f"{TAG} fold {fold} rows ridge={len(fit_pool)} gbm={len(gbm_rows)} knn={len(knn_rows)} "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

# component diagnostics (OOF) before allocation
lin_oof_note = ""
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))
sd = [rmse(meta_all[:, j], targets[:, j]) for j in range(3)]
cd = [rmse(meta_all[:, 3 + j], targets[:, 3 + j]) for j in range(3)]
print(f"{TAG} meta OOF rmse score {sd[0]:.4f}/{sd[1]:.4f}/{sd[2]:.4f} logcost {cd[0]:.4f}/{cd[1]:.4f}/{cd[2]:.4f}", flush=True)


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


rng2 = np.random.default_rng(BOOT_SEED)
samples = [rng2.integers(0, n, size=880) for _ in range(400)]
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
GRIDS = {"fast": np.arange(0.92, 1.0, 0.01), "balanced": np.arange(0.82, 0.94, 0.01),
         "premium": np.arange(0.80, 0.93, 0.01)}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}

mixed = (1 - RANK_BETA) * meta_all[:, 6:8] + RANK_BETA * rank_gain
meta = meta_all[:, :6].copy()
recon = np.column_stack([meta[:, 0], meta[:, 0] + mixed[:, 0], meta[:, 0] + mixed[:, 0] + mixed[:, 1]])
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
print(f"{TAG} RESULT weighted EV {tot_ev:.4f} score {tot_sc:.4f} | {'; '.join(per_tier)}", flush=True)
print(f"{TAG} DONE", flush=True)
