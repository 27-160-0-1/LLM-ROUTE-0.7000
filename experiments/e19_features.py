"""E19: char n-gram feature-space sweep for the ridge layer.

Configs vary char hash bins / extraction stride / text limit.  Word block,
kNN rows (k=16), family means, and fold splits are shared; the ridge (and
its inner OOF) plus the 8 GBM heads are refit per config.
"""

import math
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(HERE / "src"))

from ossp_router import learned_router, legacy_hash_regex, similarity
from ossp_router.heuristic import episode_text, extract_features
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes

similarity.NEIGHBORS = 16  # E20 adoption (VM bundle src may be stale)

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
CONFIGS = [
    ("base(8192,s3,6k)", 8192, 3, 6000),
    ("bins16384", 16384, 3, 6000),
    ("stride1", 8192, 1, 6000),
    ("limit8k", 8192, 3, 8000),
]

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

print("[e19] shared blocks", flush=True)
from scipy import sparse

WORD_BINS = artifact.word_hash_bins
DENSE_N = len(learned_router.DENSE_FEATURE_NAMES)
dense_rows, legacy_rows, fam_names = [], [], []
word_rows_cols_vals = ([], [], [])
dense_std_rows = []
char_texts = {}
token_cache = []
for ri, episode in enumerate(episodes):
    text = texts[ri]
    basic = extract_features(episode)
    tokens = learned_router._normalized_tokens(text)
    token_cache.append(tokens)
    raw_dense = learned_router._raw_dense_from_context(episode, text, basic, tokens)
    dense_rows.append(raw_dense)
    standardized = [
        (v - m) / s for v, m, s in zip(raw_dense, artifact.dense_mean, artifact.dense_scale)
    ]
    dense_std_rows.append(standardized)
    word_values = [f"w1:{t}" for t in tokens]
    word_values += [f"w2:{l}\x1f{r}" for l, r in zip(tokens, tokens[1:])]
    block = learned_router._hashed_block(word_values, WORD_BINS)
    for b, v in block.items():
        word_rows_cols_vals[0].append(ri)
        word_rows_cols_vals[1].append(b)
        word_rows_cols_vals[2].append(v)
    ls, lc = legacy_hash_regex.predict_episode(episode, legacy_artifact)
    legacy_rows.append([ls[m] for m in MODEL_IDS] + [math.log(lc[m]) for m in MODEL_IDS])
    fam_names.append(similarity.classify_family(text))
dense_rows = np.asarray(dense_rows)
legacy_rows = np.asarray(legacy_rows)
dense_std = np.asarray(dense_std_rows)
FAMILIES = list(similarity.FAMILY_NAMES)
fam_onehot = np.zeros((n, len(FAMILIES)))
for i, name in enumerate(fam_names):
    fam_onehot[i, FAMILIES.index(name)] = 1.0


def char_matrix(char_bins, stride, limit):
    rows, cols, vals = [], [], []
    space = similarity._SPACE if hasattr(similarity, "_SPACE") else None
    import re as _re

    space_re = _re.compile(r"\s+")
    digit_re = _re.compile(r"\d+")
    for ri, text in enumerate(texts):
        normalized = space_re.sub(" ", digit_re.sub("0", text.casefold())).strip()
        if len(normalized) > limit:
            half = limit // 2
            normalized = normalized[:half] + " … " + normalized[-half:]
        values = (
            f"c{size}:{normalized[start:start + size]}"
            for size in (3, 4, 5)
            for start in range(0, max(0, len(normalized) - size + 1), stride)
        )
        block = learned_router._hashed_block(values, char_bins)
        for b, v in block.items():
            rows.append(ri); cols.append(b); vals.append(v)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, char_bins))


dense_sp = sparse.csr_matrix(dense_std)
word_sp = sparse.csr_matrix(
    (word_rows_cols_vals[2], (word_rows_cols_vals[0], word_rows_cols_vals[1])),
    shape=(n, WORD_BINS),
)

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)

# fold-shared: kNN rows + family means + fold indices
print("[e19] building shared kNN/fold blocks", flush=True)
knn_fit_all = [None] * 5
knn_hold_all = [None] * 5
fam_hold_all = [None] * 5
for fold in range(5):
    t0 = time.perf_counter()
    hold = fold_of == fold
    fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
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

    knn_fit_all[fold] = np.array([kq(t, exclude=i) for i, t in enumerate(fit_texts)])
    knn_hold_all[fold] = np.array([kq(texts[i]) for i in hold_idx])
    fam_mean = {}
    by_family = defaultdict(list)
    for i in fit_idx:
        by_family[fam_names[i]].append(targets[i])
    fglobal = targets[fit_idx].mean(axis=0)
    for name in FAMILIES:
        rows = by_family.get(name, [])
        fam_mean[name] = np.mean(rows, axis=0) if len(rows) >= 8 else fglobal
    fam_hold_all[fold] = np.array([fam_mean[fam_names[i]] for i in hold_idx])
    print(f"[e19] fold {fold} shared {time.perf_counter()-t0:.0f}s", flush=True)


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

for label, char_bins, stride, limit in CONFIGS:
    t0 = time.perf_counter()
    X_sparse = sparse.hstack([dense_sp, word_sp, char_matrix(char_bins, stride, limit)]).tocsr()
    prod_all = np.zeros((n, 6))
    meta_all = np.zeros((n, 8))
    for fold in range(5):
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
        X_fit = np.hstack([dense_rows[fit_idx], fam_onehot[fit_idx], legacy_rows[fit_idx],
                           inner_oof, knn_fit_all[fold]])
        X_hold = np.hstack([dense_rows[hold_idx], fam_onehot[hold_idx], legacy_rows[hold_idx],
                            linear_hold, knn_hold_all[fold]])
        for hidx in range(8):
            m = HistGradientBoostingRegressor(**GBM_PARAMS)
            m.fit(X_fit, full_targets[fit_idx][:, hidx])
            meta_all[hold_idx, hidx] = m.predict(X_hold)
        prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
        prod = (1 - FAM_W) * prod + FAM_W * fam_hold_all[fold]
        conf = np.clip(knn_hold_all[fold][:, 6], 0.0, 1.0)[:, None] * CONF_SCALE
        prod = (1 - conf) * prod + conf * knn_hold_all[fold][:, :6]
        prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
        prod_all[hold_idx] = prod
    meta = meta_all[:, :6].copy()
    recon = np.column_stack([meta[:, 0], meta[:, 0] + meta_all[:, 6],
                             meta[:, 0] + meta_all[:, 6] + meta_all[:, 7]])
    meta[:, :3] = (1 - GAIN_ALPHA) * meta[:, :3] + GAIN_ALPHA * recon
    tot_ev = tot_sc = 0.0
    for tier, mult in MULTS.items():
        stacked = (1 - TIER_BLENDS[tier]) * prod_all + TIER_BLENDS[tier] * meta
        ps = np.clip(stacked[:, :3], 0, 1)
        pc = np.exp(np.clip(stacked[:, 3:], -50, 50))
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12))
        pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
        best = None
        for s in GRIDS[tier]:
            busts, evs = 0, []
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
                pick = allocate(ps, pc, mult, s)
                best = (ev, s, float(true_s[np.arange(n), pick].mean()))
        tot_ev += W[tier] * best[0]; tot_sc += W[tier] * best[2]
    print(f"[e19] {label}: EV {tot_ev:.4f} score {tot_sc:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
print("[e19] DONE", flush=True)
