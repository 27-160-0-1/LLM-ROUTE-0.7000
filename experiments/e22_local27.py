"""E22: feature-extraction battery, one variant per process (--variant).

Variants:
  dense2  — 15 extra dense features (big-number counts, structure, style)
  word16k — word n-gram bins 8192 -> 16384
  word3   — add word trigrams
  prefix  — extra 4096-bin char-gram block over the first 120 chars
  e21conf — E21 ordinal confirmation with a different bootstrap seed
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
VARIANT = sys.argv[sys.argv.index("--variant") + 1]

from ossp_router import learned_router, legacy_hash_regex, similarity
from ossp_router.heuristic import episode_text, extract_features
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
BOOT_SEED = 27 if VARIANT == "e21conf" else 7

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

import re as _re

_BIGNUM = _re.compile(r"\d{5,}")
_EQ = _re.compile(r"=")
_LIST = _re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", _re.MULTILINE)
_VERB = _re.compile(r"^(?:Solve|Calculate|What|Find|How|Simplify|Evaluate|Let|Suppose|Prove)\b")
_QUOTE = _re.compile(r"\"[^\"]{3,}\"|'[^']{3,}'")
_KO_IMP = _re.compile(r"(?:하시오|하라|구하시오|쓰시오)")
_UNIT = _re.compile(r"[$%€£¥]|km|kg|cm|mm|ml|kWh")


def extra_dense(text):
    lines = text.splitlines() or [text]
    nonspace = max(1, sum(not c.isspace() for c in text[:4000]))
    digits = sum(c.isdigit() for c in text[:4000])
    upper = sum(c.isupper() for c in text[:4000])
    latin = max(1, sum(c.isalpha() and ord(c) < 128 for c in text[:4000]))
    depth = best = 0
    for c in text[:4000]:
        if c in "([{":
            depth += 1; best = max(best, depth)
        elif c in ")]}":
            depth = max(0, depth - 1)
    runs = _re.findall(r"\d+", text)
    sentences = _re.split(r"[.!?。]", text)
    return [
        math.log1p(max((len(r) for r in runs), default=0)),
        math.log1p(len(_BIGNUM.findall(text))),
        math.log1p(sum(len(l) for l in lines) / max(1, len(lines))),
        math.log1p(len(_LIST.findall(text))),
        upper / latin,
        digits / nonspace,
        math.log1p(len(_EQ.findall(text))),
        math.log1p(text.count("(") + text.count("[")),
        float(best),
        math.log1p(len(_UNIT.findall(text))),
        float(bool(_VERB.match(text))),
        float(text.rstrip().endswith("?")),
        math.log1p(len(_QUOTE.findall(text))),
        float(bool(_KO_IMP.search(text))),
        math.log1p(max((len(s) for s in sentences), default=0)),
    ]


print(f"[e22:{VARIANT}] shared blocks", flush=True)
from scipy import sparse

WORD_BINS = 16384 if VARIANT == "word16k" else artifact.word_hash_bins
dense_rows, legacy_rows, fam_names = [], [], []
rows_, cols_, vals_ = [], [], []
extra_rows = []
for ri, episode in enumerate(episodes):
    text = texts[ri]
    basic = extract_features(episode)
    tokens = learned_router._normalized_tokens(text)
    raw_dense = list(learned_router._raw_dense_from_context(episode, text, basic, tokens))
    if VARIANT == "dense2":
        raw_dense = raw_dense + extra_dense(text)
    dense_rows.append(raw_dense)
    word_values = [f"w1:{t}" for t in tokens]
    word_values += [f"w2:{l}\x1f{r}" for l, r in zip(tokens, tokens[1:])]
    if VARIANT == "word3":
        word_values += [f"w3:{a}\x1f{b}\x1f{c}" for a, b, c in zip(tokens, tokens[1:], tokens[2:])]
    block = learned_router._hashed_block(word_values, WORD_BINS)
    offset = 0
    for b, v in block.items():
        rows_.append(ri); cols_.append(offset + b); vals_.append(v)
    offset = WORD_BINS
    char_text = learned_router._normalized_char_text(text)
    char_values = (
        f"c{size}:{char_text[start:start + size]}"
        for size in (3, 4, 5)
        for start in range(0, max(0, len(char_text) - size + 1), 3)
    )
    block = learned_router._hashed_block(char_values, artifact.char_hash_bins)
    for b, v in block.items():
        rows_.append(ri); cols_.append(offset + b); vals_.append(v)
    offset += artifact.char_hash_bins
    if VARIANT == "prefix":
        head = char_text[:120]
        pvals = (
            f"p{size}:{head[start:start + size]}"
            for size in (3, 4, 5)
            for start in range(0, max(0, len(head) - size + 1), 1)
        )
        block = learned_router._hashed_block(pvals, 4096)
        for b, v in block.items():
            rows_.append(ri); cols_.append(offset + b); vals_.append(v)
    ls, lc = legacy_hash_regex.predict_episode(episode, legacy_artifact)
    legacy_rows.append([ls[m] for m in MODEL_IDS] + [math.log(lc[m]) for m in MODEL_IDS])
    fam_names.append(similarity.classify_family(text))

dense_rows = np.asarray(dense_rows)
legacy_rows = np.asarray(legacy_rows)
hash_dim = WORD_BINS + artifact.char_hash_bins + (4096 if VARIANT == "prefix" else 0)
hash_sp = sparse.csr_matrix((vals_, (rows_, cols_)), shape=(n, hash_dim))
dmean = dense_rows.mean(0)
dstd = dense_rows.std(0)
dstd[dstd < 1e-6] = 1.0
dense_sp = sparse.csr_matrix((dense_rows - dmean) / dstd)
X_sparse = sparse.hstack([dense_sp, hash_sp]).tocsr()
FAMILIES = list(similarity.FAMILY_NAMES)
fam_onehot = np.zeros((n, len(FAMILIES)))
for i, name in enumerate(fam_names):
    fam_onehot[i, FAMILIES.index(name)] = 1.0

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)
prod_all = np.zeros((n, 6))
meta_all = np.zeros((n, 8))
ordinal_all = np.zeros((n, 3)) if VARIANT == "e21conf" else None

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
    if ordinal_all is not None:
        for mi in range(3):
            cumulative = np.zeros(len(hold_idx))
            for t in (0.25, 0.5, 0.75, 1.0):
                y = (true_s[fit_idx][:, mi] >= t).astype(int)
                if y.min() == y.max():
                    cumulative += float(y.min()); continue
                clf = HistGradientBoostingClassifier(**GBM_PARAMS)
                clf.fit(X_fit, y)
                cumulative += clf.predict_proba(X_hold)[:, 1]
            ordinal_all[hold_idx, mi] = 0.25 * cumulative
    prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
    fam_rows_hold = np.array([fam_mean[fam_names[i]] for i in hold_idx])
    prod = (1 - FAM_W) * prod + FAM_W * fam_rows_hold
    conf = np.clip(knn_hold[:, 6], 0.0, 1.0)[:, None] * CONF_SCALE
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
    prod_all[hold_idx] = prod
    print(f"[e22:{VARIANT}] fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)


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


def evaluate(label, meta6):
    tot_ev = tot_sc = 0.0
    for tier, mult in MULTS.items():
        stacked = (1 - TIER_BLENDS[tier]) * prod_all + TIER_BLENDS[tier] * meta6
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
    print(f"[e22:{VARIANT}] {label}: EV {tot_ev:.4f} score {tot_sc:.4f}", flush=True)


def with_gain(meta8):
    meta = meta8[:, :6].copy()
    recon = np.column_stack([meta[:, 0], meta[:, 0] + meta8[:, 6],
                             meta[:, 0] + meta8[:, 6] + meta8[:, 7]])
    meta[:, :3] = (1 - GAIN_ALPHA) * meta[:, :3] + GAIN_ALPHA * recon
    return meta


evaluate("result", with_gain(meta_all))
if ordinal_all is not None:
    ord8 = meta_all.copy()
    ord8[:, :3] = ordinal_all
    evaluate("ordinal(seed17)", with_gain(ord8))
print(f"[e22:{VARIANT}] DONE", flush=True)
