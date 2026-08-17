"""E38: external routing methods rebuilt on our data and evaluated head-to-head.

Sources: RouteLLM (lm-sys), LLMRouter (ulab-uiuc, arXiv 2608.06867),
awesome-model-routing (resolves to the same methods).  Each is rebuilt in
its published form on our 2,640 x 3 (score, cost) matrix, with the encoder
approximated by our hashed n-gram + dense features (stdlib constraint), and
then evaluated with the same nested-CV + 880-bootstrap EV harness and the
same Lagrangian allocator as the deployed router (so the comparison isolates
the ROUTER SIGNAL).  Where a method is inherently binary strong-vs-weak
(RouteLLM), it is evaluated both in its native threshold form and lifted to
3 models via our allocator.

Methods:
  R-MF     RouteLLM matrix factorization: prompt latent u=W·phi(x), model
           latent v_m, score s_m=<u,v_m>+b_m; BCE on pairwise wins.
  R-SW     RouteLLM similarity-weighted ranking: Bradley-Terry over the k
           most similar train prompts, votes weighted by similarity.
  R-BIN    RouteLLM-style binary strong-vs-weak classifier (GBM), threshold
           routing at a target strong-call fraction (native) and via allocator.
  L-KNN    LLMRouter KNNRouter: per-model score = similarity-weighted mean
           of neighbours' scores (already ~ our kNN block; included as-is).
  L-SVM    LLMRouter SVMRouter: linear SVM (one-vs-rest) on best-model label.
  L-MLP    LLMRouter MLPRouter: MLP on features -> per-model scores.
  L-ELO    LLMRouter EloRouter: per-(family, model) Elo from pairwise
           outcomes on train, routed by family.
  L-DC     RouterDC: dual-contrastive -- learn prompt encoder so that
           <u, v_m> is high for models that scored well (InfoNCE-like), then
           per-model score = softmax over models.
  L-GRAPH  GraphRouter (approx.): bipartite prompt-model graph, label
           propagation from kNN prompt neighbours through model nodes.
Deployed router row is the reference.
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

print("[e38] shared blocks", flush=True)
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
X_sparse = sparse.csr_matrix((svals, (srows, scols)), shape=(n, dim), dtype=np.float64)
FAMILIES = list(similarity.FAMILY_NAMES)
fam_idx = np.array([FAMILIES.index(f) for f in fam_names])
fam_onehot = np.zeros((n, len(FAMILIES)))
fam_onehot[np.arange(n), fam_idx] = 1.0

# a compact dense embedding for the neural-ish methods: TruncatedSVD of the sparse features (fold-pure below)
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

EMB_DIM = 128


def ridge_fit(Xf, y):
    return Ridge(alpha=30.0, solver="sparse_cg").fit(Xf, y)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


# ---------------------------------------------------------------- external routers
def routellm_mf(E_fit, S_fit, E_hold, rank=16, epochs=200, lr=0.05, l2=1e-3, seed=0):
    """RouteLLM MF: u = W phi(x); score_m = <u, v_m> + b_m; BCE on pairwise wins (m beats m')."""
    rng_ = np.random.default_rng(seed)
    d = E_fit.shape[1]
    W = rng_.normal(0, 0.05, (d, rank)); V = rng_.normal(0, 0.05, (3, rank)); b = np.zeros(3)
    pairs = [(a, c) for a in range(3) for c in range(3) if a != c]
    for _ in range(epochs):
        U = E_fit @ W
        gW = np.zeros_like(W); gV = np.zeros_like(V); gb = np.zeros_like(b)
        for a, c in pairs:
            y = (S_fit[:, a] > S_fit[:, c]).astype(float)
            mask = S_fit[:, a] != S_fit[:, c]
            if not mask.any():
                continue
            z = (U @ (V[a] - V[c])) + (b[a] - b[c])
            g = (sigmoid(z) - y) * mask / mask.sum()
            gW += np.outer(E_fit.T @ g, V[a] - V[c])
            gV[a] += g @ U; gV[c] -= g @ U
            gb[a] += g.sum(); gb[c] -= g.sum()
        W -= lr * (gW + l2 * W); V -= lr * (gV + l2 * V); b -= lr * gb
    U_h = E_hold @ W
    return U_h @ V.T + b  # (n_hold, 3) logits; softmax over models ~ win-prob ordering


def routellm_sw_ranking(sim_fn, S_fit, k=32):
    """Similarity-weighted Bradley-Terry: for a query, take k nearest train prompts, each contributes
    weighted pairwise 'votes' (model a beat c on that prompt); solve BT ratings by a few MM iterations."""
    def route(neigh_idx, neigh_sim):
        w = np.maximum(neigh_sim, 0) + 1e-6
        wins = np.zeros((3, 3))
        for j, i in enumerate(neigh_idx):
            for a in range(3):
                for c in range(3):
                    if a != c and S_fit[i, a] > S_fit[i, c]:
                        wins[a, c] += w[j]
        r = np.ones(3)
        for _ in range(30):
            for a in range(3):
                num = wins[a].sum()
                den = sum((wins[a, c] + wins[c, a]) / (r[a] + r[c]) for c in range(3) if c != a)
                r[a] = num / den if den > 0 else r[a]
            r /= r.sum()
        return np.log(r + 1e-9)
    return route


def routerdc(E_fit, S_fit, E_hold, rank=32, epochs=150, lr=0.05, tau=0.1, seed=0):
    """RouterDC (approx.): contrastive -- pull prompt latent toward embeddings of models that scored
    well, push from those that scored poorly. score_m = <u,v_m>/tau; loss = -sum_m w_m log softmax_m."""
    rng_ = np.random.default_rng(seed)
    d = E_fit.shape[1]
    W = rng_.normal(0, 0.05, (d, rank)); V = rng_.normal(0, 0.05, (3, rank))
    tgt = S_fit / np.maximum(S_fit.sum(axis=1, keepdims=True), 1e-9)  # soft target over models
    for _ in range(epochs):
        U = normalize(E_fit @ W)
        logits = (U @ normalize(V).T) / tau
        p = np.exp(logits - logits.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
        g = (p - tgt) / len(E_fit)
        gV = g.T @ U; gU = g @ V
        W -= lr * (E_fit.T @ gU); V -= lr * gV
    U_h = normalize(E_hold @ W)
    return (U_h @ normalize(V).T) / tau


rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)
prod_all = np.zeros((n, 6))
meta_all = np.zeros((n, 8))
ord_score = np.zeros((n, 3))
rank_gain = np.zeros((n, 2))
ext = {k: np.zeros((n, 3)) for k in ("R-MF", "R-SW", "R-BIN", "L-KNN", "L-SVM", "L-MLP", "L-ELO", "L-DC", "L-GRAPH")}

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

    def knn_neighbors(text, k, exclude=None):
        q = similarity.tfidf_vector(text, idf)
        scores = {}
        for g, v in q.items():
            for d, s in knn_index.postings.get(g, ()):
                if d == exclude:
                    continue
                scores[d] = scores.get(d, 0.0) + v * s
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [d for d, _ in ranked], np.array([s for _, s in ranked])

    def kq(text, exclude=None):
        idxs, sims = knn_neighbors(text, similarity.NEIGHBORS, exclude)
        if not idxs:
            return np.concatenate([gmean, [0.0]])
        tot = sims.sum()
        row = sum((s / tot) * targets[fit_idx][d] for d, s in zip(idxs, sims))
        return np.concatenate([row, [sims[0]]])

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
    prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
    fam_rows_hold = np.array([fam_mean[fam_names[i]] for i in hold_idx])
    prod = (1 - FAM_W) * prod + FAM_W * fam_rows_hold
    conf = np.clip(knn_hold[:, 6], 0.0, 1.0)[:, None] * CONF_SCALE
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
    prod_all[hold_idx] = prod

    # ---- embedding for external methods (fold-pure SVD of sparse features)
    svd = TruncatedSVD(EMB_DIM, random_state=fold).fit(X_sparse[fit_idx])
    E_fit = normalize(svd.transform(X_sparse[fit_idx])); E_hold = normalize(svd.transform(X_sparse[hold_idx]))
    S_fit = true_s[fit_idx]

    # R-MF
    ext["R-MF"][hold_idx] = sigmoid(routellm_mf(E_fit, S_fit, E_hold, seed=fold))
    # R-SW
    route = routellm_sw_ranking(None, S_fit)
    for j, i in enumerate(hold_idx):
        idxs, sims = knn_neighbors(texts[i], 32)
        ext["R-SW"][i] = sigmoid(route(idxs, sims)) if idxs else 0.5
    # R-BIN: strong(think) beats weak(light) classifier -> P(strong wins); lift to 3 via P(mid beats light)
    y_sw = (S_fit[:, 2] > S_fit[:, 0]).astype(int)
    y_mw = (S_fit[:, 1] > S_fit[:, 0]).astype(int)
    p_sw = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X_fit, y_sw).predict_proba(X_hold)[:, 1]
    p_mw = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X_fit, y_mw).predict_proba(X_hold)[:, 1]
    ext["R-BIN"][hold_idx] = np.column_stack([np.zeros(len(hold_idx)), p_mw, p_sw])  # 'win over light' scale
    # L-KNN: neighbours' mean scores (k=16, similarity weighted) -- equals our kNN block score part
    ext["L-KNN"][hold_idx] = knn_hold[:, :3]
    # L-SVM: one-vs-rest linear SVM on best-model label (ties -> cheapest best)
    best_lbl = np.array([int(np.argmax(r + np.array([2e-9, 1e-9, 0]))) for r in S_fit])
    svm = LinearSVC(C=0.5, max_iter=5000).fit(E_fit, best_lbl)
    dec = svm.decision_function(E_hold)
    if dec.ndim == 1:
        dec = np.column_stack([-dec, dec, np.zeros(len(dec))])
    ext["L-SVM"][hold_idx] = sigmoid(dec)
    # L-MLP: embedding -> per-model scores
    mlp = MLPRegressor(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=400, early_stopping=True, random_state=fold)
    mlp.fit(E_fit, S_fit)
    ext["L-MLP"][hold_idx] = np.clip(mlp.predict(E_hold), 0, 1)
    # L-ELO: per-family Elo from pairwise outcomes on fit, applied by family
    elo = {f: np.zeros(3) for f in range(len(FAMILIES))}
    K = 16.0
    for i in fit_idx:
        f = fam_idx[i]
        for a in range(3):
            for c in range(a + 1, 3):
                if true_s[i, a] == true_s[i, c]:
                    continue
                ea = 1 / (1 + 10 ** ((elo[f][c] - elo[f][a]) / 400))
                sa = 1.0 if true_s[i, a] > true_s[i, c] else 0.0
                elo[f][a] += K * (sa - ea); elo[f][c] -= K * (sa - ea)
    for i in hold_idx:
        ext["L-ELO"][i] = sigmoid(elo[fam_idx[i]] / 400)
    # L-DC
    dc = routerdc(E_fit, S_fit, E_hold, seed=fold)
    p = np.exp(dc - dc.max(axis=1, keepdims=True)); ext["L-DC"][hold_idx] = p / p.sum(axis=1, keepdims=True)
    # L-GRAPH: bipartite propagation -- prompt->neighbour prompts->their model scores, then model->prompt via
    # model-node embedding = mean of prompt embeddings weighted by score; final = 0.5*neighbour + 0.5*<E, model_node>
    model_nodes = np.stack([(E_fit * S_fit[:, m:m + 1]).sum(axis=0) / max(S_fit[:, m].sum(), 1e-9) for m in range(3)])
    aff = sigmoid(E_hold @ normalize(model_nodes).T * 5)
    ext["L-GRAPH"][hold_idx] = 0.5 * knn_hold[:, :3] + 0.5 * aff
    print(f"[e38] fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)

# ---- signal quality: correlation with true score and pairwise accuracy (think vs light)
print("[e38] === router signal quality (OOF) ===", flush=True)
def pair_acc(P):
    d = P[:, 2] - P[:, 0]; t = true_s[:, 2] - true_s[:, 0]
    m = t != 0
    return ((d[m] > 0) == (t[m] > 0)).mean()
print(f"[e38] deployed(ordinal): corr {[round(np.corrcoef(ord_score[:, k], true_s[:, k])[0,1],3) for k in range(3)]} pair-acc(think>light) {pair_acc(ord_score):.3f}", flush=True)
for k, P in ext.items():
    cs = [round(np.corrcoef(P[:, m], true_s[:, m])[0, 1], 3) if P[:, m].std() > 0 else float("nan") for m in range(3)]
    print(f"[e38] {k:8s}: corr {cs} pair-acc(think>light) {pair_acc(P):.3f}", flush=True)


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


def allocate_threshold_binary(p_strong, pc, mult, safety):
    """RouteLLM native: route to think if P(strong wins) >= theta, else light; theta bisected to budget."""
    lt = pc[:, 0].sum(); cap = lt * max(1.0, mult * safety)

    def choose(theta):
        pick = np.where(p_strong >= theta, 2, 0)
        return pick, pc[np.arange(len(pick)), pick].sum()

    lo, hi = 0.0, 1.0
    best, tot = choose(hi)
    for _ in range(40):
        mid = (lo + hi) / 2
        p, t = choose(mid)
        if t <= cap:
            hi, best = mid, p
        else:
            lo = mid
    return best


rng2 = np.random.default_rng(7)
samples = [rng2.integers(0, n, size=880) for _ in range(400)]
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
GRIDS = {"fast": np.arange(0.90, 1.0, 0.01), "balanced": np.arange(0.78, 0.94, 0.01),
         "premium": np.arange(0.74, 0.93, 0.01)}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}

# deployed cost side is shared by all (isolates the SCORE signal); deployed score for reference
meta_dep = meta_all[:, :6].copy(); meta_dep[:, :3] = ord_score
d1 = (1 - RANK_BETA) * meta_all[:, 6] + RANK_BETA * rank_gain[:, 0]
d2 = (1 - RANK_BETA) * meta_all[:, 7] + RANK_BETA * rank_gain[:, 1]
recon = np.column_stack([meta_dep[:, 0], meta_dep[:, 0] + d1, meta_dep[:, 0] + d1 + d2])
meta_dep[:, :3] = (1 - GAIN_ALPHA) * meta_dep[:, :3] + GAIN_ALPHA * recon


def cost_matrix(tier):
    stacked = (1 - TIER_BLENDS[tier]) * prod_all + TIER_BLENDS[tier] * meta_dep
    pc = np.exp(np.clip(stacked[:, 3:], -50, 50))
    pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12)); pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
    return pc


def evaluate(label, score_fn, allocator=allocate):
    tot_ev = tot_sc = 0.0; parts = []
    for tier, mult in MULTS.items():
        ps = score_fn(tier); pc = cost_matrix(tier)
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
    print(f"[e38] {label}: weighted EV {tot_ev:.4f} score {tot_sc:.4f} | {'; '.join(parts)}", flush=True)
    return tot_ev


results = {}
def dep_score(tier):
    stacked = (1 - TIER_BLENDS[tier]) * prod_all + TIER_BLENDS[tier] * meta_dep
    return np.clip(stacked[:, :3], 0, 1)
results["deployed"] = evaluate("deployed router", dep_score)
# each external router's own scores through OUR allocator (isolates router signal); tier-independent scores
for k in ("R-MF", "R-SW", "L-KNN", "L-SVM", "L-MLP", "L-ELO", "L-DC", "L-GRAPH"):
    P = ext[k]
    results[k] = evaluate(f"{k} (own scores + our allocator)", lambda tier, P=P: P)
# R-BIN native RouteLLM: binary threshold light/think on P(strong wins)
results["R-BIN native"] = evaluate("R-BIN RouteLLM native threshold (light/think)", lambda tier: ext["R-BIN"][:, 2], allocate_threshold_binary)
# R-BIN lifted: use win-over-light probs as 3-model score proxy
results["R-BIN lifted"] = evaluate("R-BIN lifted to 3 models via allocator", lambda tier: np.column_stack([np.full(n, 0.5), ext["R-BIN"][:, 1], ext["R-BIN"][:, 2]]))
# blend test: does any external signal ADD to deployed? (0.2 blend into score row)
for k in ("R-MF", "L-DC", "L-GRAPH", "L-MLP"):
    P = ext[k]
    def blend(tier, P=P):
        base = dep_score(tier)
        Pn = (P - P.mean(axis=0)) / (P.std(axis=0) + 1e-9) * base.std(axis=0) + base.mean(axis=0)
        return np.clip(0.8 * base + 0.2 * Pn, 0, 1)
    results[f"deployed+0.2·{k}"] = evaluate(f"deployed + 0.2·{k}", blend)
best_ext = max(((k, v) for k, v in results.items() if k != "deployed" and not k.startswith("deployed+")), key=lambda kv: kv[1])
best_bl = max(((k, v) for k, v in results.items() if k.startswith("deployed+")), key=lambda kv: kv[1])
print(f"[e38] BEST external {best_ext[0]} EV {best_ext[1]:.4f} vs deployed {results['deployed']:.4f} ({best_ext[1]-results['deployed']:+.4f})", flush=True)
print(f"[e38] BEST blend {best_bl[0]} EV {best_bl[1]:.4f} ({best_bl[1]-results['deployed']:+.4f})", flush=True)
print("[e38] DONE", flush=True)
