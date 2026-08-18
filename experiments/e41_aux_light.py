"""E41: self-labelled ax31-light auxiliary data (Colab, skt/A.X-3.1-Light) as extra training rows.

Question: if the light-score / light-cost columns of the ridge (and optionally the two light
meta-GBM heads) are trained on public rows + W-weighted aux rows (data/aux/light-labels.v1.json,
built by colab-label/ingest_labels.py), how far does the deployed pipeline's CV EV move?

Aux rows carry ONLY light labels: they never enter the mid/think columns, the gain/ordinal/rank
heads, the kNN index, or any hold-out fold. Aux rows whose leakage `group` (shared passage /
code / question / context) matches a hold-out item are excluded from that fold's training.

Usage: python e41_aux_light.py W WHERE [SEED] [CALIB] [AUX_PATH]
  W        aux sample weight (0 = baseline, must reproduce E40 K=0: s7 0.6969)
  WHERE    ridge | ridge_gbm     (ridge light columns only | + light GBM heads)
  SEED     bootstrap seed (default 7)
  CALIB    1 = apply per-family organizer-vs-local calibration (score bias, out-len ratio, in-len offset)
           from data/aux/light-calibration.v1.json (default 1); 0 = raw local labels
  AUX_PATH override aux json (default data/aux/light-labels.v1.json)
  SCOREONLY 1 = use aux rows only for the light SCORE column (skip their cost labels; use when the pilot
           shows poor output-length agreement), default 0
  AUX_FAMS  comma-separated family whitelist for aux rows (default: all except ruletaker, whose pilot
           agreement with the organizer was ~0.55/corr 0); '-' = no filter
Same nested CV + 880x400 bootstrap harness as E27/E39/E40, beta=0.25 rank head.
"""

import json
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
from ossp_router.protocol import MODEL_IDS, Episode, load_bundled_policy, load_input, load_outcomes

W_AUX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
WHERE = sys.argv[2] if len(sys.argv) > 2 else "ridge"
BOOT_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 7
CALIB = (sys.argv[4] != "0") if len(sys.argv) > 4 else True
AUX_PATH = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] not in ("-", "") else ROOT / "data/aux/light-labels.v1.json"
SCOREONLY = (sys.argv[6] == "1") if len(sys.argv) > 6 else False
_DEF_FAMS = "gsm8k_or_other,aime,dmmath,belebele,code,truthfulqa,longdoc,hrmcr"
AUX_FAMS = set((sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] != "-" else _DEF_FAMS).split(",")) if not (len(sys.argv) > 7 and sys.argv[7] == "-") else None
TAG = f"[e41 W={W_AUX} {WHERE} s{BOOT_SEED} calib={int(CALIB)} scoreonly={int(SCOREONLY)} fams={'all' if AUX_FAMS is None else len(AUX_FAMS)}]"
USE_AUX = W_AUX > 0
AUX_GBM = USE_AUX and WHERE == "ridge_gbm"

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
LIGHT_COLS = [0] if SCOREONLY else [0, 3]           # score_light (, logcost_light)
REST_COLS = [c for c in range(6) if c not in LIGHT_COLS]

episodes = list(inputs.episodes)
n = len(episodes)
index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}


def cost_of(mid, input_tokens, output_tokens):
    r = policy.models[mid]
    unit = Decimal(policy.token_unit)
    return float(r.fixed_cost + Decimal(int(input_tokens)) * r.input_token_rate / unit
                 + Decimal(int(output_tokens)) * r.output_token_rate / unit)


def true_cost(eid, mid):
    o = index[(eid, mid)]
    return cost_of(mid, o.input_tokens, o.output_tokens)


true_s = np.array([[float(index[(e.episode_id, m)].score) for m in MODEL_IDS] for e in episodes])
true_c = np.array([[true_cost(e.episode_id, m) for m in MODEL_IDS] for e in episodes])
targets = np.hstack([true_s, np.log(true_c)])
delta_targets = np.column_stack([targets[:, 1] - targets[:, 0], targets[:, 2] - targets[:, 1]])
full_targets = np.hstack([targets, delta_targets])

# ---- leakage groups for public items (same rule as colab-label/build_pool.group_key) ----
sys.path.insert(0, str(ROOT / "colab-label"))
from build_pool import group_key  # noqa: E402

texts_pub = [episode_text(e) for e in episodes]
fam_pub = [similarity.classify_family(t) for t in texts_pub]
group_pub = [group_key(f, t) for f, t in zip(fam_pub, texts_pub)]

# ---- aux rows ----
aux_eps, aux_group, aux_light = [], [], []   # aux_light: (score, logcost)
if USE_AUX:
    doc = json.loads(AUX_PATH.read_text(encoding="utf-8"))
    calib = {}
    if CALIB:
        cp = AUX_PATH.parent / "light-calibration.v1.json"
        if cp.exists():
            allc = json.loads(cp.read_text(encoding="utf-8"))
            calib = allc.get(str(doc.get("calibration_key")), next(iter(allc.values()), {}))
    for e in doc["episodes"]:
        if e.get("truncated"):
            continue
        if AUX_FAMS is not None and e["family"] not in AUX_FAMS:
            continue
        lab = e["ax31-light"]
        ng = lab["num_generations"]
        c = calib.get(e["family"], calib.get("_all", {})) if calib else {}
        score = float(lab["score"]) + float(c.get("score_bias", 0.0))
        score = min(1.0, max(0.0, score))
        in_per = lab["input_tokens"] / ng + float(c.get("inlen_offset", 0.0))
        out_per = math.expm1(math.log1p(lab["output_tokens"] / ng) + float(c.get("outlen_logratio", 0.0)))
        cost = cost_of("ax31-light", max(1, round(in_per * ng)), max(0, round(out_per * ng)))
        aux_eps.append(Episode(episode_id=e["episode_id"], prompt=e["prompt"]))
        aux_group.append(e.get("group") or group_key(e["family"], e["prompt"]))
        aux_light.append((score, math.log(cost)))
    print(f"{TAG} aux rows {len(aux_eps)} (calibrated={bool(calib)})", flush=True)
aux_light = np.asarray(aux_light).reshape(-1, 2)
if SCOREONLY:
    aux_light = aux_light[:, :1]
n_aux = len(aux_eps)

pool_eps = list(episodes) + aux_eps
N = len(pool_eps)
texts = texts_pub + [episode_text(e) for e in aux_eps]
groups = group_pub + aux_group

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
# light-only target matrix for the pool: public rows carry their true light labels, aux rows theirs
T_light = np.vstack([targets[:, LIGHT_COLS], aux_light]) if n_aux else targets[:, LIGHT_COLS]
print(f"{TAG} features for {N} rows in {time.perf_counter()-t0:.0f}s", flush=True)

rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)          # per public episode (identical to E27/E40)
prod_all = np.zeros((n, 6))
meta_all = np.zeros((n, 8))
rank_gain = np.zeros((n, 2))


def fit_ridge_light(rows, w):
    m = Ridge(alpha=30.0, solver="sparse_cg")
    m.fit(X_sparse[rows], T_light[rows], sample_weight=w)
    return m


for fold in range(5):
    t0 = time.perf_counter()
    hold = fold_of == fold
    fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
    hold_groups = {groups[i] for i in hold_idx}
    aux_ok = np.array([n + j for j in range(n_aux) if groups[n + j] not in hold_groups], dtype=int)
    # ---- ridge: rest columns on public fit rows; light columns on public fit + aux (weight W) ----
    ridge_rest = Ridge(alpha=30.0, solver="sparse_cg")
    ridge_rest.fit(X_sparse[fit_idx], targets[fit_idx][:, REST_COLS])
    light_rows = np.concatenate([fit_idx, aux_ok]) if USE_AUX else fit_idx
    light_w = np.concatenate([np.ones(len(fit_idx)), np.full(len(aux_ok), W_AUX)]) if USE_AUX else None
    ridge_light = fit_ridge_light(light_rows, light_w)
    linear_hold = np.zeros((len(hold_idx), 6))
    linear_hold[:, REST_COLS] = ridge_rest.predict(X_sparse[hold_idx])
    linear_hold[:, LIGHT_COLS] = ridge_light.predict(X_sparse[hold_idx]).reshape(len(hold_idx), -1)
    linear_hold[:, :3] = np.clip(linear_hold[:, :3], 0.0, 1.0)

    # ---- inner OOF (5 inner folds over public fit rows; aux joins the light ridge in every inner fit) ----
    inner_fold = np.random.default_rng(fold).integers(0, 5, size=len(fit_idx))
    gbm_rows = np.concatenate([fit_idx, aux_ok]) if AUX_GBM else fit_idx
    inner_oof = np.zeros((len(gbm_rows), 6))
    if AUX_GBM:
        # aux rows need inner-OOF light predictions too: give each aux row an inner fold by hash of its group
        aux_inner = np.array([int(groups[r][:8], 16) % 5 for r in aux_ok])
    for inner in range(5):
        tr_pub = fit_idx[inner_fold != inner]
        m_rest = Ridge(alpha=30.0, solver="sparse_cg").fit(X_sparse[tr_pub], targets[tr_pub][:, REST_COLS])
        if USE_AUX:
            tr_aux = aux_ok[aux_inner != inner] if AUX_GBM else aux_ok
            rows = np.concatenate([tr_pub, tr_aux]); w = np.concatenate([np.ones(len(tr_pub)), np.full(len(tr_aux), W_AUX)])
        else:
            rows, w = tr_pub, None
        m_light = fit_ridge_light(rows, w)
        te_pub = np.where(inner_fold == inner)[0]
        te_rows = list(te_pub)
        if AUX_GBM:
            te_rows += [len(fit_idx) + k for k in np.where(aux_inner == inner)[0]]
        te_rows = np.asarray(te_rows, dtype=int)
        Xte = X_sparse[gbm_rows[te_rows]]
        inner_oof[np.ix_(te_rows, REST_COLS)] = m_rest.predict(Xte)
        inner_oof[np.ix_(te_rows, LIGHT_COLS)] = m_light.predict(Xte).reshape(len(te_rows), -1)
    inner_oof[:, :3] = np.clip(inner_oof[:, :3], 0.0, 1.0)

    # ---- kNN index over public fit rows only ----
    knn_texts = [texts[i] for i in fit_idx]
    freqs, total = similarity.document_frequencies(knn_texts)
    idf = similarity.idf_table(freqs, total)
    knn_vectors = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS) for t in knn_texts]
    knn_index = similarity.KnnIndex(knn_vectors, targets[fit_idx].tolist())
    gmean = targets[fit_idx].mean(axis=0)

    def kq(text, exclude=None):
        q = similarity.tfidf_vector(text, idf)
        if not q:
            return np.concatenate([gmean, [0.0]])
        scores = {}
        for g, v in q.items():
            for d, s in knn_index.postings.get(g, ()):
                if exclude is not None and d == exclude:
                    continue
                scores[d] = scores.get(d, 0.0) + v * s
        if not scores:
            return np.concatenate([gmean, [0.0]])
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:similarity.NEIGHBORS]
        tot = sum(s for _d, s in ranked)
        row = np.zeros(6)
        for d, s in ranked:
            row += (s / tot) * targets[fit_idx[d]]
        return np.concatenate([row, [ranked[0][1]]])

    pos_in_fit = {int(i): k for k, i in enumerate(fit_idx)}
    knn_fit = np.array([kq(texts[r], exclude=pos_in_fit.get(int(r))) for r in gbm_rows])
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
    n_pub_fit = len(fit_idx)
    Y_pub = full_targets[fit_idx]
    for hidx in range(8):
        m = HistGradientBoostingRegressor(**GBM_PARAMS)
        if AUX_GBM and hidx in LIGHT_COLS:
            y = np.concatenate([Y_pub[:, hidx], aux_light[aux_ok - n, LIGHT_COLS.index(hidx)]])
            w = np.concatenate([np.ones(n_pub_fit), np.full(len(aux_ok), W_AUX)])
            m.fit(X_fit, y, sample_weight=w)
        else:
            m.fit(X_fit[:n_pub_fit], Y_pub[:, hidx])
        meta_all[hold_idx, hidx] = m.predict(X_hold)
    pc_hold = np.exp(np.clip(meta_all[hold_idx, 3:6], -50.0, 50.0))
    grid = np.linspace(0.0, 1.0, LUT_NODES)
    Xf_pub = X_fit[:n_pub_fit]
    for g, (a, b) in enumerate([(0, 1), (1, 2)]):
        ds = true_s[:, b] - true_s[:, a]
        dc_raw = true_c[:, b] - true_c[:, a]
        floor = max(float(np.quantile(dc_raw[fit_idx], 0.05)), 1e-9)
        dc = np.maximum(dc_raw, floor)
        eff = ds / dc
        r_fit = rankdata(eff[fit_idx], method="average") / max(len(fit_idx) - 1, 1)
        q = np.quantile(eff[fit_idx], grid)
        dchat = np.maximum(pc_hold[:, b] - pc_hold[:, a], floor)
        m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(Xf_pub, r_fit)
        eff_hat = np.interp(np.clip(m.predict(X_hold), 0.0, 1.0), grid, q)
        rank_gain[hold_idx, g] = eff_hat * dchat
    prod = LEGACY_W * legacy_rows[hold_idx] + (1 - LEGACY_W) * linear_hold
    fam_rows_hold = np.array([fam_mean[fam_names[i]] for i in hold_idx])
    prod = (1 - FAM_W) * prod + FAM_W * fam_rows_hold
    conf = np.clip(knn_hold[:, 6], 0.0, 1.0)[:, None] * CONF_SCALE
    prod = (1 - conf) * prod + conf * knn_hold[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
    prod_all[hold_idx] = prod
    print(f"{TAG} fold {fold} pub_fit={len(fit_idx)} aux_used={len(aux_ok) if USE_AUX else 0} "
          f"{time.perf_counter()-t0:.0f}s", flush=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


sd = [rmse(meta_all[:, j], targets[:, j]) for j in range(3)]
cd = [rmse(meta_all[:, 3 + j], targets[:, 3 + j]) for j in range(3)]
print(f"{TAG} meta OOF rmse score {sd[0]:.4f}/{sd[1]:.4f}/{sd[2]:.4f} logcost {cd[0]:.4f}/{cd[1]:.4f}/{cd[2]:.4f}", flush=True)
ld = [rmse(prod_all[:, j], targets[:, j]) for j in (0, 3)]
print(f"{TAG} prod OOF rmse light score {ld[0]:.4f} logcost {ld[1]:.4f}", flush=True)


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
