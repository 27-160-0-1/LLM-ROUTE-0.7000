"""E43: joint hyper-parameter sweep of the deployed stack (E27 harness semantics).

The deployed constants (ridge alpha, GBM params, LEGACY_W, FAM_W, CONF_SCALE, GAIN_ALPHA, RANK_BETA,
per-tier blends) were tuned one at a time. This script
  1. computes features once,
  2. for each *expensive* combo (ridge alpha x GBM setting) runs the 5-fold nested CV once and caches
     the fold outputs (linear_hold, meta heads, rank heads, kNN rows, family means, legacy rows),
  3. for the best expensive combo (and the deployed one) runs a coordinate-descent search over the
     *cheap* post-hoc parameters, each evaluated with the 880xNBOOT bootstrap allocation harness,
  4. re-evaluates the top candidates with seeds 7/17/23 and NBOOT=400 (adoption rule: 3-seed mean
     +0.0015 over the deployed config, unimodal along each coordinate).

Usage: python e43_joint_sweep.py [OUTDIR] [NBOOT_SWEEP]      (defaults: reports/e43, 200)
Results: OUTDIR/results.jsonl (one line per evaluation), OUTDIR/summary.txt
"""

import itertools
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
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports/e43"
NBOOT_SWEEP = int(sys.argv[2]) if len(sys.argv) > 2 else 200
import os
SMOKE = os.environ.get("E43_SMOKE") == "1"   # quick end-to-end check: 2 expensive combos, tiny bootstrap
OUT.mkdir(parents=True, exist_ok=True)
RES = OUT / "results.jsonl"
TAG = "[e43]"

DEPLOYED = dict(legacy_w=0.75, fam_w=0.3, conf_scale=0.4, gain_alpha=0.5, rank_beta=0.25,
                blend_fast=0.6, blend_balanced=0.3, blend_premium=0.45)
DEPLOYED_EXP = dict(ridge_alpha=30.0, gbm_lr=0.06, gbm_leaves=15, gbm_min_leaf=30, gbm_l2=3.0, gbm_iter=300)
LUT_NODES = 65
CHEAP_GRID = {
    "legacy_w": [0.6, 0.75, 0.9],
    "fam_w": [0.15, 0.3, 0.45],
    "conf_scale": [0.25, 0.4, 0.6],
    "gain_alpha": [0.3, 0.5, 0.7],
    "rank_beta": [0.1, 0.25, 0.4],
    "blend_fast": [0.45, 0.6, 0.75],
    "blend_balanced": [0.15, 0.3, 0.45],
    "blend_premium": [0.3, 0.45, 0.6],
}
EXPENSIVE_GRID = [
    DEPLOYED_EXP,
    dict(DEPLOYED_EXP, ridge_alpha=10.0),
    dict(DEPLOYED_EXP, ridge_alpha=100.0),
    dict(DEPLOYED_EXP, gbm_lr=0.03, gbm_iter=600),
    dict(DEPLOYED_EXP, gbm_lr=0.1),
    dict(DEPLOYED_EXP, gbm_leaves=31),
    dict(DEPLOYED_EXP, gbm_leaves=7),
    dict(DEPLOYED_EXP, gbm_min_leaf=15),
    dict(DEPLOYED_EXP, gbm_min_leaf=60),
    dict(DEPLOYED_EXP, gbm_l2=1.0),
    dict(DEPLOYED_EXP, gbm_l2=10.0),
]

if SMOKE:
    EXPENSIVE_GRID = EXPENSIVE_GRID[:2]
    NBOOT_SWEEP = 10
    for k in CHEAP_GRID:
        CHEAP_GRID[k] = CHEAP_GRID[k][:2]
similarity.NEIGHBORS = 16
policy = load_bundled_policy()
inputs = load_input(ROOT / "data/combined/inputs.json")
outcomes = load_outcomes(ROOT / "data/combined/outcomes.json")
artifact = learned_router.load_artifact(ROOT / "src/ossp_router/resources/learned-router.v1.json")
legacy_artifact = legacy_hash_regex.load_artifact(ROOT / "src/ossp_router/resources/hash-regex-public.v1.json")

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

from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

t0 = time.perf_counter()
texts = [episode_text(e) for e in episodes]
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
print(f"{TAG} features for {n} rows in {time.perf_counter()-t0:.0f}s", flush=True)

rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)

# ---- kNN + family means per fold are parameter-free (k fixed at 16): compute once ----
knn_hold_all = np.zeros((n, 7))
knn_fit_by_fold = {}
fam_rows_hold_all = np.zeros((n, 6))
for fold in range(5):
    hold = fold_of == fold
    fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
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

    knn_fit_by_fold[fold] = np.array([kq(texts[i], exclude=k) for k, i in enumerate(fit_idx)])
    knn_hold_all[hold_idx] = np.array([kq(texts[i]) for i in hold_idx])
    fam_mean = {}
    by_family = defaultdict(list)
    for i in fit_idx:
        by_family[fam_names[i]].append(targets[i])
    fglobal = targets[fit_idx].mean(axis=0)
    for name in FAMILIES:
        rows = by_family.get(name, [])
        fam_mean[name] = np.mean(rows, axis=0) if len(rows) >= 8 else fglobal
    fam_rows_hold_all[hold_idx] = np.array([fam_mean[fam_names[i]] for i in hold_idx])
print(f"{TAG} kNN/family per fold done {time.perf_counter()-t0:.0f}s", flush=True)


def run_expensive(exp):
    """5-fold nested CV for one (ridge alpha, GBM) setting -> arrays needed by the cheap evaluation."""
    gbm_params = dict(max_iter=exp["gbm_iter"], learning_rate=exp["gbm_lr"], max_leaf_nodes=exp["gbm_leaves"],
                      min_samples_leaf=exp["gbm_min_leaf"], l2_regularization=exp["gbm_l2"],
                      early_stopping=True, validation_fraction=0.15, random_state=11)
    linear_all = np.zeros((n, 6))
    meta_all = np.zeros((n, 8))
    rank_eff = np.zeros((n, 2))       # eff_hat (before multiplying by predicted cost delta)
    floors = np.zeros((n, 2))
    for fold in range(5):
        hold = fold_of == fold
        fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
        ridge = Ridge(alpha=exp["ridge_alpha"], solver="sparse_cg")
        ridge.fit(X_sparse[fit_idx], targets[fit_idx])
        linear_hold = ridge.predict(X_sparse[hold_idx])
        linear_hold[:, :3] = np.clip(linear_hold[:, :3], 0.0, 1.0)
        inner_fold = np.random.default_rng(fold).integers(0, 5, size=len(fit_idx))
        inner_oof = np.zeros((len(fit_idx), 6))
        for inner in range(5):
            tr = fit_idx[inner_fold != inner]; te = inner_fold == inner
            m = Ridge(alpha=exp["ridge_alpha"], solver="sparse_cg").fit(X_sparse[tr], targets[tr])
            inner_oof[te] = m.predict(X_sparse[fit_idx[te]])
        inner_oof[:, :3] = np.clip(inner_oof[:, :3], 0.0, 1.0)
        X_fit = np.hstack([dense_rows[fit_idx], fam_onehot[fit_idx], legacy_rows[fit_idx], inner_oof, knn_fit_by_fold[fold]])
        X_hold = np.hstack([dense_rows[hold_idx], fam_onehot[hold_idx], legacy_rows[hold_idx], linear_hold, knn_hold_all[hold_idx]])
        Y_fit = full_targets[fit_idx]
        for hidx in range(8):
            m = HistGradientBoostingRegressor(**gbm_params).fit(X_fit, Y_fit[:, hidx])
            meta_all[hold_idx, hidx] = m.predict(X_hold)
        grid = np.linspace(0.0, 1.0, LUT_NODES)
        for g, (a, b) in enumerate([(0, 1), (1, 2)]):
            ds = true_s[:, b] - true_s[:, a]
            dc_raw = true_c[:, b] - true_c[:, a]
            floor = max(float(np.quantile(dc_raw[fit_idx], 0.05)), 1e-9)
            eff = ds / np.maximum(dc_raw, floor)
            r_fit = rankdata(eff[fit_idx], method="average") / max(len(fit_idx) - 1, 1)
            q = np.quantile(eff[fit_idx], grid)
            m = HistGradientBoostingRegressor(**gbm_params).fit(X_fit, r_fit)
            rank_eff[hold_idx, g] = np.interp(np.clip(m.predict(X_hold), 0.0, 1.0), grid, q)
            floors[hold_idx, g] = floor
        linear_all[hold_idx] = linear_hold
    return dict(linear=linear_all, meta=meta_all, rank_eff=rank_eff, floors=floors)


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


MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
GRIDS = {"fast": np.arange(0.92, 1.0, 0.01), "balanced": np.arange(0.82, 0.94, 0.01),
         "premium": np.arange(0.80, 0.93, 0.01)}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
_samples_cache = {}


def samples_for(seed, nboot):
    key = (seed, nboot)
    if key not in _samples_cache:
        r = np.random.default_rng(seed)
        _samples_cache[key] = [r.integers(0, n, size=880) for _ in range(nboot)]
    return _samples_cache[key]


def evaluate(arr, cfg, seed=7, nboot=200, tiers=("fast", "balanced", "premium")):
    linear, meta_all, rank_eff, floors = arr["linear"], arr["meta"], arr["rank_eff"], arr["floors"]
    prod = cfg["legacy_w"] * legacy_rows + (1 - cfg["legacy_w"]) * linear
    prod = (1 - cfg["fam_w"]) * prod + cfg["fam_w"] * fam_rows_hold_all
    conf = np.clip(knn_hold_all[:, 6], 0.0, 1.0)[:, None] * cfg["conf_scale"]
    prod = (1 - conf) * prod + conf * knn_hold_all[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
    pc_meta = np.exp(np.clip(meta_all[:, 3:6], -50.0, 50.0))
    dchat = np.column_stack([np.maximum(pc_meta[:, 1] - pc_meta[:, 0], floors[:, 0]),
                             np.maximum(pc_meta[:, 2] - pc_meta[:, 1], floors[:, 1])])
    rank_gain = rank_eff * dchat
    mixed = (1 - cfg["rank_beta"]) * meta_all[:, 6:8] + cfg["rank_beta"] * rank_gain
    meta = meta_all[:, :6].copy()
    recon = np.column_stack([meta[:, 0], meta[:, 0] + mixed[:, 0], meta[:, 0] + mixed[:, 0] + mixed[:, 1]])
    meta[:, :3] = (1 - cfg["gain_alpha"]) * meta[:, :3] + cfg["gain_alpha"] * recon
    samples = samples_for(seed, nboot)
    out = {}
    for tier in tiers:
        mult = MULTS[tier]
        blend = cfg[f"blend_{tier}"]
        stacked = (1 - blend) * prod + blend * meta
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
                best = (ev, float(s))
        out[tier] = best
    ev = sum(W[t] * out[t][0] for t in out) if len(out) == 3 else None
    return ev, out


def log_result(kind, exp, cfg, seed, nboot, ev, out):
    rec = dict(kind=kind, exp=exp, cfg=cfg, seed=seed, nboot=nboot, ev=ev, tiers=out, t=time.time())
    with RES.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    tiers = " ".join(f"{t} {v[0]:.4f}@{v[1]:.2f}" for t, v in out.items())
    print(f"{TAG} {kind} s{seed} n{nboot} EV {ev if ev is None else round(ev,4)} | {tiers} | {json.dumps(cfg)} | {json.dumps(exp)}", flush=True)


# ---------------- stage 1: expensive combos with deployed cheap params ----------------
cache = {}
exp_scores = []
for exp in EXPENSIVE_GRID:
    key = json.dumps(exp, sort_keys=True)
    t1 = time.perf_counter()
    cache[key] = run_expensive(exp)
    ev, out = evaluate(cache[key], DEPLOYED, 7, NBOOT_SWEEP)
    log_result("expensive", exp, DEPLOYED, 7, NBOOT_SWEEP, ev, out)
    exp_scores.append((ev, key))
    print(f"{TAG}   ({time.perf_counter()-t1:.0f}s)", flush=True)
exp_scores.sort(reverse=True)
base_key = json.dumps(DEPLOYED_EXP, sort_keys=True)
best_exp_key = exp_scores[0][1]
print(f"{TAG} best expensive: {best_exp_key} EV {exp_scores[0][0]:.4f} (deployed {dict((k, e) for e, k in exp_scores)[base_key]:.4f})", flush=True)

# ---------------- stage 2: coordinate descent over cheap params (2 rounds), for deployed and best expensive ----------------
def coordinate_descent(arr, exp, start, rounds=1 if SMOKE else 2):
    cfg = dict(start)
    best_ev, out = evaluate(arr, cfg, 7, NBOOT_SWEEP)
    log_result("cd-start", exp, cfg, 7, NBOOT_SWEEP, best_ev, out)
    for rd in range(rounds):
        improved = False
        for name, values in CHEAP_GRID.items():
            tiers = (name.split("_")[1],) if name.startswith("blend_") else ("fast", "balanced", "premium")
            trial = []
            for v in values:
                c2 = dict(cfg); c2[name] = v
                if name.startswith("blend_"):
                    # only that tier changes: reuse the other tiers' scores
                    _, o1 = evaluate(arr, c2, 7, NBOOT_SWEEP, tiers=tiers)
                    o = dict(out); o.update(o1)
                    ev = sum(W[t] * o[t][0] for t in o)
                else:
                    ev, o = evaluate(arr, c2, 7, NBOOT_SWEEP)
                log_result(f"cd{rd}:{name}", exp, c2, 7, NBOOT_SWEEP, ev, o)
                trial.append((ev, v, o))
            evs = [t[0] for t in trial]
            top = max(trial, key=lambda t: t[0])
            unimodal = evs == sorted(evs) or evs == sorted(evs, reverse=True) or (evs[1] >= evs[0] and evs[1] >= evs[2])
            if top[0] > best_ev + 1e-6 and unimodal:
                cfg[name] = top[1]; best_ev = top[0]; out = top[2]; improved = True
        print(f"{TAG} round {rd} best EV {best_ev:.4f} cfg {json.dumps(cfg)}", flush=True)
        if not improved:
            break
    return cfg, best_ev


candidates = []
for key in dict.fromkeys([base_key, best_exp_key]):
    exp = json.loads(key)
    cfg, ev = coordinate_descent(cache[key], exp, DEPLOYED)
    candidates.append((ev, key, cfg))
candidates.sort(reverse=True)

# ---------------- stage 3: confirmation with 3 seeds x 400 bootstrap ----------------
summary = []
for label, key, cfg in [("deployed", base_key, DEPLOYED)] + [(f"cand{i}", k, c) for i, (_, k, c) in enumerate(candidates)]:
    evs = []
    for seed in ((7,) if SMOKE else (7, 17, 23)):
        ev, out = evaluate(cache[key], cfg, seed, 20 if SMOKE else 400)
        log_result(f"confirm:{label}", json.loads(key), cfg, seed, 400, ev, out)
        evs.append(ev)
    summary.append((label, float(np.mean(evs)), evs, key, cfg))
    print(f"{TAG} CONFIRM {label} mean {np.mean(evs):.4f} seeds {[round(e,4) for e in evs]} | {json.dumps(cfg)} | {key}", flush=True)
dep_mean = summary[0][1]
lines = [f"deployed 3-seed mean {dep_mean:.4f}"]
for label, mean, evs, key, cfg in summary[1:]:
    verdict = "ADOPT-CANDIDATE" if mean - dep_mean >= 0.0015 else "noise/reject"
    lines.append(f"{label}: mean {mean:.4f} ({mean-dep_mean:+.4f}) seeds {[round(e,4) for e in evs]} -> {verdict}\n   cfg {json.dumps(cfg)}\n   exp {key}")
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"{TAG} SUMMARY\n" + "\n".join(lines), flush=True)
print(f"{TAG} DONE", flush=True)
