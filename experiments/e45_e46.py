"""E45 + E46 in one run (E43 harness semantics; Colab CPU, ~3h).

E45  refined joint search around the E43 optimum (cand0):
       expensive: ridge alpha {5,10,15} x kNN k {8,16,24}   (GBM params fixed: E43 showed no gain)
       cheap    : legacy_w / fam_w / conf_scale (shared) + gain_alpha, rank_beta, blend PER TIER,
                  coordinate descent 2 rounds, 880 x NBOOT bootstrap, unimodal-only moves
E46  selection-weighted cost heads: the E42 diagnosis was that items picked for an upgrade have
       under-predicted costs.  Here the mid/think log-cost heads are refit with sample weights
       1 + lambda * u, where u is the fold-internal (inner-OOF linear) rank of the predicted
       efficiency of that upgrade (mid-vs-light for the mid cost head, think-vs-mid for think).
       lambda in {0.5, 1, 2}; evaluated on top of the E45 winner.
Confirmation: deployed-E43 cand0 vs E45 winner vs E46 winner, seeds 7/17/23 x 400 on the standard
       fold split, plus an ALTERNATE fold split (rng 456) for the finalists to expose selection bias.
Adoption rule: >= +0.0015 vs cand0 on the 3-seed mean AND positive on the alternate split.

Usage: python e45_e46.py [OUTDIR] [NBOOT_SWEEP]        (defaults reports/e45, 200)
Results: OUTDIR/results.jsonl, OUTDIR/summary.txt
"""

import json
import math
import os
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

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports/e45"
NBOOT_SWEEP = int(sys.argv[2]) if len(sys.argv) > 2 else 200
SMOKE = os.environ.get("E45_SMOKE") == "1"
OUT.mkdir(parents=True, exist_ok=True)
RES = OUT / "results.jsonl"
TAG = "[e45]"
TIERS3 = ("fast", "balanced", "premium")

# E43 cand0 (deployed since 2026-08-19), expressed with per-tier copies of gain_alpha / rank_beta
CAND0 = dict(legacy_w=0.9, fam_w=0.15, conf_scale=0.25,
             gain_alpha_fast=0.5, gain_alpha_balanced=0.5, gain_alpha_premium=0.5,
             rank_beta_fast=0.4, rank_beta_balanced=0.4, rank_beta_premium=0.4,
             blend_fast=0.6, blend_balanced=0.45, blend_premium=0.3)
CAND0_EXP = dict(ridge_alpha=10.0, knn_k=16, sel_lambda=0.0)
GBM_PARAMS = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=30,
                  l2_regularization=3.0, early_stopping=True, validation_fraction=0.15, random_state=11)
LUT_NODES = 65
KNN_KS = (8, 16, 24)
KNN_MAX = max(KNN_KS)
CHEAP_GRID = {
    "legacy_w": [0.85, 0.9, 0.95],
    "fam_w": [0.1, 0.15, 0.2],
    "conf_scale": [0.2, 0.25, 0.3],
    "gain_alpha_fast": [0.35, 0.5, 0.65], "gain_alpha_balanced": [0.35, 0.5, 0.65], "gain_alpha_premium": [0.35, 0.5, 0.65],
    "rank_beta_fast": [0.3, 0.4, 0.5], "rank_beta_balanced": [0.3, 0.4, 0.5], "rank_beta_premium": [0.3, 0.4, 0.5],
    "blend_fast": [0.5, 0.6, 0.7], "blend_balanced": [0.4, 0.45, 0.5], "blend_premium": [0.25, 0.3, 0.35],
}
E45_EXPENSIVE = [dict(ridge_alpha=a, knn_k=k, sel_lambda=0.0) for a in (5.0, 10.0, 15.0) for k in KNN_KS]
E46_LAMBDAS = (0.5, 1.0, 2.0)
if SMOKE:
    E45_EXPENSIVE = E45_EXPENSIVE[:2]
    E46_LAMBDAS = (1.0,)
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


class Split:
    """Everything that depends on the fold assignment: kNN candidate lists (top KNN_MAX) and family means."""

    def __init__(self, seed):
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.fold_of = rng.integers(0, 5, size=n)
        self.cand_fit = {}     # fold -> list (per fit row) of [(doc_pos, sim)...] top KNN_MAX, self excluded
        self.cand_hold = {}    # fold -> list (per hold row)
        self.gmean = {}
        self.fam_rows_hold = np.zeros((n, 6))
        t1 = time.perf_counter()
        for fold in range(5):
            hold = self.fold_of == fold
            fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
            knn_texts = [texts[i] for i in fit_idx]
            freqs, total = similarity.document_frequencies(knn_texts)
            idf = similarity.idf_table(freqs, total)
            vecs = [similarity.tfidf_vector(t, idf, top_components=similarity.TOP_COMPONENTS) for t in knn_texts]
            knn_index = similarity.KnnIndex(vecs, targets[fit_idx].tolist())
            self.gmean[fold] = targets[fit_idx].mean(axis=0)

            def cands(text, exclude=None):
                q = similarity.tfidf_vector(text, idf)
                if not q:
                    return []
                scores = {}
                get = knn_index.postings.get
                for g, v in q.items():
                    for d, s in get(g, ()):
                        if exclude is not None and d == exclude:
                            continue
                        scores[d] = scores.get(d, 0.0) + v * s
                return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:KNN_MAX]

            self.cand_fit[fold] = [cands(texts[i], exclude=k) for k, i in enumerate(fit_idx)]
            self.cand_hold[fold] = [cands(texts[i]) for i in hold_idx]
            fam_mean = {}
            by_family = defaultdict(list)
            for i in fit_idx:
                by_family[fam_names[i]].append(targets[i])
            fglobal = targets[fit_idx].mean(axis=0)
            for name in FAMILIES:
                rows = by_family.get(name, [])
                fam_mean[name] = np.mean(rows, axis=0) if len(rows) >= 8 else fglobal
            self.fam_rows_hold[hold_idx] = np.array([fam_mean[fam_names[i]] for i in hold_idx])
        print(f"{TAG} split {seed}: kNN candidates + family means in {time.perf_counter()-t1:.0f}s", flush=True)
        self._knn_cache = {}

    def knn_rows(self, k):
        """(knn_fit_by_fold, knn_hold_all) for neighbour count k, cut from the cached candidate lists."""
        if k in self._knn_cache:
            return self._knn_cache[k]
        knn_hold_all = np.zeros((n, 7))
        knn_fit_by_fold = {}
        for fold in range(5):
            hold = self.fold_of == fold
            fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
            gmean = self.gmean[fold]

            def row_of(cand):
                ranked = cand[:k]
                if not ranked:
                    return np.concatenate([gmean, [0.0]])
                tot = sum(s for _d, s in ranked)
                row = np.zeros(6)
                for d, s in ranked:
                    row += (s / tot) * targets[fit_idx[d]]
                return np.concatenate([row, [ranked[0][1]]])

            knn_fit_by_fold[fold] = np.array([row_of(c) for c in self.cand_fit[fold]])
            knn_hold_all[hold_idx] = np.array([row_of(c) for c in self.cand_hold[fold]])
        self._knn_cache[k] = (knn_fit_by_fold, knn_hold_all)
        return self._knn_cache[k]


def run_expensive(split, exp):
    """5-fold nested CV for (ridge alpha, kNN k, selection weighting lambda)."""
    knn_fit_by_fold, knn_hold_all = split.knn_rows(exp["knn_k"])
    lam = exp.get("sel_lambda", 0.0)
    linear_all = np.zeros((n, 6)); meta_all = np.zeros((n, 8)); rank_eff = np.zeros((n, 2)); floors = np.zeros((n, 2))
    for fold in range(5):
        hold = split.fold_of == fold
        fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
        ridge = Ridge(alpha=exp["ridge_alpha"], solver="sparse_cg").fit(X_sparse[fit_idx], targets[fit_idx])
        linear_hold = ridge.predict(X_sparse[hold_idx]); linear_hold[:, :3] = np.clip(linear_hold[:, :3], 0.0, 1.0)
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
        # E46: selection weights for the mid / think cost heads from the inner-OOF linear predictions
        weights = {}
        if lam > 0:
            oof_c = np.exp(np.clip(inner_oof[:, 3:6], -50, 50))
            for head, (a, b) in ((4, (0, 1)), (5, (1, 2))):
                gain = inner_oof[:, b] - inner_oof[:, a]
                dc = np.maximum(oof_c[:, b] - oof_c[:, a], 1e-9)
                u = rankdata(gain / dc, method="average") / max(len(fit_idx) - 1, 1)   # 0..1, high = likely upgrade
                weights[head] = 1.0 + lam * u
        for hidx in range(8):
            m = HistGradientBoostingRegressor(**GBM_PARAMS)
            if hidx in weights:
                m.fit(X_fit, Y_fit[:, hidx], sample_weight=weights[hidx])
            else:
                m.fit(X_fit, Y_fit[:, hidx])
            meta_all[hold_idx, hidx] = m.predict(X_hold)
        grid = np.linspace(0.0, 1.0, LUT_NODES)
        for g, (a, b) in enumerate([(0, 1), (1, 2)]):
            ds = true_s[:, b] - true_s[:, a]; dc_raw = true_c[:, b] - true_c[:, a]
            floor = max(float(np.quantile(dc_raw[fit_idx], 0.05)), 1e-9)
            eff = ds / np.maximum(dc_raw, floor)
            r_fit = rankdata(eff[fit_idx], method="average") / max(len(fit_idx) - 1, 1)
            q = np.quantile(eff[fit_idx], grid)
            m = HistGradientBoostingRegressor(**GBM_PARAMS).fit(X_fit, r_fit)
            rank_eff[hold_idx, g] = np.interp(np.clip(m.predict(X_hold), 0.0, 1.0), grid, q)
            floors[hold_idx, g] = floor
        linear_all[hold_idx] = linear_hold
    return dict(linear=linear_all, meta=meta_all, rank_eff=rank_eff, floors=floors, knn_hold=knn_hold_all,
                fam_rows=split.fam_rows_hold)


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


def evaluate(arr, cfg, seed=7, nboot=200, tiers=TIERS3):
    linear, meta_all, rank_eff, floors = arr["linear"], arr["meta"], arr["rank_eff"], arr["floors"]
    knn_hold_all, fam_rows = arr["knn_hold"], arr["fam_rows"]
    prod = cfg["legacy_w"] * legacy_rows + (1 - cfg["legacy_w"]) * linear
    prod = (1 - cfg["fam_w"]) * prod + cfg["fam_w"] * fam_rows
    conf = np.clip(knn_hold_all[:, 6], 0.0, 1.0)[:, None] * cfg["conf_scale"]
    prod = (1 - conf) * prod + conf * knn_hold_all[:, :6]
    prod[:, :3] = np.clip(prod[:, :3], 0.0, 1.0)
    pc_meta = np.exp(np.clip(meta_all[:, 3:6], -50.0, 50.0))
    dchat = np.column_stack([np.maximum(pc_meta[:, 1] - pc_meta[:, 0], floors[:, 0]),
                             np.maximum(pc_meta[:, 2] - pc_meta[:, 1], floors[:, 1])])
    rank_gain = rank_eff * dchat
    samples = samples_for(seed, nboot)
    out = {}
    for tier in tiers:
        beta = cfg[f"rank_beta_{tier}"]; galpha = cfg[f"gain_alpha_{tier}"]; blend = cfg[f"blend_{tier}"]
        mixed = (1 - beta) * meta_all[:, 6:8] + beta * rank_gain
        meta = meta_all[:, :6].copy()
        recon = np.column_stack([meta[:, 0], meta[:, 0] + mixed[:, 0], meta[:, 0] + mixed[:, 0] + mixed[:, 1]])
        meta[:, :3] = (1 - galpha) * meta[:, :3] + galpha * recon
        mult = MULTS[tier]
        stacked = (1 - blend) * prod + blend * meta
        ps = np.clip(stacked[:, :3], 0, 1)
        pc = np.exp(np.clip(stacked[:, 3:], -50, 50))
        pc[:, 1] = np.maximum(pc[:, 1], pc[:, 0] * (1 + 1e-12)); pc[:, 2] = np.maximum(pc[:, 2], pc[:, 1] * (1 + 1e-12))
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


def log_result(kind, exp, cfg, seed, nboot, ev, out, split_seed=123):
    rec = dict(kind=kind, exp=exp, cfg=cfg, seed=seed, nboot=nboot, ev=ev, tiers=out, split=split_seed, t=time.time())
    with RES.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    tiers = " ".join(f"{t} {v[0]:.4f}@{v[1]:.2f}" for t, v in out.items())
    print(f"{TAG} {kind} split{split_seed} s{seed} n{nboot} EV {ev if ev is None else round(ev,4)} | {tiers} | {json.dumps(cfg)} | {json.dumps(exp)}", flush=True)


def tier_of(name):
    for t in TIERS3:
        if name.endswith("_" + t):
            return t
    return None


def coordinate_descent(arr, exp, start, rounds=1 if SMOKE else 2):
    cfg = dict(start)
    best_ev, out = evaluate(arr, cfg, 7, NBOOT_SWEEP)
    log_result("cd-start", exp, cfg, 7, NBOOT_SWEEP, best_ev, out)
    for rd in range(rounds):
        improved = False
        for name, values in CHEAP_GRID.items():
            t = tier_of(name)
            trial = []
            for v in values:
                c2 = dict(cfg); c2[name] = v
                if t is not None:
                    _, o1 = evaluate(arr, c2, 7, NBOOT_SWEEP, tiers=(t,))
                    o = dict(out); o.update(o1)
                    ev = sum(W[x] * o[x][0] for x in o)
                else:
                    ev, o = evaluate(arr, c2, 7, NBOOT_SWEEP)
                log_result(f"cd{rd}:{name}", exp, c2, 7, NBOOT_SWEEP, ev, o)
                trial.append((ev, v, o))
            evs = [x[0] for x in trial]
            top = max(trial, key=lambda x: x[0])
            unimodal = evs == sorted(evs) or evs == sorted(evs, reverse=True) or (len(evs) == 3 and evs[1] >= evs[0] and evs[1] >= evs[2])
            if top[0] > best_ev + 1e-6 and unimodal:
                cfg[name] = top[1]; best_ev = top[0]; out = top[2]; improved = True
        print(f"{TAG} round {rd} best EV {best_ev:.4f} cfg {json.dumps(cfg)}", flush=True)
        if not improved:
            break
    return cfg, best_ev


# ================= standard split =================
split = Split(123)
cache = {}


def get_arr(split_obj, exp):
    key = (split_obj.seed, json.dumps(exp, sort_keys=True))
    if key not in cache:
        t1 = time.perf_counter()
        cache[key] = run_expensive(split_obj, exp)
        print(f"{TAG}   expensive {json.dumps(exp)} on split {split_obj.seed}: {time.perf_counter()-t1:.0f}s", flush=True)
    return cache[key]


# ---- E45 stage 1: expensive grid under cand0 cheap params ----
scores = []
for exp in E45_EXPENSIVE:
    ev, out = evaluate(get_arr(split, exp), CAND0, 7, NBOOT_SWEEP)
    log_result("e45-expensive", exp, CAND0, 7, NBOOT_SWEEP, ev, out)
    scores.append((ev, json.dumps(exp, sort_keys=True)))
scores.sort(reverse=True)
best_exp = json.loads(scores[0][1])
base_key = json.dumps(CAND0_EXP, sort_keys=True)
print(f"{TAG} E45 best expensive {scores[0][1]} EV {scores[0][0]:.4f} (cand0 exp EV {dict((k, e) for e, k in scores).get(base_key)})", flush=True)

# ---- E45 stage 2: coordinate descent on best expensive and on cand0's expensive ----
cands = []
for exp in ({json.dumps(best_exp, sort_keys=True): best_exp, base_key: CAND0_EXP}).values():
    cfg, ev = coordinate_descent(get_arr(split, exp), exp, CAND0)
    cands.append((ev, exp, cfg))
cands.sort(key=lambda x: -x[0])
e45_ev, e45_exp, e45_cfg = cands[0]
print(f"{TAG} E45 winner EV {e45_ev:.4f} exp {json.dumps(e45_exp)} cfg {json.dumps(e45_cfg)}", flush=True)

# ---- E46: selection-weighted cost heads on top of the E45 winner ----
e46 = []
for lam in E46_LAMBDAS:
    exp = dict(e45_exp, sel_lambda=lam)
    ev, out = evaluate(get_arr(split, exp), e45_cfg, 7, NBOOT_SWEEP)
    log_result("e46-lambda", exp, e45_cfg, 7, NBOOT_SWEEP, ev, out)
    e46.append((ev, exp))
e46.sort(key=lambda x: -x[0])
e46_ev, e46_exp = e46[0]
print(f"{TAG} E46 best lambda {e46_exp['sel_lambda']} EV {e46_ev:.4f} (E45 winner {e45_ev:.4f})", flush=True)

# ---- confirmation: 3 seeds x 400 on the standard split ----
finalists = [("cand0", CAND0_EXP, CAND0), ("e45", e45_exp, e45_cfg), ("e46", e46_exp, e45_cfg)]
summary = []
for label, exp, cfg in finalists:
    evs = []
    for seed in ((7,) if SMOKE else (7, 17, 23)):
        ev, out = evaluate(get_arr(split, exp), cfg, seed, 20 if SMOKE else 400)
        log_result(f"confirm:{label}", exp, cfg, seed, 400, ev, out)
        evs.append(ev)
    summary.append([label, float(np.mean(evs)), evs, exp, cfg, None])
    print(f"{TAG} CONFIRM {label} mean {np.mean(evs):.4f} seeds {[round(e,4) for e in evs]}", flush=True)

# ---- alternate fold split (selection-bias check) ----
split2 = Split(456)
for row in summary:
    label, _, _, exp, cfg, _ = row
    ev, out = evaluate(get_arr(split2, exp), cfg, 7, 20 if SMOKE else 400)
    log_result(f"altsplit:{label}", exp, cfg, 7, 400, ev, out, split_seed=456)
    row[5] = ev
    print(f"{TAG} ALTSPLIT {label} EV {ev:.4f}", flush=True)

base_mean, base_alt = summary[0][1], summary[0][5]
lines = [f"cand0 (deployed E43): 3-seed mean {base_mean:.4f}, alt-split {base_alt:.4f}"]
for label, mean, evs, exp, cfg, alt in summary[1:]:
    d1, d2 = mean - base_mean, alt - base_alt
    verdict = "ADOPT-CANDIDATE" if (d1 >= 0.0015 and d2 > 0) else "noise/reject"
    lines.append(f"{label}: mean {mean:.4f} ({d1:+.4f}) seeds {[round(e,4) for e in evs]} | alt-split {alt:.4f} ({d2:+.4f}) -> {verdict}"
                 f"\n   exp {json.dumps(exp)}\n   cfg {json.dumps(cfg)}")
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"{TAG} SUMMARY\n" + "\n".join(lines), flush=True)
print(f"{TAG} DONE", flush=True)
