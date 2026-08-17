"""Cost-error diagnosis: where does the deployed cost prediction go wrong?

Decomposes true cost into input-token and output-token parts, checks how
predictable each is from the prompt, and locates residual mass by model /
family / length / prompt-token-per-char ratio.  Pure diagnosis, no EV.
"""

import math
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from ossp_router import learned_router, similarity
from ossp_router.heuristic import episode_text
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes

policy = load_bundled_policy()
inputs = load_input(HERE / "data/combined/inputs.json")
outcomes = load_outcomes(HERE / "data/combined/outcomes.json")
artifact = learned_router.load_artifact(HERE / "src/ossp_router/resources/learned-router.v1.json")
episodes = inputs.episodes
n = len(episodes)
texts = [episode_text(e) for e in episodes]
index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}

unit = float(policy.token_unit)
rates = {m: (float(policy.models[m].input_token_rate), float(policy.models[m].output_token_rate)) for m in MODEL_IDS}
in_tok = np.array([[index[(e.episode_id, m)].input_tokens for m in MODEL_IDS] for e in episodes], dtype=float)
out_tok = np.array([[index[(e.episode_id, m)].output_tokens for m in MODEL_IDS] for e in episodes], dtype=float)
score = np.array([[float(index[(e.episode_id, m)].score) for m in MODEL_IDS] for e in episodes])
cost = np.array([[(rates[m][0] * in_tok[i, k] + rates[m][1] * out_tok[i, k]) / unit for k, m in enumerate(MODEL_IDS)] for i in range(n)])
chars = np.array([len(t) for t in texts], dtype=float)
fam = [similarity.classify_family(t) for t in texts]

print("[diag] === decomposition of true cost ===")
for k, m in enumerate(MODEL_IDS):
    in_part = rates[m][0] * in_tok[:, k] / unit
    out_part = rates[m][1] * out_tok[:, k] / unit
    print(f"[diag] {m:12s} mean cost {cost[:, k].mean():.4f} | input share {in_part.sum()/cost[:, k].sum():.2%} output share {out_part.sum()/cost[:, k].sum():.2%} "
          f"| out_tok mean {out_tok[:, k].mean():.0f} median {np.median(out_tok[:, k]):.0f} p95 {np.percentile(out_tok[:, k], 95):.0f} max {out_tok[:, k].max():.0f}")

print("[diag] === input tokens: same across models? predictable from chars? ===")
print(f"[diag] in_tok identical across models: {np.allclose(in_tok[:, 0], in_tok[:, 1]) and np.allclose(in_tok[:, 0], in_tok[:, 2])}")
for k, m in enumerate(MODEL_IDS):
    r = np.corrcoef(np.log1p(chars), np.log1p(in_tok[:, k]))[0, 1]
    ratio = in_tok[:, k] / np.maximum(chars, 1)
    print(f"[diag] {m:12s} corr(log chars, log in_tok) = {r:.4f} | tok/char mean {ratio.mean():.3f} sd {ratio.std():.3f}")

# per-family token/char ratio (Korean vs code vs math tokenize very differently)
print("[diag] === tok/char by family (model 0) ===")
by_f = defaultdict(list)
for i in range(n):
    by_f[fam[i]].append(in_tok[i, 0] / max(chars[i], 1))
for f, v in sorted(by_f.items(), key=lambda kv: -len(kv[1])):
    print(f"[diag]   {f:22s} n={len(v):4d} tok/char {np.mean(v):.3f} sd {np.std(v):.3f}")

print("[diag] === output tokens: variance structure ===")
lo = np.log1p(out_tok)
for k, m in enumerate(MODEL_IDS):
    print(f"[diag] {m:12s} log1p(out_tok) sd {lo[:, k].std():.3f} | corr with log chars {np.corrcoef(np.log1p(chars), lo[:, k])[0,1]:.3f} "
          f"| corr with own score {np.corrcoef(score[:, k], lo[:, k])[0,1]:.3f}")
print(f"[diag] corr(log out light, log out think) = {np.corrcoef(lo[:,0], lo[:,2])[0,1]:.3f}; (light, mid) {np.corrcoef(lo[:,0], lo[:,1])[0,1]:.3f}")

# what if we knew input tokens exactly and only mispredict output? (deployed-style OOF ridge on log cost vs log out)
print("[diag] === OOF ridge: predict log(cost) directly vs log(out_tok) + exact input ===")
from scipy import sparse
from sklearn.linear_model import Ridge
srows, scols, svals = [], [], []
for ri, episode in enumerate(episodes):
    dense = learned_router.raw_dense_features(episode)
    items = learned_router.feature_items(episode, word_hash_bins=artifact.word_hash_bins, char_hash_bins=artifact.char_hash_bins,
                                         dense_mean=artifact.dense_mean, dense_scale=artifact.dense_scale, raw_dense=dense)
    for c, v in items.items():
        srows.append(ri); scols.append(c); svals.append(v)
dim = len(learned_router.DENSE_FEATURE_NAMES) + artifact.word_hash_bins + artifact.char_hash_bins
X = sparse.csr_matrix((svals, (srows, scols)), shape=(n, dim))
fold = np.random.default_rng(123).integers(0, 5, size=n)
oof_cost = np.zeros((n, 3)); oof_out = np.zeros((n, 3)); oof_in = np.zeros((n, 3))
for f in range(5):
    h = fold == f
    Ridge(alpha=30.0, solver="sparse_cg").fit(X[~h], np.log(cost[~h]));
    m1 = Ridge(alpha=30.0, solver="sparse_cg").fit(X[~h], np.log(cost[~h])); oof_cost[h] = m1.predict(X[h])
    m2 = Ridge(alpha=30.0, solver="sparse_cg").fit(X[~h], np.log1p(out_tok[~h])); oof_out[h] = m2.predict(X[h])
    m3 = Ridge(alpha=30.0, solver="sparse_cg").fit(X[~h], np.log1p(in_tok[~h])); oof_in[h] = m3.predict(X[h])
for k, m in enumerate(MODEL_IDS):
    direct = np.log(cost[:, k]) - oof_cost[:, k]
    # reconstruct cost from predicted out + exact input
    recon_exact_in = np.log((rates[m][0] * in_tok[:, k] + rates[m][1] * np.expm1(np.maximum(oof_out[:, k], 0))) / unit)
    recon_pred_in = np.log((rates[m][0] * np.expm1(np.maximum(oof_in[:, k], 0)) + rates[m][1] * np.expm1(np.maximum(oof_out[:, k], 0))) / unit)
    e1 = np.log(cost[:, k]) - recon_exact_in
    e2 = np.log(cost[:, k]) - recon_pred_in
    print(f"[diag] {m:12s} RMSE log-cost: direct {np.sqrt((direct**2).mean()):.3f} | out-pred + exact-in {np.sqrt((e1**2).mean()):.3f} | out-pred + in-pred {np.sqrt((e2**2).mean()):.3f} "
          f"| in_tok RMSE(log1p) {np.sqrt(((np.log1p(in_tok[:,k])-oof_in[:,k])**2).mean()):.3f}")

print("[diag] === where is residual mass? |direct resid| by family / length quintile (think) ===")
resid = np.abs(np.log(cost[:, 2]) - oof_cost[:, 2])
by_f = defaultdict(list)
for i in range(n): by_f[fam[i]].append(resid[i])
for f, v in sorted(by_f.items(), key=lambda kv: -np.mean(kv[1])):
    print(f"[diag]   {f:22s} n={len(v):4d} mean|resid| {np.mean(v):.3f}")
q = np.quantile(chars, [0.2, 0.4, 0.6, 0.8])
bins = np.digitize(chars, q)
for b in range(5):
    print(f"[diag]   len quintile {b}: mean|resid| {resid[bins==b].mean():.3f}  mean out_tok think {out_tok[bins==b,2].mean():.0f}")
print("[diag] === budget-relevant: sum-level error of predicted vs true think cost on 880 bootstraps ===")
rng = np.random.default_rng(7)
ratios = []
for _ in range(400):
    s = rng.integers(0, n, size=880)
    ratios.append(np.exp(oof_cost[s, 2]).sum() / cost[s, 2].sum())
ratios = np.array(ratios)
print(f"[diag] pred/true think total: mean {ratios.mean():.3f} sd {ratios.std():.3f} p5 {np.percentile(ratios,5):.3f} p95 {np.percentile(ratios,95):.3f}  (exp(mean log) underestimates: Jensen)")
print("[diag] DONE")
