"""E35: pure IPR reproduction (Sivasubramanian et al., arXiv 2509.06274).

Faithful transplant of the paper's recipe, standing alone (no ensemble, no
Lagrangian, no cost heads):

  encoder  : shared representation of the prompt.  The paper uses a small
             transformer encoder; under the stdlib-runtime constraint we use
             our text featurizer (dense 30 + 16,384 hashed n-gram bins) fed
             through a shared linear+ReLU projection trained end-to-end.
  heads    : one logistic head per model, q_k(x) = sigmoid(h_k(E(x))),
             label = acceptable (score >= 0.5).
  loss     : BCE(acceptability) + lambda * pairwise ranking loss
             sum_{k>j} log sigmoid(z_k - z_j) over pairs where model k is
             acceptable and j is not (paper Sec 2.2).
  calib    : per-head temperature scaling on a 20% held split (Sec 3).
  select   : cheapest model with q_k >= theta; fallback argmax q; theta
             re-solved per batch so the batch cost fits the budget (Sec
             2.3/2.4).  Costs = per-model train-mean cost (Bedrock-style
             fixed price per model, as in the paper's setting).

Evaluated with the same 5-fold nested CV + 880-bootstrap EV harness as every
other experiment.  Variants: lambda in {0, 0.5, 1.0}; with/without
temperature; and, as a bridge, IPR heads + our Lagrangian allocator with
predicted per-episode costs (to isolate "predictor" vs "allocator").
"""

import math
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from ossp_router import learned_router
from ossp_router.protocol import MODEL_IDS, load_bundled_policy, load_input, load_outcomes

policy = load_bundled_policy()
inputs = load_input(HERE / "data/combined/inputs.json")
outcomes = load_outcomes(HERE / "data/combined/outcomes.json")
artifact = learned_router.load_artifact(HERE / "src/ossp_router/resources/learned-router.v1.json")

episodes = inputs.episodes
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
accept = (true_s >= 0.5).astype(np.float32)
print(f"[e35] acceptability rates: {accept.mean(axis=0).round(3).tolist()}", flush=True)

print("[e35] featurizing", flush=True)
from scipy import sparse

srows, scols, svals = [], [], []
for ri, episode in enumerate(episodes):
    dense = learned_router.raw_dense_features(episode)
    items = learned_router.feature_items(
        episode, word_hash_bins=artifact.word_hash_bins, char_hash_bins=artifact.char_hash_bins,
        dense_mean=artifact.dense_mean, dense_scale=artifact.dense_scale, raw_dense=dense)
    for c, v in items.items():
        srows.append(ri); scols.append(c); svals.append(v)
dim = len(learned_router.DENSE_FEATURE_NAMES) + artifact.word_hash_bins + artifact.char_hash_bins
X = sparse.csr_matrix((svals, (srows, scols)), shape=(n, dim), dtype=np.float32)

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(11)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN = 128
EPOCHS = 60
LR = 2e-3
WD = 1e-4
LAMBDAS = (0.0, 0.5, 1.0)


def to_torch(csr):
    coo = csr.tocoo()
    idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(coo.data), size=coo.shape).coalesce().to(DEVICE)


class IPR(nn.Module):
    def __init__(self, d_in, hidden, k):
        super().__init__()
        self.enc = nn.Linear(d_in, hidden)      # shared encoder (paper: transformer)
        self.heads = nn.Linear(hidden, k)       # model-specific heads

    def forward(self, xs):
        h = F.relu(torch.sparse.mm(xs, self.enc.weight.t()) + self.enc.bias)
        h = F.dropout(h, 0.2, self.training)
        return self.heads(h)                    # logits z_k


def train_ipr(Xfit, yfit, lam):
    model = IPR(Xfit.shape[1], HIDDEN, 3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    xs = to_torch(Xfit)
    y = torch.tensor(yfit, device=DEVICE)
    for _ in range(EPOCHS):
        model.train()
        z = model(xs)
        loss = F.binary_cross_entropy_with_logits(z, y)
        if lam > 0:
            # pairwise ranking: for pairs (k acceptable, j not) push z_k > z_j
            rank = 0.0; cnt = 0
            for k in range(3):
                for j in range(3):
                    if k == j:
                        continue
                    mask = (y[:, k] == 1) & (y[:, j] == 0)
                    if mask.any():
                        rank = rank - F.logsigmoid(z[mask, k] - z[mask, j]).sum()
                        cnt += int(mask.sum())
            if cnt:
                loss = loss + lam * rank / cnt
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def logits_of(model, Xs):
    with torch.no_grad():
        return model(to_torch(Xs)).cpu().numpy()


def fit_temperature(z, y):
    best_t, best = 1.0, np.inf
    for t in np.linspace(0.5, 3.0, 26):
        p = np.clip(1 / (1 + np.exp(-z / t)), 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best:
            best_t, best = t, nll
    return best_t


rng = np.random.default_rng(123)
fold_of = rng.integers(0, 5, size=n)
Z = {lam: np.zeros((n, 3)) for lam in LAMBDAS}
T = {lam: np.ones((n, 3)) for lam in LAMBDAS}
mean_cost = np.zeros((n, 3))   # per-fold train-mean cost, broadcast to hold rows

for fold in range(5):
    t0 = time.perf_counter()
    hold = fold_of == fold
    fit_idx = np.where(~hold)[0]; hold_idx = np.where(hold)[0]
    mean_cost[hold_idx] = true_c[fit_idx].mean(axis=0)
    cal = np.random.default_rng(100 + fold).random(len(fit_idx)) < 0.2
    tr_idx, cal_idx = fit_idx[~cal], fit_idx[cal]
    for lam in LAMBDAS:
        # full-fit model for hold predictions
        m = train_ipr(X[fit_idx], accept[fit_idx], lam)
        Z[lam][hold_idx] = logits_of(m, X[hold_idx])
        # calibration model on 80%, temperature on 20%
        mc = train_ipr(X[tr_idx], accept[tr_idx], lam)
        zc = logits_of(mc, X[cal_idx])
        for k in range(3):
            T[lam][hold_idx, k] = fit_temperature(zc[:, k], accept[cal_idx, k])
    print(f"[e35] fold {fold} {time.perf_counter()-t0:.0f}s", flush=True)

for lam in LAMBDAS:
    q = 1 / (1 + np.exp(-Z[lam]))
    auc = []
    for k in range(3):
        from sklearn.metrics import roc_auc_score
        auc.append(roc_auc_score(accept[:, k], q[:, k]))
    print(f"[e35] lambda={lam}: OOF AUC per model {np.round(auc, 3).tolist()}, mean temp {T[lam].mean(axis=0).round(2).tolist()}", flush=True)


# ---------------------------------------------------------------- selection
def select_ipr(q, cost, mult, safety):
    """Paper Sec 2.3: cheapest model with q >= theta, fallback argmax; theta
    bisected per batch (Sec 2.4) so batch cost <= budget*safety."""
    lt = cost[:, 0].sum(); cap = lt * max(1.0, mult * safety)
    argmax_pick = np.argmax(q + np.array([2e-12, 1e-12, 0.0]), axis=1)

    def choose(theta):
        ok = q >= theta
        pick = np.where(ok[:, 0], 0, np.where(ok[:, 1], 1, np.where(ok[:, 2], 2, argmax_pick)))
        return pick, cost[np.arange(len(pick)), pick].sum()

    lo, hi = 0.0, 1.0
    best, tot = choose(lo)
    if tot > cap:
        return np.zeros(len(q), dtype=int)
    for _ in range(40):
        mid = (lo + hi) / 2
        p, t = choose(mid)
        if t <= cap:
            lo, best = mid, p
        else:
            hi = mid
    return best


def select_lagrange(q, cost, mult, safety):
    lt = cost[:, 0].sum(); cap = lt * max(1.0, mult * safety)

    def choose(pen):
        u = q - pen * cost / lt
        pick = np.argmax(u + np.array([2e-12, 1e-12, 0.0]), axis=1)
        return pick, cost[np.arange(len(pick)), pick].sum()

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
        pick = np.zeros(len(q), dtype=int)
    return pick


rng2 = np.random.default_rng(7)
samples = [rng2.integers(0, n, size=880) for _ in range(400)]
MULTS = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
GRIDS = {"fast": np.arange(0.80, 1.0, 0.01), "balanced": np.arange(0.70, 0.96, 0.01),
         "premium": np.arange(0.60, 0.95, 0.01)}
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}


def evaluate(label, q, cost, selector):
    tot_ev = tot_sc = 0.0
    parts = []
    for tier, mult in MULTS.items():
        best = None
        for s in GRIDS[tier]:
            evs = []
            for sample in samples:
                p = selector(q[sample], cost[sample], mult, s)
                r = np.arange(len(sample))
                ratio = true_c[sample][r, p].sum() / true_c[sample][:, 0].sum()
                evs.append(0.0 if ratio > mult else true_s[sample][r, p].mean())
            ev = float(np.mean(evs))
            if best is None or ev > best[0]:
                pick = selector(q, cost, mult, s)
                best = (ev, s, float(true_s[np.arange(n), pick].mean()))
        tot_ev += W[tier] * best[0]; tot_sc += W[tier] * best[2]
        parts.append(f"{tier} {best[0]:.4f}@{best[1]:.2f}")
    print(f"[e35] {label}: weighted EV {tot_ev:.4f} score {tot_sc:.4f} | {'; '.join(parts)}", flush=True)
    return tot_ev


results = {}
for lam in LAMBDAS:
    for use_t in (False, True):
        q = 1 / (1 + np.exp(-Z[lam] / (T[lam] if use_t else 1.0)))
        tag = f"IPR lam={lam}{' +temp' if use_t else ''}"
        results[tag] = evaluate(f"{tag} | paper selector, fixed model cost", q, mean_cost, select_ipr)
# bridge: same predictor, our allocator with fixed cost
q_best = None
best_tag = max(results, key=results.get)
lam_b = float(best_tag.split("lam=")[1].split(" ")[0]); temp_b = "+temp" in best_tag
q_best = 1 / (1 + np.exp(-Z[lam_b] / (T[lam_b] if temp_b else 1.0)))
results["bridge:lagrange"] = evaluate(f"bridge {best_tag} + Lagrangian, fixed model cost", q_best, mean_cost, select_lagrange)
# bridge 2: paper selector with TRUE per-episode cost (upper bound of cost knowledge)
results["bridge:oracle-cost"] = evaluate(f"bridge {best_tag} + paper selector, oracle cost", q_best, true_c, select_ipr)
best = max(results.items(), key=lambda kv: kv[1])
print(f"[e35] BEST {best[0]} EV {best[1]:.4f} | deployed reference 0.6982 | baseline 0.6954", flush=True)
print("[e35] DONE", flush=True)
