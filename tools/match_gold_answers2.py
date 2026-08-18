# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Second-pass gold matching for families the first pass missed:
ruletaker (context+question), truthfulqa (binary via mc2 correctness sets +
fuzzy), HRMCR (date/zodiac), babilong (longdoc, target answer).
Merges into data/gold/gold-answers.v1.json."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import similarity  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import load_input  # noqa: E402
from datasets import load_dataset  # noqa: E402

OUT = ROOT / "data/gold/gold-answers.v1.json"
gold = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


eps = []
for split in ("train", "dev"):
    aime = {e["episode_id"] for e in json.loads((ROOT / f"data/{split}/aime-selection.json").read_text(encoding="utf-8"))["episodes"]}
    for e in load_input(ROOT / f"data/materialized/{split}/inputs.json").episodes:
        if e.episode_id not in aime:
            eps.append((e.episode_id, episode_text(e)))
fam = {eid: similarity.classify_family(t) for eid, t in eps}
todo = [(eid, t) for eid, t in eps if eid not in gold]
print(f"[gold2] {len(todo)} unmatched; families {dict(Counter(fam[e] for e,_ in todo))}", flush=True)

# ---- RuleTaker: our prompt = context sentences + last line = question. Index by exact context.
ds = load_dataset("tasksource/ruletaker", split="test")
ctx_idx = defaultdict(list)
for r in ds:
    ctx_idx[norm(r["context"])].append((norm(r["question"]), r["label"]))
n0 = len(gold)
for eid, t in todo:
    if fam[eid] != "ruletaker" or eid in gold:
        continue
    lines = [l for l in t.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        continue
    q = norm(lines[-1]); ctx = norm(" ".join(lines[:-1]))
    cands = ctx_idx.get(ctx)
    if cands is None:
        # context may be single line: try full-text minus question
        cands = ctx_idx.get(norm(t[: t.rfind(lines[-1])]))
    if cands:
        for qq, lab in cands:
            if qq == q:
                gold[eid] = {"family": "ruletaker", "answer": str(lab), "source": "ruletaker"}
                break
print(f"[gold2] ruletaker +{len(gold)-n0}", flush=True)

# ---- TruthfulQA binary: match question; decide letter by which option is in mc2 'correct' set (fuzzy >= .9)
ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
q2 = {norm(r["question"]): r for r in ds}
n0 = len(gold)
def best_ratio(s, pool):
    return max((SequenceMatcher(None, s, p).ratio() for p in pool), default=0.0)
for eid, t in todo:
    if fam[eid] != "truthfulqa" or eid in gold:
        continue
    m = re.search(r"Question:\s*(.+?)\nA\.\s*(.+?)\nB\.\s*(.+?)\s*$", t, re.S)
    if not m:
        continue
    r = q2.get(norm(m.group(1)))
    if r is None:
        continue
    corr = [norm(c) for c, l in zip(r["mc2_targets"]["choices"], r["mc2_targets"]["labels"]) if l == 1]
    wrong = [norm(c) for c, l in zip(r["mc2_targets"]["choices"], r["mc2_targets"]["labels"]) if l == 0]
    a, b = norm(m.group(2)), norm(m.group(3))
    sa, sb = best_ratio(a, corr) - best_ratio(a, wrong), best_ratio(b, corr) - best_ratio(b, wrong)
    if sa > 0.15 and sb < -0.15:
        gold[eid] = {"family": "truthfulqa", "answer": "A", "source": "truthfulqa"}
    elif sb > 0.15 and sa < -0.15:
        gold[eid] = {"family": "truthfulqa", "answer": "B", "source": "truthfulqa"}
print(f"[gold2] truthfulqa +{len(gold)-n0}", flush=True)

# ---- HRMCR (date / zodiac): match on question text
n0 = len(gold)
for cfg in ("date", "zodiac"):
    try:
        ds = load_dataset("HAERAE-HUB/HRMCR", cfg, split="test")
    except Exception as exc:
        print(f"[gold2] hrmcr {cfg} skipped: {exc}", flush=True); continue
    cols = ds.column_names
    qcol = next((c for c in cols if c in ("question", "prompt", "input", "problem")), cols[0])
    acol = next((c for c in cols if c in ("answer", "label", "target", "output")), cols[-1])
    q2a = {norm(str(r[qcol])): str(r[acol]) for r in ds}
    for eid, t in todo:
        if fam[eid] != "hrmcr" or eid in gold:
            continue
        nt = norm(t)
        a = q2a.get(nt)
        if a is None:
            # prompt may contain instruction wrapper; try containment
            for q, ans in q2a.items():
                if len(q) > 40 and q in nt:
                    a = ans; break
        if a is not None:
            gold[eid] = {"family": "hrmcr", "answer": a, "source": f"hrmcr-{cfg}"}
print(f"[gold2] hrmcr +{len(gold)-n0}", flush=True)

# ---- BABILong (longdoc): configs 4k..16k, tasks qa1..; match by question line + target
n0 = len(gold)
try:
    for length in ("4k", "8k", "16k"):
        for task in ("qa1", "qa2", "qa3", "qa4", "qa5"):
            try:
                ds = load_dataset("RMT-team/babilong", length, split=task)
            except Exception:
                continue
            for r in ds:
                key = norm(r["input"][-400:])  # tail of the long input is distinctive
                for eid, t in todo:
                    if fam[eid] != "longdoc" or eid in gold:
                        continue
                    if key and key in norm(t[-600:]):
                        gold[eid] = {"family": "longdoc", "answer": str(r["target"]), "source": f"babilong-{length}-{task}"}
except Exception as exc:
    print(f"[gold2] babilong error: {exc}", flush=True)
print(f"[gold2] longdoc +{len(gold)-n0}", flush=True)

OUT.write_text(json.dumps(gold, ensure_ascii=False, indent=0), encoding="utf-8")
cov = Counter(v["family"] for v in gold.values()); tot = Counter(fam.values())
print("[gold2] coverage by family:")
for f in sorted(tot):
    print(f"  {f:16s} {cov.get(f,0):4d} / {tot[f]:4d}")
print(f"[gold2] total {len(gold)} / {len(eps)}")
