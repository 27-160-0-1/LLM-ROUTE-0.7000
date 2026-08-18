# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0
"""RuleTaker gold matching: prompt = context + '\\nQuestion: <stmt>'; label entailment/not_entailment -> True/False."""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router.heuristic import episode_text
from ossp_router.protocol import load_input
from datasets import load_dataset
OUT = ROOT / "data/gold/gold-answers.v1.json"
gold = json.loads(OUT.read_text(encoding="utf-8"))
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
eps = []
for split in ("train", "dev"):
    for e in load_input(ROOT / f"data/materialized/{split}/inputs.json").episodes:
        eps.append((e.episode_id, episode_text(e)))
ds = load_dataset("tasksource/ruletaker", split="test")
idx = defaultdict(dict)
for r in ds:
    idx[norm(r["context"])][norm(r["question"])] = r["label"]
n0 = len(gold); tried = 0
for eid, t in eps:
    if eid in gold: continue
    m = re.match(r"^(.*)\nQuestion:\s*(.+?)\s*$", t, re.S)
    if not m: continue
    tried += 1
    ctx, q = norm(m.group(1)), norm(m.group(2))
    lab = idx.get(ctx, {}).get(q)
    if lab is None: continue
    ans = "True" if str(lab).lower().startswith("entail") or str(lab) in ("1", "True", "true") else "False"
    gold[eid] = {"family": "ruletaker", "answer": ans, "source": "ruletaker"}
OUT.write_text(json.dumps(gold, ensure_ascii=False, indent=0), encoding="utf-8")
print(f"[ruletaker] tried {tried}, matched +{len(gold)-n0}, total gold {len(gold)}")
print(dict(Counter(v["family"] for v in gold.values())))
