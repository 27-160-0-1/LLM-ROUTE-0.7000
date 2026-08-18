"""E42 (build-time): side-information lookup tables from the pinned public sources.

Writes experiments/e42_lookup.json:
  gsm8k:     sha16(norm(question)) -> [n_steps, n_ops, log10(|answer|+1)]         (train+test, 8,792)
  ruletaker: sha16(norm(context))  -> depth code (0..5, 6=3ext, 7=NatLang)         (test split, by context)
  truthfulqa: sha16(norm(question)) -> category id (needs the 'generation' config; skipped if offline)
Keys are content hashes of a prompt sub-part, so a private prompt drawn from the same public source
hits the table; anything else falls back to 'missing'.
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if "--offline" in sys.argv:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset  # noqa: E402


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def sha16(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


out = {"gsm8k": {}, "ruletaker": {}, "truthfulqa": {}, "truthfulqa_categories": []}
import math
for split in ("train", "test"):
    for r in load_dataset("openai/gsm8k", "main", split=split):
        sol, _, ans = r["answer"].partition("####")
        steps = len([l for l in sol.strip().split("\n") if l.strip()])
        ops = len(re.findall(r"<<", sol))
        try:
            a = abs(float(ans.strip().replace(",", "")))
        except Exception:
            a = 0.0
        out["gsm8k"][sha16(norm(r["question"]))] = [steps, ops, round(math.log10(a + 1), 3)]
print("gsm8k", len(out["gsm8k"]))
DEPTH = {"depth-0": 0, "depth-1": 1, "depth-2": 2, "depth-3": 3, "depth-5": 5, "depth-3ext": 6, "depth-3ext-NatLang": 7, "NatLang": 8}
ds = load_dataset("tasksource/ruletaker", split="test")
for r in ds:
    out["ruletaker"].setdefault(sha16(norm(r["context"])), DEPTH.get(r["config"], 9))
print("ruletaker contexts", len(out["ruletaker"]), Counter(out["ruletaker"].values()))
try:
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    cats = sorted(set(ds["category"]))
    out["truthfulqa_categories"] = cats
    for r in ds:
        out["truthfulqa"][sha16(norm(r["question"]))] = cats.index(r["category"])
    print("truthfulqa", len(out["truthfulqa"]), len(cats))
except Exception as exc:
    print("truthfulqa generation config unavailable:", str(exc)[:120])
(HERE / "e42_lookup.json").write_text(json.dumps(out), encoding="utf-8")
print("wrote", HERE / "e42_lookup.json", (HERE / "e42_lookup.json").stat().st_size / 1e6, "MB")
