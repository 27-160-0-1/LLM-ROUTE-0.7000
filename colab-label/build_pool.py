# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Build the self-labeling bundle: (a) pilot set = public Train/Dev items with gold answers and
organizer ax31-light labels, (b) pool = NEW prompts from the same pinned public sources, rendered
in the organizer's prompt template, never overlapping public Train/Dev.

Run locally (HF datasets are cached under ~/.cache/huggingface) or on Colab (online):

    python colab-label/build_pool.py --out colab-label/bundle [--verify] [--offline]

Outputs (bundle/):
    pilot.jsonl   {id, family, source, prompt, gold, org: {score, out_per_gen, in_per_gen, num_generations}}
    pool.jsonl    {id, family, source, prompt, gold}
    meta.json     per-family counts + template verification report
The prompt templates were reverse-engineered from public Train/Dev; --verify re-renders every
gold-matched public item from its source and reports exact-match rate per family (should be ~1.0).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import similarity  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import load_input, load_outcomes  # noqa: E402

CAPS = {  # pool cap per family (None = all available)
    "gsm8k_or_other": 2500, "belebele": None, "code": None, "ruletaker": 1500,
    "truthfulqa": None, "longdoc": 500, "hrmcr": None,
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ---------------- organizer templates (verified by --verify) ----------------
def t_gsm8k(r):
    return r["question"]


def t_belebele(r):
    return (f"{r['flores_passage']}\n\nQuestion: {r['question']}\nA. {r['mc_answer1']}\nB. {r['mc_answer2']}"
            f"\nC. {r['mc_answer3']}\nD. {r['mc_answer4']}")


def t_crux_in(r):
    return f"{r['code']}\n\nassert f(??) == {r['output']}"


def t_crux_out(r):
    return f"{r['code']}\n\nassert f({r['input']}) == ??"


def t_ruletaker(r):
    return f"{r['context'].strip()}\nQuestion: {r['question'].strip()}"


def _opt(s: str) -> str:
    return s.strip().rstrip(".").strip()


def t_truthfulqa(q, a, b):
    return f"Question: {q}\nA. {_opt(a)}\nB. {_opt(b)}"


def t_babilong(r):
    return f"{r['input']}\n{r['question']}"


def t_hrmcr(r):
    return r["question"]


def group_key(family: str, prompt: str) -> str:
    """Leakage group: siblings that share a passage/code/question/context get the same key so the
    CV harness can keep aux siblings of a hold-out item out of the fold-train set."""
    NL2Q = chr(10) + chr(10) + "Question:"
    NL2A = chr(10) + chr(10) + "assert"
    NLQ = chr(10) + "Question:"
    if family == "belebele" and NL2Q in prompt:
        return sha(norm(prompt.split(NL2Q)[0]))
    if family == "code" and NL2A in prompt:
        return sha(norm(prompt.split(NL2A)[0]))
    if family == "truthfulqa":
        return sha(norm(prompt.split(chr(10))[0]))
    if family == "ruletaker" and NLQ in prompt:
        return sha(norm(prompt.rsplit(NLQ, 1)[0]))
    return sha(norm(prompt))


def ruletaker_label(v) -> str:
    s = str(v).strip().lower()
    if s in ("entailment", "true", "1", "yes"):
        return "True"
    return "False"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "bundle")
    ap.add_argument("--verify", action="store_true", help="re-render public gold items and report template exact-match")
    ap.add_argument("--offline", action="store_true", help="HF_DATASETS_OFFLINE=1 (use local cache only)")
    ap.add_argument("--seed", type=int, default=20260818)
    args = ap.parse_args()
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset  # build-time only

    rng = random.Random(args.seed)
    gold = json.loads((ROOT / "data/gold/gold-answers.v1.json").read_text(encoding="utf-8"))
    public = {}  # eid -> text
    labels = {}
    for split in ("train", "dev"):
        for o in load_outcomes(ROOT / f"data/{split}/outcomes.json").outcomes:
            labels.setdefault(o.episode_id, {})[o.model_id] = o
        for e in load_input(ROOT / f"data/materialized/{split}/inputs.json").episodes:
            public[e.episode_id] = episode_text(e)
    public_norm = {norm(t) for t in public.values()}
    print(f"[pool] public items {len(public)}, gold-covered {len(gold)}", flush=True)

    # ---------- pilot set: every gold-covered public item with its organizer light label ----------
    pilot = []
    for eid, g in gold.items():
        lab = labels[eid]["ax31-light"]
        pilot.append({
            "id": eid, "family": g["family"], "source": g["source"], "prompt": public[eid],
            "group": group_key(g["family"], public[eid]),
            "gold": {k: v for k, v in g.items() if k in ("answer", "check", "source")},
            "org": {"score": float(lab.score), "num_generations": lab.num_generations,
                    "out_per_gen": lab.output_tokens / lab.num_generations,
                    "in_per_gen": lab.input_tokens / lab.num_generations},
        })
    print(f"[pool] pilot items {len(pilot)}: {dict(Counter(p['family'] for p in pilot))}", flush=True)

    # ---------- pool candidates ----------
    cands = defaultdict(list)  # family -> [(prompt, gold, source)]
    verify = defaultdict(lambda: [0, 0])  # family -> [rendered==public, gold items tried]

    def add(family, prompt, answer, source, check=None):
        if norm(prompt) in public_norm:
            return
        g = {"answer": answer, "source": source}
        if check:
            g["check"] = check
        cands[family].append((prompt, g, source))

    def chk(family, prompt):
        if args.verify:
            verify[family][1] += 1
            verify[family][0] += norm(prompt) in public_norm

    # GSM8K: organizer used the test split; pool = train split (same distribution)
    for split in ("test", "train"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for r in ds:
            p = t_gsm8k(r)
            a = r["answer"].split("####")[-1].strip().replace(",", "")
            if split == "test":
                chk("gsm8k_or_other", p)
            add("gsm8k_or_other", p, a, f"gsm8k-{split}")
    # Belebele kor_Hang
    ds = load_dataset("facebook/belebele", "kor_Hang", split="test")
    for r in ds:
        p = t_belebele(r)
        chk("belebele", p)
        add("belebele", p, "ABCD"[int(r["correct_answer_num"]) - 1], "belebele")
    # CRUXEval input / output prediction
    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    for r in ds:
        pi, po = t_crux_in(r), t_crux_out(r)
        chk("code", pi); chk("code", po)
        add("code", pi, r["input"], "cruxeval-input", check=f"f({r['input']}) == {r['output']}")
        add("code", po, r["output"], "cruxeval-output", check=f"f({r['input']}) == {r['output']}")
    # RuleTaker: organizer used test; pool = a depth-balanced sample of train
    ds = load_dataset("tasksource/ruletaker", split="test")
    for i, r in enumerate(ds):
        if i > 20000:
            break
        chk("ruletaker", t_ruletaker(r))
    ds = load_dataset("tasksource/ruletaker", split="train")
    by_cfg = defaultdict(list)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    want = CAPS["ruletaker"] or 1500
    for i in idxs[:60000]:
        r = ds[i]
        cfg = r.get("config", "")
        if len(by_cfg[cfg]) < max(50, want // 6):
            by_cfg[cfg].append(r)
    for cfg, rows in by_cfg.items():
        for r in rows:
            add("ruletaker", t_ruletaker(r), ruletaker_label(r["label"]), f"ruletaker-{cfg}")
    # TruthfulQA binary: one correct (mc1) vs one random incorrect, random A/B order
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    for r in ds:
        ch, lb = r["mc1_targets"]["choices"], r["mc1_targets"]["labels"]
        correct = [c for c, l in zip(ch, lb) if l == 1]
        wrong = [c for c, l in zip(ch, lb) if l == 0]
        if not correct or not wrong:
            continue
        c = correct[0]
        for w in wrong:  # any rendering that hits a public item counts as verified
            chk("truthfulqa", t_truthfulqa(r["question"], c, w)); chk("truthfulqa", t_truthfulqa(r["question"], w, c))
        w = rng.choice(wrong)
        if rng.random() < 0.5:
            add("truthfulqa", t_truthfulqa(r["question"], c, w), "A", "truthfulqa")
        else:
            add("truthfulqa", t_truthfulqa(r["question"], w, c), "B", "truthfulqa")
    # BABILong 4k/16k qa1..qa10 (whatever is available)
    for length in ("4k", "16k"):
        for task in [f"qa{i}" for i in range(1, 11)]:
            try:
                ds = load_dataset("RMT-team/babilong", length, split=task)
            except Exception:
                continue
            for r in ds:
                p = t_babilong(r)
                chk("longdoc", p)
                add("longdoc", p, str(r["target"]), f"babilong-{length}-{task}")
    # HRMCR date / zodiac
    for cfg in ("date", "zodiac"):
        try:
            ds = load_dataset("HAERAE-HUB/HRMCR", cfg, split="test")
        except Exception as exc:
            print(f"[pool] hrmcr {cfg} skipped: {exc}", flush=True)
            continue
        for r in ds:
            p = t_hrmcr(r)
            chk("hrmcr", p)
            add("hrmcr", p, str(r["answer"]), f"hrmcr-{cfg}")

    # ---------- cap, shuffle, write ----------
    pool = []
    for fam, lst in sorted(cands.items()):
        rng.shuffle(lst)
        cap = CAPS.get(fam)
        keep = lst if cap is None else lst[:cap]
        for prompt, g, src in keep:
            pool.append({"id": f"aux-{fam}-{sha(prompt)}", "family": fam, "source": src, "prompt": prompt,
                         "group": group_key(fam, prompt), "gold": g})
    # exact-dup guard inside the pool
    seen, dedup = set(), []
    for p in pool:
        k = norm(p["prompt"])
        if k in seen:
            continue
        seen.add(k); dedup.append(p)
    pool = dedup
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "pilot.jsonl").open("w", encoding="utf-8") as fh:
        for p in pilot:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    with (args.out / "pool.jsonl").open("w", encoding="utf-8") as fh:
        for p in pool:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {"pilot": dict(Counter(p["family"] for p in pilot)), "pool": dict(Counter(p["family"] for p in pool)),
            "pool_sources": dict(Counter(p["source"] for p in pool)),
            "verify": {f: {"match": v[0], "tried": v[1], "rate": (v[0] / v[1] if v[1] else None)} for f, v in verify.items()},
            "seed": args.seed}
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[pool] pool items {len(pool)}: {meta['pool']}", flush=True)
    if args.verify:
        print("[pool] template verification (rendered-from-source == some public prompt):")
        for f, v in sorted(verify.items()):
            print(f"  {f:16s} {v[0]:5d} / {v[1]:5d}")
    print(f"[pool] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
