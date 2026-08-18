# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Attach gold answers to public Train/Dev prompts by matching against the
pinned upstream benchmarks (data/sources/source-pins.v1.json).

Sources handled here (redistributable ones only; AIME excluded):
  gsm8k (main/test)         -> numeric answer after '####'
  truthfulqa binary          -> derived from TruthfulQA mc; we match question text
  belebele kor_Hang          -> correct_answer_num (1-4) -> letter
  cruxeval                   -> input/output literal
  ruletaker                  -> label True/False
  babilong (longdoc)         -> target answer
  deepmind mathematics       -> answer field (from pinned selection file if present)

Writes data/gold/gold-answers.v1.json : {episode_id: {"family":..., "answer":..., "source":...}}
Matching is exact on normalized prompt text where possible, else on a
distinctive substring (question line).  Reports coverage per family.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import similarity  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import load_input  # noqa: E402


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_episodes():
    eps = []
    for split in ("train", "dev"):
        aime = {e["episode_id"] for e in json.loads((ROOT / f"data/{split}/aime-selection.json").read_text(encoding="utf-8"))["episodes"]}
        inp = load_input(ROOT / f"data/materialized/{split}/inputs.json")
        for e in inp.episodes:
            if e.episode_id in aime:
                continue
            eps.append((e.episode_id, episode_text(e)))
    return eps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/gold/gold-answers.v1.json")
    args = parser.parse_args()
    from datasets import load_dataset  # build-time only

    eps = load_episodes()
    fam = {eid: similarity.classify_family(t) for eid, t in eps}
    gold: dict[str, dict] = {}
    print(f"[gold] {len(eps)} episodes (AIME excluded); families {dict(Counter(fam.values()))}", flush=True)

    # ---- GSM8K: prompt == question
    ds = load_dataset("openai/gsm8k", "main", split="test")
    q2a = {norm(r["question"]): r["answer"].split("####")[-1].strip().replace(",", "") for r in ds}
    for eid, t in eps:
        a = q2a.get(norm(t))
        if a is not None:
            gold[eid] = {"family": fam[eid], "answer": a, "source": "gsm8k"}
    print(f"[gold] gsm8k matched {sum(1 for v in gold.values() if v['source']=='gsm8k')}", flush=True)

    # ---- Belebele kor_Hang: prompt = passage + "\n\nQuestion: ..." + options; match on question text
    ds = load_dataset("facebook/belebele", "kor_Hang", split="test")
    q2a = {}
    for r in ds:
        q2a[norm(r["question"])] = "ABCD"[int(r["correct_answer_num"]) - 1]
    for eid, t in eps:
        if eid in gold:
            continue
        m = re.search(r"Question:\s*(.+?)\n", t)
        if m and norm(m.group(1)) in q2a:
            gold[eid] = {"family": fam[eid], "answer": q2a[norm(m.group(1))], "source": "belebele"}
    print(f"[gold] belebele matched {sum(1 for v in gold.values() if v['source']=='belebele')}", flush=True)

    # ---- CruxEval: prompt = code + 'assert f(??) == <output>' (input pred) or 'assert f(<input>) == ??' (output pred)
    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    by_code = {}
    for r in ds:
        by_code[norm(r["code"])] = (r["input"], r["output"])
    for eid, t in eps:
        if eid in gold:
            continue
        m = re.search(r"^(def f\(.*?)\n\nassert f\((.*)\) == (.*)$", t, re.S)
        if not m:
            continue
        code, lhs, rhs = m.group(1), m.group(2).strip(), m.group(3).strip()
        io = by_code.get(norm(code))
        if io is None:
            continue
        inp, out = io
        if lhs == "??":
            gold[eid] = {"family": fam[eid], "answer": inp, "source": "cruxeval-input", "check": f"f({inp}) == {out}"}
        elif rhs == "??":
            gold[eid] = {"family": fam[eid], "answer": out, "source": "cruxeval-output", "check": f"f({inp}) == {out}"}
    print(f"[gold] cruxeval matched {sum(1 for v in gold.values() if v['source'].startswith('cruxeval'))}", flush=True)

    # ---- RuleTaker: prompt = context + question; label
    try:
        ds = load_dataset("tasksource/ruletaker", split="test")
        idx = defaultdict(list)
        for r in ds:
            idx[norm(r["question"])].append((norm(r["context"]), r["label"]))
        for eid, t in eps:
            if eid in gold or fam[eid] != "ruletaker":
                continue
            lines = t.strip().split("\n")
            qline = norm(lines[-1]) if lines else ""
            for ctx, lab in idx.get(qline, []):
                if ctx[:80] in norm(t):
                    gold[eid] = {"family": fam[eid], "answer": str(lab), "source": "ruletaker"}
                    break
        print(f"[gold] ruletaker matched {sum(1 for v in gold.values() if v['source']=='ruletaker')}", flush=True)
    except Exception as exc:  # dataset schema may differ
        print(f"[gold] ruletaker skipped: {exc}", flush=True)

    # ---- TruthfulQA (binary): match question, answer letter by comparing option text to best answer
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
        q2 = {norm(r["question"]): r for r in ds}
        for eid, t in eps:
            if eid in gold or fam[eid] != "truthfulqa":
                continue
            m = re.search(r"Question:\s*(.+?)\nA\.\s*(.+?)\nB\.\s*(.+?)$", t, re.S)
            if not m:
                continue
            r = q2.get(norm(m.group(1)))
            if r is None:
                continue
            choices = r["mc1_targets"]["choices"]; labels = r["mc1_targets"]["labels"]
            correct = {norm(c) for c, l in zip(choices, labels) if l == 1}
            a_txt, b_txt = norm(m.group(2)), norm(m.group(3))
            if a_txt in correct and b_txt not in correct:
                gold[eid] = {"family": fam[eid], "answer": "A", "source": "truthfulqa"}
            elif b_txt in correct and a_txt not in correct:
                gold[eid] = {"family": fam[eid], "answer": "B", "source": "truthfulqa"}
        print(f"[gold] truthfulqa matched {sum(1 for v in gold.values() if v['source']=='truthfulqa')}", flush=True)
    except Exception as exc:
        print(f"[gold] truthfulqa skipped: {exc}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gold, ensure_ascii=False, indent=0), encoding="utf-8")
    cov = Counter(v["family"] for v in gold.values()); tot = Counter(fam.values())
    print("[gold] coverage by family:")
    for f in sorted(tot):
        print(f"  {f:16s} {cov.get(f,0):4d} / {tot[f]:4d}")
    print(f"[gold] wrote {len(gold)} gold answers -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
