# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Local pilot: reproduce organizer labels for ax31-light with A.X-3.1-Light (GGUF, llama-server).

Steps:
  1. family-balanced sample of public prompts (AIME excluded) that HAVE gold answers
  2. generate n=2 completions per prompt (temperature grid) via llama-server /v1/chat/completions
  3. compare (a) output length vs organizer output_tokens/num_generations, (b) our correctness
     (gold answer match) vs organizer score -> agreement rate, correlation
Prints a per-family table; writes reports/pilot_light.json.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ossp_router import similarity  # noqa: E402
from ossp_router.heuristic import episode_text  # noqa: E402
from ossp_router.protocol import load_input, load_outcomes  # noqa: E402

sys.path.insert(0, str(ROOT / "colab-label"))
from judge import final_answer, judge  # noqa: E402  (shared with the Colab labeler)

INSTR = {
    "belebele": "다음 지문을 읽고 질문에 답하시오. 먼저 근거를 간단히 설명한 뒤, 마지막 줄에 '정답: X' (X는 A, B, C, D 중 하나) 형식으로 답하시오.\n\n",
    "truthfulqa": "Answer the following question. Briefly explain your reasoning, then end with 'Answer: X' where X is A or B.\n\n",
    "ruletaker": "Given the facts and rules, decide whether the final statement is True or False. Reason step by step, then end with 'Answer: True' or 'Answer: False'.\n\n",
    "gsm8k_or_other": "Solve the following problem step by step. Put the final answer on the last line as 'Answer: <number>'.\n\n",
    "dmmath": "Solve the following problem step by step. Put the final answer on the last line as 'Answer: ...'.\n\n",
    "aime": "Solve the following problem step by step. Put the final answer on the last line as 'Answer: <number>'.\n\n",
    "hrmcr": "다음 문제를 단계별로 풀고, 마지막 줄에 '정답: ...' 형식으로 최종 답을 쓰시오.\n\n",
    "longdoc": "Read the following text carefully and answer the question at the end. Explain briefly, then give the final answer on the last line as 'Answer: ...'.\n\n",
    "code": "You are given a Python function and an assertion with a missing value (??). Reason step by step about what the function does, then give the missing value on the last line as 'Answer: <python literal>'.\n\n",
}


def chat(server: str, prompt: str, temperature: float, n: int, max_tokens: int):
    body = {"model": "local", "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
            "top_p": 0.95, "n": n, "max_tokens": max_tokens}
    req = urllib.request.Request(f"{server}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    outs = []
    for ch in data["choices"]:
        outs.append((ch["message"]["content"], None))
    usage = data.get("usage", {})
    return outs, usage


def wait_health(server: str, max_s: int) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < max_s:
        try:
            with urllib.request.urlopen(f"{server}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(15)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--per-family", type=int, default=15)
    ap.add_argument("--temps", default="0.7")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--max-prompt-chars", type=int, default=9000, help="skip prompts longer than this (server ctx 4096)")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/pilot_light.json")
    ap.add_argument("--instruct", action="store_true",
                    help="prepend an answer-format instruction (default: raw organizer-format prompt, no instruction)")
    args = ap.parse_args()

    gold = json.loads((ROOT / "data/gold/gold-answers.v1.json").read_text(encoding="utf-8"))
    labels = {}
    eps = []
    for split in ("train", "dev"):
        aime = {e["episode_id"] for e in json.loads((ROOT / f"data/{split}/aime-selection.json").read_text(encoding="utf-8"))["episodes"]}
        for o in load_outcomes(ROOT / f"data/{split}/outcomes.json").outcomes:
            labels.setdefault(o.episode_id, {})[o.model_id] = o
        for e in load_input(ROOT / f"data/materialized/{split}/inputs.json").episodes:
            if e.episode_id in aime or e.episode_id not in gold:
                continue
            eps.append((e.episode_id, episode_text(e)))
    byf = defaultdict(list)
    for eid, t in eps:
        byf[gold[eid]["family"]].append((eid, t))
    random.seed(7)
    pilot = []
    for f, lst in sorted(byf.items()):
        for eid, t in random.sample(lst, min(args.per_family, len(lst))):
            pilot.append({"id": eid, "family": f, "prompt": t})
    print(f"[pilot] {len(pilot)} items: {dict(Counter(p['family'] for p in pilot))}", flush=True)

    temps = [float(x) for x in args.temps.split(",")]
    results = []
    if args.out.exists():
        try:
            results = json.loads(args.out.read_text(encoding="utf-8"))
        except Exception:
            results = []
    done = {r["id"] for r in results}
    print(f"[pilot] resuming with {len(done)} already done", flush=True)
    t0 = time.perf_counter()
    for i, p in enumerate(pilot):
        if p["id"] in done:
            continue
        lab = labels[p["id"]]["ax31-light"]
        row = {"id": p["id"], "family": p["family"], "org_score": float(lab.score),
               "org_out_per_gen": lab.output_tokens / lab.num_generations, "org_in_per_gen": lab.input_tokens / lab.num_generations,
               "gold": gold[p["id"]]["answer"], "runs": {}}
        if len(p["prompt"]) > args.max_prompt_chars:
            print(f"[pilot] skip {p['id']} ({p['family']}): prompt {len(p['prompt'])} chars > ctx", flush=True)
            continue
        for T in temps:
            prefix = INSTR.get(p["family"], INSTR["gsm8k_or_other"]) if args.instruct else ""
            outs = None
            for attempt in range(3):
                try:
                    outs, usage = chat(args.server, prefix + p["prompt"], T, args.n, args.max_tokens)
                    break
                except urllib.error.HTTPError as exc:  # context overflow etc.: skip item
                    print(f"[pilot] error {p['id']} ({p['family']}): HTTP {exc.code}", flush=True)
                    break
                except Exception as exc:  # server down / timeout: wait for health, retry
                    print(f"[pilot] error {p['id']} ({p['family']}) attempt {attempt}: {str(exc)[:120]}", flush=True)
                    wait_health(args.server, 1800)
            if outs is None:
                break
            if not outs:
                break
            comp = usage.get("completion_tokens")
            per = comp / max(len(outs), 1) if comp else None
            preds = [final_answer(o[0]) for o in outs]
            corr = [judge(p["family"], o[0], gold[p["id"]]) for o in outs]
            row["runs"][str(T)] = {"our_out_per_gen": per, "prompt_tokens": usage.get("prompt_tokens"),
                                   "our_score": float(np.mean(corr)), "preds": preds, "correct": corr, "texts": [o[0] for o in outs]}
        if not outs or len(row["runs"]) < len(temps):
            continue
        results.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=0), encoding="utf-8")
        if (i + 1) % 10 == 0:
            print(f"[pilot] {i+1}/{len(pilot)} {time.perf_counter()-t0:.0f}s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=0), encoding="utf-8")

    for T in temps:
        k = str(T)
        org_s = np.array([r["org_score"] for r in results]); our_s = np.array([r["runs"][k]["our_score"] for r in results])
        org_o = np.array([r["org_out_per_gen"] for r in results]); our_o = np.array([r["runs"][k]["our_out_per_gen"] or 0 for r in results])
        org_i = np.array([r["org_in_per_gen"] for r in results]); our_i = np.array([r["runs"][k]["prompt_tokens"] or 0 for r in results])
        agree = np.mean((org_s >= 0.5) == (our_s >= 0.5))
        print(f"\n[pilot] === temperature {T} ===")
        print(f"[pilot] SCORE  organizer mean {org_s.mean():.3f} | ours {our_s.mean():.3f} | agree(>=.5) {agree:.3f} | corr {np.corrcoef(org_s, our_s)[0,1]:.3f}")
        print(f"[pilot] OUTLEN organizer median {np.median(org_o):.0f} | ours {np.median(our_o):.0f} | log-ratio median {np.median(np.log1p(our_o)-np.log1p(org_o)):+.2f} | corr(log) {np.corrcoef(np.log1p(org_o), np.log1p(our_o))[0,1]:.3f}")
        print(f"[pilot] INLEN  organizer median {np.median(org_i):.0f} | ours {np.median(our_i):.0f} | diff median {np.median(org_i-our_i):+.0f}")
        print(f"[pilot] {'family':16s} n  org_score our_score agree | org_out our_out ratio")
        for f in sorted(set(r["family"] for r in results)):
            sub = [r for r in results if r["family"] == f]
            os_ = np.mean([r["org_score"] for r in sub]); us = np.mean([r["runs"][k]["our_score"] for r in sub])
            ag = np.mean([(r["org_score"] >= .5) == (r["runs"][k]["our_score"] >= .5) for r in sub])
            oo = np.median([r["org_out_per_gen"] for r in sub]); uo = np.median([r["runs"][k]["our_out_per_gen"] or 0 for r in sub])
            print(f"[pilot] {f:16s} {len(sub):2d}  {os_:.2f}      {us:.2f}     {ag:.2f}  | {oo:6.0f}  {uo:6.0f}  {uo/max(oo,1):.2f}")
    print("[pilot] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
