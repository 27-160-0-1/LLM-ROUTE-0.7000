# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Generate ax31-light self-labels on a GPU box (Colab L4/A100) with vLLM.

    python run_labels.py --stage pilot --temps 0.7,1.0          # agreement vs organizer labels
    python run_labels.py --stage pool  --temp 0.7 --n 4         # label the new-prompt pool
    python run_labels.py --stage report                         # re-judge + summarize existing outputs

Inputs : bundle/pilot.jsonl, bundle/pool.jsonl (from build_pool.py), judge.py (same dir)
Outputs: out/labels_pilot_T{temp}.jsonl, out/labels_pool_T{temp}.jsonl, out/pilot_report.json
Each output row: {id, family, source, group, temp, prompt_tokens, gens:[{text, tokens, correct, finish}],
                  score, out_per_gen, n}
Prompts are sent RAW through the model's chat template (single user turn, no system prompt, no
answer-format instruction) — that is how the organizer's public token counts line up.
Resumable: rows already present in the output file are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge import judge  # noqa: E402

DEFAULT_MODEL = "skt/A.X-3.1-Light"

# Per-family instructions (v1). The organizer's public input_tokens exceed the raw prompt by a
# family-constant amount (math 24, truthfulqa 29, hrmcr 33, belebele 42, longdoc ~46, ruletaker 66,
# code ~221/259 tokens after the 6-token chat template) and their light outputs are very short for
# belebele/ruletaker/code/longdoc -> a format-constraining instruction was used. These are our best
# reconstructions; --instruct v1 turns them on. Prefix unless noted.
_MATH = "Solve the following math problem step by step. Put your final answer within \\boxed{}."
INSTR_V1 = {
    "gsm8k_or_other": _MATH,
    "aime": _MATH,
    "dmmath": _MATH,
    "hrmcr": "다음 문제를 단계별로 풀고, 마지막 줄에 '정답: ' 뒤에 최종 답만 쓰세요.",
    "truthfulqa": "Choose the correct answer to the following question. Explain briefly, then end with 'Answer: A' or 'Answer: B'.",
    "belebele": "다음 지문을 읽고 질문에 대한 정답을 A, B, C, D 중에서 하나만 고르세요. 설명 없이 '정답: X' 형식으로만 답하세요.",
    "ruletaker": ("You are given a set of facts and rules followed by a statement after 'Question:'. Using only the given "
                  "facts and rules (assume anything not stated or derivable is false), decide whether the statement is "
                  "True or False. Answer with only one word: True or False."),
    "longdoc": ("I will give you a long text with some facts about people and objects hidden inside, followed by a "
                "question. Answer the question based only on those facts, using the most recent information. "
                "Reply with a short answer only."),
}
CRUX_OUT = """You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{prompt}
[/PYTHON]
[ANSWER]
"""
CRUX_IN = """You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function. Do NOT output any extra information.

[PYTHON]
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
assert f(??) == 3
[/PYTHON]
[ANSWER]
assert f(["mq", "px", "zy"]) == 3
[/ANSWER]

[PYTHON]
def f(s1, s2):
    return s1 + s2
assert f(??) == "banana"
[/PYTHON]
[ANSWER]
assert f("ba", "nana") == "banana"
[/ANSWER]

[PYTHON]
{prompt}
[/PYTHON]
[ANSWER]
"""


# Variants explored on the pilot subset (ruletaker was the only family that did not agree under v1)
INSTR_VARIANTS = {
    "v1": {},
    "v2": {"ruletaker": ("You are given a set of facts and rules, followed by a statement after 'Question:'. Based only on "
                         "the given facts and rules, is the statement True or False? Respond in the format "
                         "'Answer: True' or 'Answer: False'.")},
    "v3": {"ruletaker": ("Given the facts and rules below, decide whether the final statement (after 'Question:') is "
                         "true or false. Assume the closed-world assumption: anything that cannot be derived from the "
                         "facts and rules is false. Think briefly, then answer with 'True' or 'False'.")},
    "v4": {"ruletaker": ("Determine whether the statement after 'Question:' is True or False given the facts and rules. "
                         "Answer: True or False.")},
}


def apply_instruction(item, mode):
    """Return the user message for an item under the given instruction mode ('' or 'v1'/'v2'/...)."""
    p = item["prompt"]
    if not mode:
        return p
    fam = item["family"]
    if fam == "code":
        return (CRUX_IN if "assert f(??)" in p else CRUX_OUT).format(prompt=p)
    ins = INSTR_VARIANTS.get(mode, {}).get(fam, INSTR_V1.get(fam, ""))
    return (ins + chr(10) + chr(10) + p) if ins else p


def read_jsonl(p: Path):
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def append_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


class VllmEngine:
    def __init__(self, model, max_model_len, gpu_util, dtype, quant, kv_dtype="auto"):
        from vllm import LLM
        kw = dict(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_util,
                  dtype=dtype, trust_remote_code=True, enable_prefix_caching=True)
        if quant:
            kw["quantization"] = quant
        if kv_dtype and kv_dtype != "auto":
            kw["kv_cache_dtype"] = kv_dtype
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.max_model_len = max_model_len

    def render(self, prompt):
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True)

    def count(self, text):
        return len(self.tok.encode(text, add_special_tokens=False))

    def generate(self, prompts, n, temperature, top_p, max_tokens):
        from vllm import SamplingParams
        rendered = [self.render(p) for p in prompts]
        # per-prompt max_tokens so prompt + generation never exceeds max_model_len; over-long prompts -> skipped
        sps, keep = [], []
        for i, r in enumerate(rendered):
            room = self.max_model_len - self.count(r) - 8
            if room < 64:
                continue
            keep.append(i)
            sps.append(SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=min(max_tokens, room), seed=None))
        outs = self.llm.generate([rendered[i] for i in keep], sps, use_tqdm=True) if keep else []
        res = [None] * len(prompts)
        for i, o in zip(keep, outs):
            gens = [{"text": c.text, "tokens": len(c.token_ids), "finish": c.finish_reason} for c in o.outputs]
            res[i] = (len(o.prompt_token_ids), gens)
        return res


class HfEngine:
    """Fallback (slow): transformers generate with batching. Use only when vLLM is unavailable."""

    def __init__(self, model, load_8bit):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        kw = dict(device_map="auto", torch_dtype=torch.bfloat16)
        if load_8bit:
            kw["load_in_8bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)

    def render(self, prompt):
        return self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True)

    def generate(self, prompts, n, temperature, top_p, max_tokens):
        import torch
        res = []
        B = 4
        for i in range(0, len(prompts), B):
            chunk = [self.render(p) for p in prompts[i:i + B]]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**enc, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=top_p,
                                          max_new_tokens=max_tokens, num_return_sequences=n,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            plen = enc["input_ids"].shape[1]
            for j in range(len(chunk)):
                gens = []
                for k in range(n):
                    ids = out[j * n + k, plen:]
                    ids = ids[ids != (self.tok.pad_token_id if self.tok.pad_token_id is not None else -1)]
                    text = self.tok.decode(ids, skip_special_tokens=True)
                    fin = "stop" if (self.tok.eos_token_id in ids.tolist()) else "length"
                    gens.append({"text": text, "tokens": int((ids != self.tok.eos_token_id).sum()), "finish": fin})
                ptoks = int(enc["attention_mask"][j].sum())
                res.append((ptoks, gens))
        return res


class LlamaCppEngine:
    """llama-server (OpenAI-compatible HTTP) engine for CPU/partial-GPU boxes. Concurrency = --workers."""

    def __init__(self, server="http://127.0.0.1:8080", workers=2, ctx_per_slot=2048):
        import urllib.request
        self.server = server.rstrip("/")
        self.workers = workers
        self.ctx = ctx_per_slot
        self._req = urllib.request

    def _one(self, prompt, n, temperature, top_p, max_tokens):
        import json as _json
        body = {"model": "local", "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
                "top_p": top_p, "n": n, "max_tokens": max_tokens}
        req = self._req.Request(f"{self.server}/v1/chat/completions", data=_json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
        try:
            with self._req.urlopen(req, timeout=900) as r:
                data = _json.load(r)
        except Exception as exc:  # over-long prompt for the slot context, server error, ...
            return None
        gens = [{"text": c["message"]["content"], "tokens": None, "finish": c.get("finish_reason")} for c in data["choices"]]
        usage = data.get("usage", {})
        comp = usage.get("completion_tokens")
        per = (comp / max(len(gens), 1)) if comp is not None else 0
        for g in gens:
            g["tokens"] = per
        return (usage.get("prompt_tokens", 0), gens)

    def _many(self, prompt, n, temperature, top_p, max_tokens):
        """n independent single-sample requests (keeps one slot per request; needed on tiny GPUs)."""
        outs = [self._one(prompt, 1, temperature, top_p, max_tokens) for _ in range(n)]
        outs = [o for o in outs if o is not None]
        if not outs:
            return None
        return (outs[0][0], [g for o in outs for g in o[1]])

    def generate(self, prompts, n, temperature, top_p, max_tokens):
        from concurrent.futures import ThreadPoolExecutor
        # crude length guard: ~4 chars/token; leave room for generation
        budget_chars = (self.ctx - min(max_tokens, 1024)) * 3
        with ThreadPoolExecutor(self.workers) as ex:
            futs = [ex.submit(self._many, p, n, temperature, top_p, max_tokens) if len(p) < budget_chars else None
                    for p in prompts]
            return [f.result() if f is not None else None for f in futs]


class MockEngine:
    """Smoke-test engine: echoes the gold answer half the time. No GPU needed."""

    def __init__(self, items_by_id):
        self.items = items_by_id
        self.k = 0

    def generate(self, prompts, n, temperature, top_p, max_tokens):
        res = []
        for p in prompts:
            gens = []
            for j in range(n):
                self.k += 1
                gens.append({"text": f"Reasoning... Answer: {'42' if self.k % 2 else 'X'}", "tokens": 5 + self.k % 7, "finish": "stop"})
            res.append((len(p) // 4, gens))
        return res


def run_stage(engine, items, out_path, temp, n, top_p, max_tokens, batch, instruct=""):
    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [it for it in items if it["id"] not in done]
    print(f"[labels] {out_path.name}: {len(done)} done, {len(todo)} to go", flush=True)
    # long prompts first so the scheduler packs them with short ones (vLLM), and so OOMs show early
    todo.sort(key=lambda it: -len(it["prompt"]))
    t0 = time.perf_counter()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        res = engine.generate([apply_instruction(it, instruct) for it in chunk], n, temp, top_p, max_tokens)
        rows = []
        skipped = 0
        for it, r in zip(chunk, res):
            if r is None:
                skipped += 1
                continue
            ptoks, gens = r
            code = it["prompt"].split(chr(10) * 2 + "assert")[0] if it["family"] == "code" else None
            for g in gens:
                g["correct"] = bool(judge(it["family"], g["text"], it["gold"], code))
            score = sum(g["correct"] for g in gens) / max(len(gens), 1)
            rows.append({"id": it["id"], "family": it["family"], "source": it["source"], "group": it.get("group"),
                         "temp": temp, "instruct": instruct, "prompt_tokens": ptoks, "n": len(gens), "score": score,
                         "out_per_gen": sum(g["tokens"] for g in gens) / max(len(gens), 1), "gens": gens})
        append_jsonl(out_path, rows)
        el = time.perf_counter() - t0
        print(f"[labels] {min(i + batch, len(todo))}/{len(todo)} {el:.0f}s" + (f" (skipped {skipped} over-long)" if skipped else ""), flush=True)


def report(bundle, out_dir):
    import numpy as np
    pilot = {p["id"]: p for p in read_jsonl(bundle / "pilot.jsonl")}
    rep = {}
    for f in sorted(out_dir.glob("labels_pilot_T*.jsonl")):
        rows = read_jsonl(f)
        # re-judge with the current judge.py (lets you fix extraction without regenerating)
        for r in rows:
            it = pilot.get(r["id"])
            if it is None:
                continue
            code = it["prompt"].split(chr(10) * 2 + "assert")[0] if it["family"] == "code" else None
            for g in r["gens"]:
                g["correct"] = bool(judge(it["family"], g["text"], it["gold"], code))
            r["score"] = sum(g["correct"] for g in r["gens"]) / max(len(r["gens"]), 1)
        rows = [r for r in rows if r["id"] in pilot]
        if not rows:
            continue
        org_s = np.array([pilot[r["id"]]["org"]["score"] for r in rows]); our_s = np.array([r["score"] for r in rows])
        org_o = np.array([pilot[r["id"]]["org"]["out_per_gen"] for r in rows]); our_o = np.array([r["out_per_gen"] for r in rows])
        org_i = np.array([pilot[r["id"]]["org"]["in_per_gen"] for r in rows]); our_i = np.array([r["prompt_tokens"] for r in rows])
        trunc = np.mean([any(g["finish"] == "length" for g in r["gens"]) for r in rows])
        d = {"n": len(rows), "org_score_mean": float(org_s.mean()), "our_score_mean": float(our_s.mean()),
             "agree_bin": float(np.mean((org_s >= .5) == (our_s >= .5))),
             "within_025": float(np.mean(np.abs(org_s - our_s) <= 0.25 + 1e-9)),
             "score_corr": float(np.corrcoef(org_s, our_s)[0, 1]),
             "score_mae": float(np.mean(np.abs(org_s - our_s))),
             "outlen_corr_log": float(np.corrcoef(np.log1p(org_o), np.log1p(our_o))[0, 1]),
             "outlen_logratio_median": float(np.median(np.log1p(our_o) - np.log1p(org_o))),
             "inlen_diff_median": float(np.median(org_i - our_i)),
             "truncated_frac": float(trunc), "families": {}}
        for fam in sorted({r["family"] for r in rows}):
            idx = np.array([r["family"] == fam for r in rows])
            d["families"][fam] = {"n": int(idx.sum()), "org": float(org_s[idx].mean()), "ours": float(our_s[idx].mean()),
                                  "agree_bin": float(np.mean((org_s[idx] >= .5) == (our_s[idx] >= .5))),
                                  "within_025": float(np.mean(np.abs(org_s[idx] - our_s[idx]) <= 0.25 + 1e-9)),
                                  "score_corr": float(np.corrcoef(org_s[idx], our_s[idx])[0, 1]) if idx.sum() > 2 else None,
                                  "outlen_ratio_median": float(np.median(our_o[idx]) / max(np.median(org_o[idx]), 1))}
        rep[f.name] = d
        print(f"\n[report] {f.name}: n={d['n']} agree(bin)={d['agree_bin']:.3f} within.25={d['within_025']:.3f} "
              f"corr={d['score_corr']:.3f} mae={d['score_mae']:.3f} | out-len corr {d['outlen_corr_log']:.3f} "
              f"logratio {d['outlen_logratio_median']:+.2f} | in-len diff {d['inlen_diff_median']:+.0f} | trunc {d['truncated_frac']:.3f}")
        print(f"[report] {'family':16s} n   org   ours  agree w.25  corr  outlen")
        for fam, v in d["families"].items():
            c = f"{v['score_corr']:.2f}" if v["score_corr"] is not None else "  - "
            print(f"[report] {fam:16s} {v['n']:4d} {v['org']:.2f}  {v['ours']:.2f}  {v['agree_bin']:.2f}  {v['within_025']:.2f}  {c}  {v['outlen_ratio_median']:.2f}")
    (out_dir / "pilot_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for f in sorted(out_dir.glob("labels_pool_T*.jsonl")):
        rows = read_jsonl(f)
        byf = defaultdict(list)
        for r in rows:
            byf[r["family"]].append(r["score"])
        print(f"\n[report] {f.name}: {len(rows)} rows; mean score by family: " +
              ", ".join(f"{k} {sum(v)/len(v):.2f} (n={len(v)})" for k, v in sorted(byf.items())))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["pilot", "pool", "both", "report"], default="both")
    ap.add_argument("--bundle", type=Path, default=HERE / "bundle")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--engine", choices=["vllm", "hf", "llamacpp", "mock"], default="vllm")
    ap.add_argument("--server", default="http://127.0.0.1:8080", help="llamacpp engine: llama-server URL")
    ap.add_argument("--workers", type=int, default=2, help="llamacpp engine: concurrent requests (slots / n)")
    ap.add_argument("--ctx-per-slot", type=int, default=2048, help="llamacpp engine: per-slot context for the length guard")
    ap.add_argument("--temps", default="0.7", help="pilot temperatures, comma-separated")
    ap.add_argument("--temp", type=float, default=None, help="pool temperature (default: first of --temps)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quant", default=None, help="vLLM quantization (e.g. awq, fp8) if the model repo is quantized")
    ap.add_argument("--kv-dtype", default="auto", help="vLLM kv_cache_dtype (fp8 halves KV memory; use on 24GB GPUs)")
    ap.add_argument("--load-8bit", action="store_true", help="hf engine: bitsandbytes 8-bit")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit-per-family", type=int, default=None, help="cap pool items per family (smoke tests)")
    ap.add_argument("--pilot-limit-per-family", type=int, default=None)
    ap.add_argument("--instruct", default="", help="'' = raw prompt, 'v1' = per-family instructions (see INSTR_V1)")
    ap.add_argument("--tag", default="", help="suffix for output file names (e.g. sub for subset runs)")
    ap.add_argument("--families", default="", help="comma-separated family filter (subset experiments)")
    args = ap.parse_args()

    if args.stage == "report":
        report(args.bundle, args.out)
        return 0
    if args.engine == "vllm":
        engine = VllmEngine(args.model, args.max_model_len, args.gpu_util, args.dtype, args.quant, args.kv_dtype)
    elif args.engine == "hf":
        engine = HfEngine(args.model, args.load_8bit)
    elif args.engine == "llamacpp":
        engine = LlamaCppEngine(args.server, args.workers, args.ctx_per_slot)
    else:
        engine = MockEngine({})
    temps = [float(x) for x in args.temps.split(",")]
    pool_temp = args.temp if args.temp is not None else temps[0]

    fams = {f for f in args.families.split(",") if f}
    if args.stage in ("pilot", "both"):
        items = read_jsonl(args.bundle / "pilot.jsonl")
        if fams:
            items = [it for it in items if it["family"] in fams]
        if args.pilot_limit_per_family:
            byf = defaultdict(list)
            for it in items:
                if len(byf[it["family"]]) < args.pilot_limit_per_family:
                    byf[it["family"]].append(it)
            items = [it for lst in byf.values() for it in lst]
        for T in temps:
            run_stage(engine, items, args.out / f"labels_pilot_T{T}{args.tag}.jsonl", T, args.n, args.top_p, args.max_tokens,
                      args.batch, args.instruct)
        report(args.bundle, args.out)
    if args.stage in ("pool", "both"):
        items = read_jsonl(args.bundle / "pool.jsonl")
        if fams:
            items = [it for it in items if it["family"] in fams]
        if args.limit_per_family:
            byf = defaultdict(list)
            for it in items:
                if len(byf[it["family"]]) < args.limit_per_family:
                    byf[it["family"]].append(it)
            items = [it for lst in byf.values() for it in lst]
        run_stage(engine, items, args.out / f"labels_pool_T{pool_temp}{args.tag}.jsonl", pool_temp, args.n, args.top_p,
                  args.max_tokens, args.batch, args.instruct)
        report(args.bundle, args.out)
    print("[labels] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
