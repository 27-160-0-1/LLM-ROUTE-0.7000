# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Answer extraction + correctness judging shared by the local pilot and the Colab labeler.

The organizer scored raw model output (no answer-format instruction was in the prompt), so
extraction has to be lenient: \\boxed{}, 'Answer:'/'정답:', markdown-bold finals, then a
family-specific fallback (last number / last A-D letter / True-False / literal containment).
"""

from __future__ import annotations

import ast
import re

_BOXED = re.compile(r"\\boxed\s*\{")
_ANS = re.compile(r"(?:final answer|answer|정답|답)\s*(?:is|:|：|=)?\s*[:：]?\s*(.+)", re.I)


def _last_boxed(text: str):
    """Return the content of the last \\boxed{...} with balanced braces, or None."""
    starts = [m.end() for m in _BOXED.finditer(text)]
    for st in reversed(starts):
        depth, i = 1, st
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[st:i - 1].strip()
    return None


def _clean(s: str) -> str:
    s = s.strip().strip("*`'\" ").rstrip(".。").strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    return s.strip()


def final_answer(text: str) -> str:
    """Best-effort final-answer span (family-agnostic)."""
    b = _last_boxed(text)
    if b is not None:
        return _clean(b)
    lines = [l for l in text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        m = _ANS.search(line)
        if m:
            return _clean(m.group(1))
    return _clean(lines[-1]) if lines else ""


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?")


def norm_num(s: str):
    s = s.replace("$", "").replace("\\%", "").replace("%", "").strip()
    m = _NUM.findall(s)
    if not m:
        return None
    v = m[-1].replace(",", "").replace(" ", "")
    try:
        if "/" in v:
            a, b = v.split("/")
            return float(a) / float(b)
        return float(v)
    except Exception:
        return None


def _all_nums(text: str):
    return [norm_num(x) for x in _NUM.findall(text.replace(",", ""))]


def _lit_eq(a: str, b: str) -> bool:
    try:
        return ast.literal_eval(a) == ast.literal_eval(b)
    except Exception:
        return re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b)


_CHECK_SNIPPET = chr(10).join([
    "import sys, json, ast",
    "d = json.load(sys.stdin)",
    "env = {}",
    "exec('import string, math, re, itertools, collections' + chr(10) + d['code'], env)",
    "def _args(inp):",
    "    try:",
    "        return ast.literal_eval('(' + inp + ',)')",
    "    except Exception:",
    "        return (ast.literal_eval(inp),)",
    "try:",
    "    ok = env['f'](*_args(d['inp'])) == ast.literal_eval(d['out'])",
    "except Exception:",
    "    ok = False",
    "print('1' if ok else '0')",
])


def _run_check(code: str, inp: str, out: str, timeout: float = 3.0) -> bool:
    """Execute the CRUXEval function on a candidate input in a throw-away subprocess with a hard
    timeout (a wrong candidate can send the function into an infinite loop) and compare the result
    with the expected output literal."""
    import json as _json
    import subprocess
    import sys
    try:
        r = subprocess.run([sys.executable, "-c", _CHECK_SNIPPET],
                           input=_json.dumps({"code": code, "inp": inp, "out": out}),
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() == "1"
    except Exception:  # timeout (child is killed) or spawn failure
        return False


def _as_args(inp: str):
    try:
        v = ast.literal_eval("(" + inp + ",)")
        return v
    except Exception:
        return (ast.literal_eval(inp),)


def judge(family: str, text: str, gold: dict, code: str = None) -> bool:
    """Judge a raw completion against gold {'answer':..., optional 'check':...}. `code` (the CRUXEval
    function source) enables execution-based checking of input predictions."""
    ans = str(gold["answer"]).strip()
    # [ANSWER] ... [/ANSWER] tags (CRUXEval-style instruction) take precedence
    mt = re.findall(r"\[ANSWER\](.*?)(?:\[/ANSWER\]|$)", text, re.S)
    if mt:
        text = mt[-1].strip() or text
    p = final_answer(text)
    tail = text[-400:]
    if family in ("belebele", "truthfulqa"):
        letters = "ABCD" if family == "belebele" else "AB"
        m = re.search(rf"\b([{letters}])\b", p.upper())
        if m:
            return m.group(1) == ans.upper()
        # fallback: last standalone letter mention like "(B)" / "B." in the tail
        ms = re.findall(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", tail)
        return bool(ms) and ms[-1] == ans.upper()
    if family == "ruletaker":
        low = p.lower()
        if ("true" in low) != ("false" in low):
            return ("true" in low) == (ans.lower() == "true")
        ms = re.findall(r"\b(true|false)\b", tail.lower())
        return bool(ms) and ms[-1] == ans.lower()
    if family == "hrmcr":
        a = ans.strip()
        if a.startswith("["):
            try:
                items = [str(x) for x in ast.literal_eval(a)]
            except Exception:
                items = [a]
            span = p if all(it in p for it in items) else tail[-200:]
            return all(it in span for it in items)
        nums = re.findall(r"\d+", a)
        if len(nums) >= 2:  # date like 1993.8.20 -> tolerate 1993년 8월 20일 / 1993-08-20 / 1993. 8. 20
            pat = r"\D{0,6}".join(rf"0?{n}" for n in nums)
            return re.search(pat, p) is not None or re.search(pat, tail[-200:]) is not None
        # fall through to numeric / string handling
    if family in ("gsm8k_or_other", "aime", "hrmcr", "dmmath"):
        b = norm_num(ans)
        a = norm_num(p)
        if b is not None:
            if a is not None and abs(a - b) < 1e-6:
                return True
            # dmmath answers may be symbolic; try exact normalized string
            if re.sub(r"\s+", "", p.lower()) == re.sub(r"\s+", "", ans.lower()):
                return True
            # last number in the whole text as a last resort
            nums = [x for x in _all_nums(tail) if x is not None]
            return bool(nums) and abs(nums[-1] - b) < 1e-6
        # symbolic gold (dmmath): normalized string equality
        return re.sub(r"\s+", "", p.lower()) == re.sub(r"\s+", "", ans.lower())
    if family == "code":
        want_input = gold.get("source", "").endswith("input")
        cands = []
        b = _last_boxed(text)
        if b is not None:
            cands.append(b)
        # inline code spans in the tail (last first) — quotes preserved
        cands += list(reversed(re.findall(r"`([^`\n]+)`", tail)))
        # echoed assertion
        for m in re.finditer(r"assert\s+f\((.*)\)\s*==\s*(.*)", tail):
            cands.append(m.group(1).strip() if want_input else m.group(2).strip())
        # 'answer is X' phrasing without stripping quotes
        for line in reversed(tail.split("\n")):
            m = _ANS.search(line)
            if m:
                cands.append(m.group(1).strip().strip("*").rstrip(".").strip())
                break
        cands.append(p)
        # last literal-looking span in the tail (list/tuple/dict/str/number/bool)
        lits = re.findall(r"(\[[^\[\]\n]*\]|\([^()\n]*\)|\{[^{}\n]*\}|'[^'\n]*'|\"[^\"\n]*\"|-?\d+(?:\.\d+)?|True|False|None)", tail)
        cands += list(reversed(lits[-3:]))
        chk = gold.get("check") or ""
        mchk = re.match(r"f\((.*)\) == (.*)$", chk, re.S)
        for cand in cands:
            cand = re.sub(r"^```\w*|```$", "", cand).strip()
            m = re.search(r"^f\((.*)\)$", cand)
            if m and want_input:
                cand = m.group(1)
            if want_input and code and mchk and cand and _run_check(code, cand, mchk.group(2).strip()):
                return True
            if cand and _lit_eq(cand, ans):
                return True
        return False
    if family == "longdoc":
        return ans.lower() in p.lower() or ans.lower() in tail.lower()
    return re.sub(r"\s+", "", p.lower()) == re.sub(r"\s+", "", ans.lower())
