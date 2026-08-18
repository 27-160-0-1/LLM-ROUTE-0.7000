"""E42 side-information features (stdlib only, so they can be ported into ossp_router if they win).

Two groups, both computed from the prompt text alone at runtime:
  PARSE  — family-specific structure parsed with regexes (always available)
  LOOKUP — content-hash lookup into tables built from the pinned public sources
           (experiments/e42_lookup.json; hits only when the prompt is drawn from those sources,
           otherwise the 'missing' indicator fires and values are 0)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
_LOOKUP = None


def lookup():
    global _LOOKUP
    if _LOOKUP is None:
        _LOOKUP = json.loads((HERE / "e42_lookup.json").read_text(encoding="utf-8"))
    return _LOOKUP


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def sha16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


PARSE_NAMES = [
    # ruletaker-like
    "rt_n_sent", "rt_n_rules", "rt_n_facts", "rt_n_neg", "rt_q_neg", "rt_n_names", "rt_n_someone", "rt_n_attr",
    # cruxeval-like
    "cx_is_code", "cx_input_mode", "cx_n_lines", "cx_n_loops", "cx_n_if", "cx_n_str_ops", "cx_n_calls", "cx_lit_len",
    "cx_recursion", "cx_n_return",
    # babilong-like
    "bl_n_babi", "bl_qa_where", "bl_qa_before", "bl_qa_dir", "bl_qa_give", "bl_qa_yesno", "bl_qa_count", "bl_qa_carry",
    "bl_qa_other", "bl_log_len",
    # belebele-like
    "bb_is_mc4", "bb_log_passage", "bb_log_q", "bb_opt_mean", "bb_opt_std",
    # truthfulqa-like
    "tq_is_bin", "tq_log_q", "tq_opt_diff",
    # math-like (gsm8k / dmmath / aime)
    "mt_n_numbers", "mt_n_ops", "mt_dm_round", "mt_dm_solve", "mt_dm_simplify", "mt_dm_calc", "mt_dm_prime",
    "mt_dm_biggest", "mt_dm_deriv", "mt_dm_prob", "mt_dm_base", "mt_dm_sort", "mt_dm_expand", "mt_dm_units",
    "mt_dm_seq", "mt_dm_remainder", "mt_dm_gcd", "mt_dm_short",
    # hrmcr-like
    "hr_zodiac", "hr_date", "hr_n_lines",
]
LOOKUP_NAMES = ["lk_gsm_hit", "lk_gsm_steps", "lk_gsm_ops", "lk_gsm_logans",
                "lk_rt_hit", "lk_rt_depth", "lk_rt_ext", "lk_rt_natlang",
                "lk_tq_hit", "lk_tq_cat"]

_BABI_NAMES = r"(?:Mary|John|Sandra|Daniel|Fred|Bill|Jeff|Julie|Jason|Antoine|Sumit|Yann|Emily|Winona|Jessica|Brian|Lily|Bernhard|Greg|Gertrude)"
_BABI_SENT = re.compile(_BABI_NAMES + r" (?:went|moved|journeyed|travelled|traveled|got|grabbed|picked up|took|dropped|discarded|put down|left|handed|gave|passed|is|went back)\b")


def parse_features(text: str) -> list[float]:
    f = {}
    t = text
    low = t.lower()
    lines = [l for l in t.split("\n") if l.strip()]
    # ---- ruletaker ----
    is_rt = bool(re.search(r"\nQuestion: .+\.$", t.strip())) and "\nA. " not in t
    sents = re.split(r"(?<=[.!?])\s+", t.split("\nQuestion:")[0]) if is_rt else []
    rules = [s for s in sents if re.match(r"\s*(If|All|Every|Some|Any|Things|People|Cold|Big|Red|Rough|Round|Kind|Nice|Smart|Young|White|Blue|Green|Furry|Quiet)\b", s) and (" then " in s or " are " in s or " is " in s and "If" in s)]
    f["rt_n_sent"] = len(sents)
    f["rt_n_rules"] = len(rules)
    f["rt_n_facts"] = max(0, len(sents) - len(rules))
    f["rt_n_neg"] = low.count(" not ") + low.count("does not") if is_rt else 0
    q = t.rsplit("\nQuestion:", 1)[-1] if is_rt else ""
    f["rt_q_neg"] = 1.0 if " not " in q or "does not" in q else 0.0
    f["rt_n_names"] = len(set(re.findall(r"\b(Anne|Bob|Charlie|Dave|Erin|Fiona|Gary|Harry)\b", t))) if is_rt else 0
    f["rt_n_someone"] = len(re.findall(r"\b(someone|something|things|people)\b", low)) if is_rt else 0
    f["rt_n_attr"] = len(set(re.findall(r"\bis (?:not )?(\w+)\.", t))) if is_rt else 0
    # ---- cruxeval ----
    is_cx = t.lstrip().startswith("def f(") or "\nassert f(" in t or t.startswith("s = ") or "def f(" in t[:400]
    f["cx_is_code"] = 1.0 if is_cx else 0.0
    f["cx_input_mode"] = 1.0 if "assert f(??)" in t else 0.0
    code = t.split("\n\nassert")[0] if is_cx else ""
    f["cx_n_lines"] = len([l for l in code.split("\n") if l.strip()])
    f["cx_n_loops"] = len(re.findall(r"\b(for|while)\b", code))
    f["cx_n_if"] = len(re.findall(r"\bif\b|\belif\b", code))
    f["cx_n_str_ops"] = len(re.findall(r"\.(?:split|join|replace|strip|upper|lower|find|index|count|format|startswith|endswith|isdigit|isalpha|isupper|islower|title|swapcase|translate|zfill|center|ljust|rjust)\(", code))
    f["cx_n_calls"] = len(re.findall(r"\w+\(", code))
    m = re.search(r"assert f\((.*)\) == (.*)$", t, re.S)
    lit = (m.group(2) if f["cx_input_mode"] else m.group(1)) if (m and is_cx) else ""
    f["cx_lit_len"] = math.log1p(len(lit))
    f["cx_recursion"] = 1.0 if re.search(r"\n\s+.*\bf\(", code) else 0.0
    f["cx_n_return"] = len(re.findall(r"\breturn\b", code))
    # ---- babilong ----
    n_babi = len(_BABI_SENT.findall(t))
    f["bl_n_babi"] = math.log1p(n_babi)
    ql = lines[-1].strip().lower() if lines else ""
    is_bl = n_babi >= 2 and ql.endswith("?")
    f["bl_qa_where"] = 1.0 if is_bl and ql.startswith("where is") and "before" not in ql else 0.0
    f["bl_qa_before"] = 1.0 if is_bl and "before" in ql else 0.0
    f["bl_qa_dir"] = 1.0 if is_bl and re.search(r"(east|west|north|south) of", ql) else 0.0
    f["bl_qa_give"] = 1.0 if is_bl and ("who did" in ql or "who gave" in ql or "who received" in ql) else 0.0
    f["bl_qa_yesno"] = 1.0 if is_bl and ql.startswith("is ") else 0.0
    f["bl_qa_count"] = 1.0 if is_bl and ql.startswith("how many") else 0.0
    f["bl_qa_carry"] = 1.0 if is_bl and "carrying" in ql and not ql.startswith("how many") else 0.0
    f["bl_qa_other"] = 1.0 if is_bl and not any(f[k] for k in ("bl_qa_where", "bl_qa_before", "bl_qa_dir", "bl_qa_give", "bl_qa_yesno", "bl_qa_count", "bl_qa_carry")) else 0.0
    f["bl_log_len"] = math.log1p(len(t)) if is_bl else 0.0
    # ---- belebele ----
    is_bb = "\n\nQuestion:" in t and "\nD. " in t
    f["bb_is_mc4"] = 1.0 if is_bb else 0.0
    if is_bb:
        passage, rest = t.split("\n\nQuestion:", 1)
        opts = re.findall(r"\n[A-D]\. (.*)", rest)
        ol = [len(o) for o in opts] or [0]
        f["bb_log_passage"] = math.log1p(len(passage))
        f["bb_log_q"] = math.log1p(len(rest.split("\n")[0]))
        f["bb_opt_mean"] = sum(ol) / len(ol)
        f["bb_opt_std"] = (sum((x - f["bb_opt_mean"]) ** 2 for x in ol) / len(ol)) ** 0.5
    else:
        f["bb_log_passage"] = f["bb_log_q"] = f["bb_opt_mean"] = f["bb_opt_std"] = 0.0
    # ---- truthfulqa ----
    is_tq = t.startswith("Question:") and "\nA. " in t and "\nB. " in t and "\nC. " not in t
    f["tq_is_bin"] = 1.0 if is_tq else 0.0
    if is_tq:
        m = re.search(r"Question:\s*(.+?)\nA\.\s*(.+?)\nB\.\s*(.+?)$", t, re.S)
        f["tq_log_q"] = math.log1p(len(m.group(1))) if m else 0.0
        f["tq_opt_diff"] = (len(m.group(2)) - len(m.group(3))) if m else 0.0
    else:
        f["tq_log_q"] = f["tq_opt_diff"] = 0.0
    # ---- math ----
    f["mt_n_numbers"] = math.log1p(len(re.findall(r"-?\d+(?:\.\d+)?", t))) if len(t) < 3000 else 0.0
    f["mt_n_ops"] = len(re.findall(r"[+\-*/^=]", t)) if len(t) < 600 else 0
    short = len(t) < 400
    f["mt_dm_round"] = 1.0 if short and low.startswith("round") else 0.0
    f["mt_dm_solve"] = 1.0 if short and low.startswith("solve") else 0.0
    f["mt_dm_simplify"] = 1.0 if short and low.startswith("simplify") else 0.0
    f["mt_dm_calc"] = 1.0 if short and (low.startswith("calculate") or low.startswith("evaluate") or low.startswith("what is") or low.startswith("work out")) else 0.0
    f["mt_dm_prime"] = 1.0 if short and ("prime" in low or "factor" in low) else 0.0
    f["mt_dm_biggest"] = 1.0 if short and ("biggest" in low or "smallest" in low or "closest" in low or "which is" in low) else 0.0
    f["mt_dm_deriv"] = 1.0 if short and ("derivative" in low or "differentiate" in low) else 0.0
    f["mt_dm_prob"] = 1.0 if short and ("prob" in low or "picked" in low or "chosen" in low or "sequence" in low and "letters" in low) else 0.0
    f["mt_dm_base"] = 1.0 if short and ("base " in low or "in base" in low) else 0.0
    f["mt_dm_sort"] = 1.0 if short and ("sort" in low or "put " in low and "order" in low) else 0.0
    f["mt_dm_expand"] = 1.0 if short and ("expand" in low or "collect" in low or "polynomial" in low) else 0.0
    f["mt_dm_units"] = 1.0 if short and re.search(r"\b(convert|how many (?:milli|centi|kilo|micro|nano)|grams|metres|meters|seconds|minutes|hours|litres|liters|tonnes)\b", low) else 0.0
    f["mt_dm_seq"] = 1.0 if short and ("next term" in low or "sequence" in low or "nth term" in low) else 0.0
    f["mt_dm_remainder"] = 1.0 if short and ("remainder" in low or "divided by" in low or "divisible" in low or "multiple of" in low or "divisor" in low) else 0.0
    f["mt_dm_gcd"] = 1.0 if short and ("common divisor" in low or "common multiple" in low or "highest common" in low or "least common" in low) else 0.0
    f["mt_dm_short"] = 1.0 if short and len(t) < 120 else 0.0
    # ---- hrmcr ----
    f["hr_zodiac"] = 1.0 if "띠" in t and ("나이" in t or "생년" in t or "선배" in t or "후배" in t) else 0.0
    f["hr_date"] = 1.0 if re.search(r"음력|양력|생일|날짜", t) and re.search(r"\d{4}년", t) and "띠" not in t else 0.0
    f["hr_n_lines"] = len(lines) if (f["hr_zodiac"] or f["hr_date"]) else 0
    return [float(f[k]) for k in PARSE_NAMES]


def lookup_features(text: str) -> list[float]:
    L = lookup()
    v = {k: 0.0 for k in LOOKUP_NAMES}
    t = text.strip()
    g = L["gsm8k"].get(sha16(norm(t)))
    if g is not None:
        v["lk_gsm_hit"] = 1.0
        v["lk_gsm_steps"], v["lk_gsm_ops"], v["lk_gsm_logans"] = float(g[0]), float(g[1]), float(g[2])
    if "\nQuestion:" in t and "\nA. " not in t:
        ctx = t.rsplit("\nQuestion:", 1)[0]
        d = L["ruletaker"].get(sha16(norm(ctx)))
        if d is not None:
            v["lk_rt_hit"] = 1.0
            v["lk_rt_depth"] = float(d if d <= 5 else 3)
            v["lk_rt_ext"] = 1.0 if d in (6, 7) else 0.0
            v["lk_rt_natlang"] = 1.0 if d in (7, 8) else 0.0
    if t.startswith("Question:") and "\nA. " in t:
        q = t.split("\n")[0][len("Question:"):].strip()
        c = L["truthfulqa"].get(sha16(norm(q)))
        if c is not None:
            v["lk_tq_hit"] = 1.0
            v["lk_tq_cat"] = float(c)
    return [v[k] for k in LOOKUP_NAMES]
