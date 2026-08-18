# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Ingest Colab self-label outputs into data/aux/ for the E41 CV experiment.

    python colab-label/ingest_labels.py router_label_out.zip          # or a directory with labels_*.jsonl

Writes
  data/aux/light-labels.v1.json      aux episodes with ax31-light {score, num_generations, input_tokens, output_tokens}
  data/aux/light-calibration.v1.json per-family organizer-vs-local calibration measured on the pilot set
                                     (score bias, output-length log-ratio, input-token offset)
Both files are build-time only; nothing here goes into the submission image.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from judge import judge  # noqa: E402


def read_jsonl(p: Path):
    with p.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="router_label_out.zip or directory containing labels_*.jsonl")
    ap.add_argument("--bundle", type=Path, default=HERE / "bundle")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/aux")
    ap.add_argument("--rejudge", action="store_true", help="re-run judge.py on stored completions before ingesting")
    args = ap.parse_args()

    src = args.src
    tmp = None
    if src.is_file() and src.suffix == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp.name)
        src = Path(tmp.name)
    files = sorted(src.rglob("labels_*.jsonl"))
    if not files:
        print(f"[ingest] no labels_*.jsonl under {src}")
        return 1
    pilot_items = {p["id"]: p for p in read_jsonl(args.bundle / "pilot.jsonl")}
    pool_items = {p["id"]: p for p in read_jsonl(args.bundle / "pool.jsonl")}

    pilot_rows, pool_rows = {}, {}
    for f in files:
        rows = read_jsonl(f)
        if args.rejudge:
            for r in rows:
                it = pilot_items.get(r["id"]) or pool_items.get(r["id"])
                if it is None:
                    continue
                for g in r["gens"]:
                    g["correct"] = bool(judge(it["family"], g["text"], it["gold"]))
                r["score"] = sum(g["correct"] for g in r["gens"]) / max(len(r["gens"]), 1)
        target = pilot_rows if "pilot" in f.name else pool_rows
        for r in rows:
            key = f"{r['temp']}" + (f"_{r['instruct']}" if r.get("instruct") else "")  # temp + instruction mode
            target.setdefault(key, {})[r["id"]] = r
    print(f"[ingest] pilot temps {sorted(pilot_rows)} pool temps {sorted(pool_rows)}")
    if not pool_rows:
        print("[ingest] no pool labels found; only calibration will be written")

    # ---- calibration on the pilot set, per temperature and family ----
    calib = {}
    for T, rows in pilot_rows.items():
        byf = defaultdict(list)
        for r in rows.values():
            it = pilot_items.get(r["id"])
            if it is None:
                continue
            byf[it["family"]].append((it["org"]["score"], r["score"], it["org"]["out_per_gen"], r["out_per_gen"],
                                      it["org"]["in_per_gen"], r["prompt_tokens"]))
        c = {}
        allrows = [x for v in byf.values() for x in v]
        a = np.array(allrows)
        c["_all"] = {"n": len(a), "score_bias": float(a[:, 0].mean() - a[:, 1].mean()),
                     "within_025": float(np.mean(np.abs(a[:, 0] - a[:, 1]) <= 0.25 + 1e-9)),
                     "score_corr": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
                     "outlen_logratio": float(np.median(np.log1p(a[:, 2]) - np.log1p(a[:, 3]))),
                     "inlen_offset": float(np.median(a[:, 4] - a[:, 5]))}
        for fam, v in byf.items():
            a = np.array(v)
            c[fam] = {"n": len(a), "score_bias": float(a[:, 0].mean() - a[:, 1].mean()),
                      "within_025": float(np.mean(np.abs(a[:, 0] - a[:, 1]) <= 0.25 + 1e-9)),
                      "score_corr": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if len(a) > 2 else None,
                      "outlen_logratio": float(np.median(np.log1p(a[:, 2]) - np.log1p(a[:, 3]))),
                      "inlen_offset": float(np.median(a[:, 4] - a[:, 5]))}
        calib[str(T)] = c
        print(f"[ingest] calibration T={T}: within.25 {c['_all']['within_025']:.3f} corr {c['_all']['score_corr']:.3f} "
              f"score bias {c['_all']['score_bias']:+.3f} outlen logratio {c['_all']['outlen_logratio']:+.2f} "
              f"inlen offset {c['_all']['inlen_offset']:+.0f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "light-calibration.v1.json").write_text(json.dumps(calib, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- aux episodes (pool) ----
    if pool_rows:
        T = max(pool_rows, key=lambda t: len(pool_rows[t]))
        rows = pool_rows[T]
        c = calib.get(str(T)) or (next(iter(calib.values())) if calib else {})
        instruct = next(iter(rows.values())).get("instruct", "")
        eps = []
        for r in rows.values():
            it = pool_items.get(r["id"])
            if it is None:
                continue
            n = r["n"]
            eps.append({"episode_id": r["id"], "family": it["family"], "source": it["source"], "group": it.get("group"),
                        "prompt": it["prompt"],
                        "ax31-light": {"score": r["score"], "num_generations": n,
                                       "input_tokens": int(r["prompt_tokens"]) * n,
                                       "output_tokens": int(round(r["out_per_gen"] * n))},
                        "truncated": any(g.get("finish") == "length" for g in r["gens"])})
        doc = {"schema_version": 1, "model": "skt/A.X-3.1-Light (self-label, vLLM bf16)", "temperature": T.split("_")[0],
               "instruct": instruct, "calibration_key": str(T), "n_episodes": len(eps), "episodes": eps}
        out = args.out_dir / "light-labels.v1.json"
        out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        byf = defaultdict(list)
        for e in eps:
            byf[e["family"]].append(e["ax31-light"]["score"])
        print(f"[ingest] wrote {out} with {len(eps)} aux episodes (T={T}); mean score by family: " +
              ", ".join(f"{k} {np.mean(v):.2f} (n={len(v)})" for k, v in sorted(byf.items())))
    # also keep the pilot self-labels: useful as an extra 'local light' feature/target check
    if pilot_rows:
        T = max(pilot_rows, key=lambda t: len(pilot_rows[t]))
        keep = {eid: {"score": r["score"], "out_per_gen": r["out_per_gen"], "prompt_tokens": r["prompt_tokens"]}
                for eid, r in pilot_rows[T].items()}
        (args.out_dir / "light-pilot-self.v1.json").write_text(json.dumps({"temperature": T, "items": keep}), encoding="utf-8")
        print(f"[ingest] wrote light-pilot-self.v1.json ({len(keep)} public items, T={T})")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
