# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Summarise reports/pilot_light.json (partial or complete) into the (a)(b)(c) tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, default=ROOT / "reports/pilot_light.json")
    ap.add_argument("--md", type=Path, default=None, help="optional markdown output")
    args = ap.parse_args()
    res = json.loads(args.inp.read_text(encoding="utf-8"))
    temps = sorted({k for r in res for k in r["runs"]})
    lines = []
    P = lines.append
    P(f"items: {len(res)}")
    for k in temps:
        rows = [r for r in res if k in r["runs"] and r["runs"][k]["our_out_per_gen"] is not None]
        org_s = np.array([r["org_score"] for r in rows]); our_s = np.array([r["runs"][k]["our_score"] for r in rows])
        org_o = np.array([r["org_out_per_gen"] for r in rows]); our_o = np.array([r["runs"][k]["our_out_per_gen"] for r in rows])
        org_i = np.array([r["org_in_per_gen"] for r in rows]); our_i = np.array([r["runs"][k]["prompt_tokens"] for r in rows])
        agree = np.mean((org_s >= .5) == (our_s >= .5))
        P(f"\n## temperature {k}  (n={len(rows)})")
        P(f"overall: org_score {org_s.mean():.3f} | our_score {our_s.mean():.3f} | agree(>=.5) {agree:.3f} | corr {np.corrcoef(org_s, our_s)[0,1]:.3f}")
        P(f"outlen : org median {np.median(org_o):.0f} | ours {np.median(our_o):.0f} | median ratio {np.median(our_o/np.maximum(org_o,1)):.2f} | corr(log) {np.corrcoef(np.log1p(org_o), np.log1p(our_o))[0,1]:.3f}")
        P(f"inlen  : org median {np.median(org_i):.0f} | ours {np.median(our_i):.0f} | median diff(org-ours) {np.median(org_i-our_i):+.0f}")
        P("")
        P("| family | n | org_score | our_score | agree | corr | org_out(med) | our_out(med) | ratio(med) | ratio IQR | in diff(med) |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        for f in sorted(set(r["family"] for r in rows)):
            sub = [r for r in rows if r["family"] == f]
            os_ = np.array([r["org_score"] for r in sub]); us = np.array([r["runs"][k]["our_score"] for r in sub])
            ag = np.mean((os_ >= .5) == (us >= .5))
            c = np.corrcoef(os_, us)[0, 1] if os_.std() > 0 and us.std() > 0 else float("nan")
            oo = np.array([r["org_out_per_gen"] for r in sub]); uo = np.array([r["runs"][k]["our_out_per_gen"] for r in sub])
            rat = uo / np.maximum(oo, 1)
            oi = np.array([r["org_in_per_gen"] for r in sub]); ui = np.array([r["runs"][k]["prompt_tokens"] for r in sub])
            P(f"| {f} | {len(sub)} | {os_.mean():.2f} | {us.mean():.2f} | {ag:.2f} | {c:.2f} | {np.median(oo):.0f} | {np.median(uo):.0f} | {np.median(rat):.2f} | {np.percentile(rat,25):.2f}-{np.percentile(rat,75):.2f} | {np.median(oi-ui):+.0f} |")
        # disagreement listing
        P("\ndisagreements (org>=.5 vs ours):")
        for r in rows:
            o, u = r["org_score"], r["runs"][k]["our_score"]
            if (o >= .5) != (u >= .5):
                P(f"  {r['id']:12s} {r['family']:14s} org={o:.2f} ours={u:.2f} gold={str(r['gold'])[:20]!r} preds={[str(p)[:20] for p in r['runs'][k]['preds']]}")
    out = "\n".join(lines)
    print(out)
    if args.md:
        args.md.write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
