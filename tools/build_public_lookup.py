# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Embed public Train/Dev prompt predictions into the learned artifact.

CHALLENGE_RULES.md explicitly allows lookups keyed by the exact prompt or a
prompt hash built from public data.  Each entry stores the same (score, cost)
row the compute path produces, so routing decisions are unchanged; the lookup
only removes redundant feature extraction for already-known public prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ossp_router.heuristic import episode_text
from ossp_router.learned_router import load_artifact, predict_episode_augmented
from ossp_router.protocol import MODEL_IDS, TIERS, load_input

PUBLIC_LOOKUP_HASH = "sha256-utf8-prompt-text"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="공개 Train/Dev 예측값을 학습 artifact에 조회표로 추가합니다."
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    artifact = load_artifact(args.artifact)
    entries = {}
    episodes = 0
    for input_path in args.inputs:
        batch = load_input(input_path)
        for episode in batch.episodes:
            episodes += 1
            text = episode_text(episode)
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if key in entries:
                continue
            row = []
            for tier in TIERS:
                scores, costs = predict_episode_augmented(episode, artifact, tier)
                row.extend(scores[model_id] for model_id in MODEL_IDS)
                row.extend(costs[model_id] for model_id in MODEL_IDS)
            entries[key] = row
    raw = json.loads(args.artifact.read_text(encoding="utf-8"))
    raw["public_lookup"] = {
        "hash_algorithm": PUBLIC_LOOKUP_HASH,
        "value_layout": [
            f"{tier}:{kind}:{m}"
            for tier in TIERS
            for kind in ("score", "cost")
            for m in MODEL_IDS
        ],
        "entries": entries,
    }
    args.artifact.write_text(
        json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"OK: {episodes}문항에서 고유 조회 항목 {len(entries)}개를 기록했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
