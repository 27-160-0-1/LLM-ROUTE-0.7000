# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Merge the public Train/Dev inputs and outcomes into one JSON file.

The repository keeps four overlapping JSON files (materialized inputs and
outcomes per split).  This tool joins each episode's prompt text with its
per-model outcomes and writes a single combined file, so analysis and
training scripts can load one path instead of stitching the pairs together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _merge_split(inputs_path: Path, outcomes_path: Path, split: str) -> dict:
    inputs = _load(inputs_path)
    outcomes = _load(outcomes_path)
    if inputs.get("split") != split or outcomes.get("split") != split:
        raise ValueError(f"{split} 입력/outcome 파일의 split 값이 다릅니다.")
    if inputs.get("challenge_id") != outcomes.get("challenge_id"):
        raise ValueError(f"{split} 입력과 outcome의 challenge_id가 다릅니다.")
    outcome_map = {
        episode["episode_id"]: episode["models"]
        for episode in outcomes["episodes"]
    }
    merged = []
    for episode in inputs["episodes"]:
        episode_id = episode["episode_id"]
        models = outcome_map.pop(episode_id, None)
        if models is None:
            raise ValueError(f"{split} outcome이 없는 문항: {episode_id}")
        row = dict(episode)
        row["models"] = models
        merged.append(row)
    if outcome_map:
        raise ValueError(f"{split} 입력에 없는 outcome: {sorted(outcome_map)[:3]}")
    return {
        "split": split,
        "episode_count": len(merged),
        "episodes": merged,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="공개 Train/Dev 입력과 outcome을 하나의 JSON으로 병합합니다."
    )
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo
    sources = {
        "train_inputs": repo / "data/materialized/train/inputs.json",
        "train_outcomes": repo / "data/train/outcomes.json",
        "dev_inputs": repo / "data/materialized/dev/inputs.json",
        "dev_outcomes": repo / "data/dev/outcomes.json",
    }
    train = _merge_split(sources["train_inputs"], sources["train_outcomes"], "train")
    dev = _merge_split(sources["dev_inputs"], sources["dev_outcomes"], "dev")
    train_inputs = _load(sources["train_inputs"])

    seen_prompts: dict = {}
    duplicate_pairs = []
    for split_record in (train, dev):
        for episode in split_record["episodes"]:
            text = episode.get("prompt")
            if text is None:
                text = "\n".join(
                    message["content"] for message in episode.get("messages", ())
                )
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if key in seen_prompts:
                duplicate_pairs.append(
                    (seen_prompts[key], episode["episode_id"])
                )
            else:
                seen_prompts[key] = episode["episode_id"]

    combined = {
        "schema_version": train_inputs["schema_version"],
        "challenge_id": train_inputs["challenge_id"],
        "description": (
            "공개 Train/Dev의 materialized 입력과 모델별 outcome을 병합한 "
            "단일 분석용 파일입니다. 평가·제출 도구는 기존 분리 파일을 계속 "
            "사용합니다."
        ),
        "sources": {
            name: {"path": str(path.relative_to(repo)), "sha256": _sha256(path)}
            for name, path in sources.items()
        },
        "duplicate_prompt_pairs": [
            {"first": first, "second": second}
            for first, second in duplicate_pairs
        ],
        "splits": {"train": train, "dev": dev},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(combined, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    total = train["episode_count"] + dev["episode_count"]
    print(
        f"OK: {total}문항(train {train['episode_count']}, dev {dev['episode_count']})을 "
        f"{args.output}에 병합했습니다. 중복 프롬프트 쌍 {len(duplicate_pairs)}개."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
