# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Score a (lookup-free) artifact on a split through the real runtime path.

Used for honest held-out evaluation: build the full chain on Train only,
then score Dev here without the public lookup so every prompt goes through
feature extraction -> ensemble -> meta GBM -> allocation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ossp_router import learned_router
from ossp_router.protocol import (
    Decision,
    Submission,
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)
from ossp_router.scoring import score_submissions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    args = parser.parse_args()

    policy = load_bundled_policy()
    raw = json.loads(args.artifact.read_text(encoding="utf-8"))
    had_lookup = bool(raw.pop("public_lookup", None))
    artifact = learned_router.parse_artifact(raw, base_path=args.artifact.parent)
    safety = raw.get("tier_safety_ratios") or {}
    inputs = load_input(args.input)
    outcomes = load_outcomes(args.outcomes)
    print(f"[holdout] {len(inputs.episodes)} episodes, lookup stripped={had_lookup}, safety={safety}")

    submissions = []
    for tier in TIERS:
        predictions = [
            learned_router.predict_episode_augmented(e, artifact, tier)
            for e in inputs.episodes
        ]
        selected, ratio = learned_router.select_models(
            [p[0] for p in predictions],
            [p[1] for p in predictions],
            budget_multiplier=float(policy.tiers[tier].budget_multiplier),
            safety_ratio=float(safety.get(tier, 1.0)),
        )
        submissions.append(Submission(
            schema_version=inputs.schema_version,
            challenge_id=inputs.challenge_id,
            policy_id=policy.policy_id,
            split=inputs.split,
            tier=tier,
            decisions=tuple(
                Decision(e.episode_id, m) for e, m in zip(inputs.episodes, selected)
            ),
        ))
    report = score_submissions(inputs, outcomes, submissions, policy)
    print(f"final={report['final_score']}")
    for tier in TIERS:
        t = report["tiers"][tier]
        print(f"  {tier}: score={t['tier_score']} ratio={t['budget_ratio']} passed={t['budget_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
