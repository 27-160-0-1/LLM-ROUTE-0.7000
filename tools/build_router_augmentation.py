# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Add family-mean and public-train kNN blending tables to the artifact.

Builds content-only tables from public Train (family target means, hashed
tf-idf vectors, per-episode targets), injects them as an ``augmentation``
block, recalibrates tier safety ratios on public Dev with the exact runtime
prediction path, and rewrites the artifact.  Run BEFORE
``build_public_lookup.py`` so lookup rows reflect the augmented predictor.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from ossp_router import learned_router, similarity
from ossp_router.heuristic import episode_text
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

import os as _os
FAMILY_BLEND_WEIGHT = float(_os.environ.get("ROUTER_FAM_W", 0.3))       # E43 override hook
KNN_CONF_SCALE = float(_os.environ.get("ROUTER_CONF_SCALE", 0.4))       # E43 override hook
MIN_FAMILY_ROWS = 8
VALUE_DECIMALS = 5
GUARD_RATIOS = {"fast": 0.98, "balanced": 0.96, "premium": 0.92}


def _cost(outcome, policy) -> float:
    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    return float(
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )


def _targets(inputs, outcomes, policy):
    index = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    rows = []
    for episode in inputs.episodes:
        values = [index[(episode.episode_id, m)] for m in MODEL_IDS]
        rows.append(
            [float(v.score) for v in values]
            + [math.log(_cost(v, policy)) for v in values]
        )
    return rows


def _calibrate_safety(dev_inputs, dev_outcomes, policy, artifact, grid_size=121):
    """Pick per-tier safety ratios maximizing actual dev score under guards."""

    from ossp_router.scoring import score_submissions
    from ossp_router.protocol import Decision, Submission

    predictions = [
        learned_router.predict_episode_augmented(episode, artifact)
        for episode in dev_inputs.episodes
    ]
    scores = [row[0] for row in predictions]
    costs = [row[1] for row in predictions]

    def tier_report(tier, selected):
        light = tuple(policy.light_model_id for _ in dev_inputs.episodes)
        submissions = [
            Submission(
                schema_version=dev_inputs.schema_version,
                challenge_id=dev_inputs.challenge_id,
                policy_id=policy.policy_id,
                split=dev_inputs.split,
                tier=candidate,
                decisions=tuple(
                    Decision(e.episode_id, m)
                    for e, m in zip(
                        dev_inputs.episodes,
                        selected if candidate == tier else light,
                    )
                ),
            )
            for candidate in TIERS
        ]
        return score_submissions(dev_inputs, dev_outcomes, submissions, policy)["tiers"][tier]

    ratios, reports = {}, {}
    for tier in TIERS:
        multiplier = float(policy.tiers[tier].budget_multiplier)
        minimum = 1.0 / multiplier
        best = None
        for step in range(grid_size):
            safety = minimum + (1.0 - minimum) * step / (grid_size - 1)
            selected, _predicted = learned_router.select_models(
                scores, costs, budget_multiplier=multiplier, safety_ratio=safety
            )
            report = tier_report(tier, selected)
            ratio = float(report["budget_ratio"])
            if not report["budget_passed"] or ratio > multiplier * GUARD_RATIOS[tier]:
                continue
            rank = (float(report["tier_score"]), -ratio, -safety)
            if best is None or rank > best[0]:
                best = (rank, safety, report)
        if best is None:
            best = ((0.0, 0.0, 0.0), minimum, None)
        ratios[tier] = best[1]
        reports[tier] = {
            "safety_ratio": best[1],
            "tier_score": str(best[2]["tier_score"]) if best[2] else "0",
            "actual_budget_ratio": str(best[2]["budget_ratio"]) if best[2] else "n/a",
            "model_counts": best[2]["model_counts"] if best[2] else {},
        }
    return ratios, reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-outcomes", type=Path, required=True)
    parser.add_argument("--dev-input", type=Path, required=True)
    parser.add_argument("--dev-outcomes", type=Path, required=True)
    args = parser.parse_args()

    policy = load_bundled_policy()
    train_inputs = load_input(args.train_input)
    train_outcomes = load_outcomes(args.train_outcomes)
    dev_inputs = load_input(args.dev_input)
    dev_outcomes = load_outcomes(args.dev_outcomes)

    train_texts = [episode_text(e) for e in train_inputs.episodes]
    targets = _targets(train_inputs, train_outcomes, policy)

    families = [similarity.classify_family(t) for t in train_texts]
    rows_by_family = defaultdict(list)
    for name, row in zip(families, targets):
        rows_by_family[name].append(row)
    width = 2 * len(MODEL_IDS)
    global_mean = [
        math.fsum(row[i] for row in targets) / len(targets) for i in range(width)
    ]
    family_means = {}
    for name in similarity.FAMILY_NAMES:
        rows = rows_by_family.get(name, [])
        family_means[name] = (
            [math.fsum(r[i] for r in rows) / len(rows) for i in range(width)]
            if len(rows) >= MIN_FAMILY_ROWS
            else list(global_mean)
        )

    frequencies, total = similarity.document_frequencies(train_texts)
    idf = similarity.idf_table(frequencies, total)
    vectors = []
    for text in train_texts:
        vector = similarity.tfidf_vector(
            text, idf, top_components=similarity.TOP_COMPONENTS
        )
        vectors.append(
            [[index, round(value, VALUE_DECIMALS)] for index, value in sorted(vector.items())]
        )

    raw = json.loads(args.artifact.read_text(encoding="utf-8"))
    raw["augmentation"] = {
        "family_blend_weight": FAMILY_BLEND_WEIGHT,
        "knn_conf_scale": KNN_CONF_SCALE,
        "family_means": family_means,
        "knn": {
            "hash_bits": similarity.HASH_BITS,
            "ngram_sizes": list(similarity.NGRAM_SIZES),
            "text_limit": similarity.TEXT_LIMIT,
            "top_components": similarity.TOP_COMPONENTS,
            "neighbors": similarity.NEIGHBORS,
            "idf": {str(index): value for index, value in sorted(idf.items())},
            "train_vectors": vectors,
            "train_targets": targets,
        },
    }
    raw.pop("public_lookup", None)  # stale rows; rebuild with build_public_lookup

    artifact = learned_router.parse_artifact(raw)
    ratios, reports = _calibrate_safety(dev_inputs, dev_outcomes, policy, artifact)
    raw["tier_safety_ratios"] = {tier: ratios[tier] for tier in TIERS}
    args.artifact.write_text(
        json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    for tier in TIERS:
        print(f"{tier}: {reports[tier]}")
    print("OK: augmentation 블록과 재보정된 안전계수를 기록했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
