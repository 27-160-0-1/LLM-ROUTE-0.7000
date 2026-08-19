# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pathlib
import sys
import unittest

from ossp_router.heuristic import episode_text
from ossp_router.learned_router import feature_items, load_artifact, make_submission
from ossp_router.protocol import load_bundled_policy, load_input, parse_input


ROOT = pathlib.Path(__file__).resolve().parents[1]


class LearnedRouterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = load_artifact(
            ROOT / "src/ossp_router/resources/learned-router.v1.json"
        )
        cls.policy = load_bundled_policy()

    def test_artifact_load_does_not_import_gpu_training_stack(self) -> None:
        self.assertFalse(
            any(name == "cupy" or name.startswith("cupy.") for name in sys.modules)
        )
        self.assertEqual(16414, self.artifact.dimension)
        self.assertEqual(0.9, self.artifact.legacy_blend_weight)

    def test_prompt_and_messages_are_supported(self) -> None:
        batch = parse_input(
            {
                "schema_version": 1,
                "challenge_id": "test",
                "split": "test",
                "episodes": [
                    {"episode_id": "p", "prompt": "Prove that 2 + 2 = 4."},
                    {
                        "episode_id": "m",
                        "messages": [
                            {"role": "system", "content": "Answer briefly."},
                            {"role": "user", "content": "정답만 출력하세요: 3+4"},
                        ],
                    },
                ],
            }
        )
        for episode in batch.episodes:
            items = feature_items(
                episode,
                word_hash_bins=self.artifact.word_hash_bins,
                char_hash_bins=self.artifact.char_hash_bins,
                dense_mean=self.artifact.dense_mean,
                dense_scale=self.artifact.dense_scale,
            )
            self.assertTrue(items)
            self.assertTrue(
                all(0 <= index < self.artifact.dimension for index in items)
            )

    def test_episode_id_and_order_do_not_change_content_decisions(self) -> None:
        original = load_input(ROOT / "data/toy/inputs.json")
        changed = parse_input(
            {
                "schema_version": original.schema_version,
                "challenge_id": original.challenge_id,
                "split": original.split,
                "episodes": [
                    {
                        "episode_id": f"opaque-{index}",
                        **(
                            {"prompt": episode.prompt}
                            if episode.prompt is not None
                            else {
                                "messages": [
                                    {"role": message.role, "content": message.content}
                                    for message in episode.messages or ()
                                ]
                            }
                        ),
                    }
                    for index, episode in enumerate(reversed(original.episodes))
                ],
            }
        )
        for tier in ("fast", "balanced", "premium"):
            first = make_submission(original, self.policy, self.artifact, tier)
            second = make_submission(changed, self.policy, self.artifact, tier)
            first_decisions = {
                row.episode_id: row.model_id for row in first.decisions
            }
            second_decisions = {
                row.episode_id: row.model_id for row in second.decisions
            }
            by_content_first = {
                episode_text(episode): first_decisions[episode.episode_id]
                for episode in original.episodes
            }
            by_content_second = {
                episode_text(episode): second_decisions[episode.episode_id]
                for episode in changed.episodes
            }
            self.assertEqual(by_content_first, by_content_second)


if __name__ == "__main__":
    unittest.main()
