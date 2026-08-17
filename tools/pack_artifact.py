# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Move heavy artifact blobs into a lazily-loaded companion resource.

The kNN train vectors/targets and the meta trees are only needed when a
prompt misses the public lookup, yet parsing them dominates container
start-up.  This tool moves them into ``learned-router-heavy.v1.json`` next
to the main artifact and records digest-checked references.  Run after
build_meta_gbm.py (and before or after build_public_lookup.py — the lookup
block is unaffected).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HEAVY_NAME = "learned-router-heavy.v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.artifact.read_text(encoding="utf-8"))
    knn = raw.get("augmentation", {}).get("knn")
    meta = raw.get("meta_gbm")
    if knn is None or meta is None:
        raise SystemExit("augmentation/meta_gbm 블록이 필요합니다.")
    if "train_vectors" not in knn and "trees" not in meta:
        print("이미 분리되어 있습니다.")
        return 0

    heavy_path_existing = args.artifact.with_name(HEAVY_NAME)
    heavy = {}
    if heavy_path_existing.exists():
        heavy = json.loads(heavy_path_existing.read_text(encoding="utf-8"))
    if "train_vectors" in knn:
        heavy["knn_train_vectors"] = knn.pop("train_vectors")
        heavy["knn_train_targets"] = knn.pop("train_targets")
    if "trees" in meta:
        heavy["meta_trees"] = meta.pop("trees")
    if "delta_trees" in meta:
        heavy["meta_delta_trees"] = meta.pop("delta_trees")
    if "ordinal_trees" in meta:
        heavy["meta_ordinal_trees"] = meta.pop("ordinal_trees")
    if "rank_trees" in meta:
        heavy["meta_rank_trees"] = meta.pop("rank_trees")
    heavy_path = args.artifact.with_name(HEAVY_NAME)
    payload = json.dumps(heavy, ensure_ascii=False, separators=(",", ":"))
    heavy_path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    reference = {"resource": HEAVY_NAME, "sha256": digest}
    knn["heavy_ref"] = dict(reference)
    meta["heavy_ref"] = dict(reference)
    args.artifact.write_text(
        json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"OK: heavy {heavy_path.name} ({heavy_path.stat().st_size:,}B, sha256 {digest[:16]}...), "
        f"main {args.artifact.stat().st_size:,}B"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
