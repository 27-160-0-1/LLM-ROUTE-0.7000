# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Train the portable learned router on an NVIDIA GPU.

The script is intentionally not copied into the submission image.  It uses
NumPy/SciPy to construct a sparse matrix and CuPy LSMR to fit each linear head,
then writes a standard-library-only JSON artifact for CPU inference.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from ossp_router import learned_router, legacy_hash_regex
from ossp_router.heuristic import episode_text
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    InputBatch,
    Outcome,
    OutcomeBatch,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
    policy_sha256,
)
from ossp_router.scoring import score_submissions


GUARD_RATIOS = {"fast": 0.98, "balanced": 0.96, "premium": 0.92}
DEFAULT_LEGACY_BLEND_WEIGHT = 0.75
DEFAULT_CONTEXT_LIMIT = 256


def _load_gpu() -> Tuple[Any, Any, Any, Any, list[Any]]:
    """Load the pip CUDA DLLs before CuPy on Windows."""

    handles: list[Any] = []
    if platform.system() == "Windows":
        nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
        for relative in (
            "cuda_runtime/bin/cudart64_110.dll",
            "cuda_nvrtc/bin/nvrtc-builtins64_117.dll",
            "cuda_nvrtc/bin/nvrtc64_112_0.dll",
            "cublas/lib/x64/cublasLt64_11.dll",
            "cublas/lib/x64/cublas64_11.dll",
            "cusparse/bin/cusparse64_11.dll",
            "cusolver/bin/cusolver64_11.dll",
        ):
            path = nvidia_root / relative
            if path.exists():
                handles.append(ctypes.WinDLL(str(path)))
    import cupy as cp
    import cupyx.scipy.sparse as cupy_sparse
    from cupyx.scipy.sparse.linalg import LinearOperator, lsmr

    cp.cuda.Device(0).use()
    cp.asarray([0.0], dtype=cp.float32)
    return cp, cupy_sparse, LinearOperator, lsmr, handles


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cost(outcome: Outcome, policy: RoutingPolicy) -> float:
    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    return float(
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )


def _targets(inputs: InputBatch, outcomes: OutcomeBatch, policy: RoutingPolicy) -> np.ndarray:
    index = {(row.episode_id, row.model_id): row for row in outcomes.outcomes}
    expected = {
        (episode.episode_id, model_id)
        for episode in inputs.episodes
        for model_id in MODEL_IDS
    }
    if set(index) != expected:
        raise ValueError("outcomes do not form a complete episode/model matrix")
    rows = []
    for episode in inputs.episodes:
        values = [index[(episode.episode_id, model_id)] for model_id in MODEL_IDS]
        rows.append(
            [float(row.score) for row in values]
            + [math.log(_cost(row, policy)) for row in values]
        )
    return np.asarray(rows, dtype=np.float32)


def _feature_matrix(
    inputs: InputBatch,
    *,
    word_bins: int,
    char_bins: int,
    dense_mean: Optional[np.ndarray] = None,
    dense_scale: Optional[np.ndarray] = None,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    raw_dense = np.asarray(
        [learned_router.raw_dense_features(episode) for episode in inputs.episodes],
        dtype=np.float32,
    )
    if dense_mean is None:
        dense_mean = raw_dense.mean(axis=0)
    if dense_scale is None:
        dense_scale = raw_dense.std(axis=0)
        dense_scale = np.where(dense_scale > 1e-6, dense_scale, 1.0).astype(np.float32)
    rows, columns, values = [], [], []
    for row_index, episode in enumerate(inputs.episodes):
        items = learned_router.feature_items(
            episode,
            word_hash_bins=word_bins,
            char_hash_bins=char_bins,
            dense_mean=dense_mean,
            dense_scale=dense_scale,
            raw_dense=raw_dense[row_index],
        )
        for column, value in items.items():
            rows.append(row_index)
            columns.append(column)
            values.append(value)
    dimension = len(learned_router.DENSE_FEATURE_NAMES) + word_bins + char_bins
    matrix = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(len(inputs.episodes), dimension),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix, np.asarray(dense_mean, dtype=np.float32), np.asarray(dense_scale, dtype=np.float32)


def _fit_head_gpu(
    matrix: sparse.csr_matrix,
    target: np.ndarray,
    *,
    alpha: float,
    label: str,
    gpu: Tuple[Any, Any, Any, Any, list[Any]],
    maxiter: int,
    tolerance: float,
) -> Tuple[float, np.ndarray, Mapping[str, Any]]:
    cp, cupy_sparse, LinearOperator, lsmr, _handles = gpu
    chunk_nnz = 750_000
    ranges = []
    start = 0
    while start < matrix.shape[0]:
        target_nnz = int(matrix.indptr[start]) + chunk_nnz
        end = int(np.searchsorted(matrix.indptr, target_nnz, side="right") - 1)
        end = max(start + 1, min(matrix.shape[0], end))
        ranges.append((start, end))
        start = end
    chunks = [(start, end, cupy_sparse.csr_matrix(matrix[start:end])) for start, end in ranges]
    gpu_target = cp.asarray(np.asarray(target, dtype=np.float32))
    target_mean = cp.mean(gpu_target)
    feature_mean = cp.asarray(np.asarray(matrix.mean(axis=0), dtype=np.float32).reshape(-1))
    centered_target = gpu_target - target_mean

    def matvec(vector: Any) -> Any:
        return cp.concatenate([chunk @ vector for _start, _end, chunk in chunks]) - cp.dot(feature_mean, vector)

    def rmatvec(vector: Any) -> Any:
        products = [chunk.T @ vector[start:end] for start, end, chunk in chunks]
        total = products[0]
        for product in products[1:]:
            total = total + product
        return total - feature_mean * cp.sum(vector)

    operator = LinearOperator(matrix.shape, matvec=matvec, rmatvec=rmatvec, dtype=cp.float32)
    print(
        f"[gpu] {label}: rows={matrix.shape[0]} features={matrix.shape[1]} "
        f"nnz={matrix.nnz} chunks={len(chunks)} alpha={alpha}",
        flush=True,
    )
    result = lsmr(
        operator,
        centered_target,
        damp=math.sqrt(alpha),
        atol=tolerance,
        btol=tolerance,
        maxiter=maxiter,
    )
    coefficients_gpu = result[0]
    intercept_gpu = target_mean - cp.dot(feature_mean, coefficients_gpu)
    coefficients = cp.asnumpy(coefficients_gpu).astype(np.float64, copy=False)
    intercept = float(intercept_gpu.item())
    diagnostics = {
        "label": label,
        "backend": "gpu",
        "solver": "cupyx-lsmr",
        "iterations": int(result[2]),
        "stop_code": int(result[1]),
        "residual_norm": float(result[3]),
        "chunks": len(chunks),
    }
    del chunks, operator, coefficients_gpu, intercept_gpu, centered_target, feature_mean, gpu_target
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return intercept, coefficients, diagnostics


def _predict(matrix: sparse.csr_matrix, heads: Sequence[Tuple[float, np.ndarray]]) -> np.ndarray:
    columns = [np.asarray(matrix @ coefficients).reshape(-1) + intercept for intercept, coefficients in heads]
    return np.column_stack(columns)


def _prediction_rows(predictions: np.ndarray) -> Tuple[list[Mapping[str, float]], list[Mapping[str, float]]]:
    score_rows, cost_rows = [], []
    count = len(MODEL_IDS)
    for row in predictions:
        scores = {
            model_id: min(1.0, max(0.0, float(row[index])))
            for index, model_id in enumerate(MODEL_IDS)
        }
        costs = {
            model_id: math.exp(min(50.0, max(-50.0, float(row[count + index]))))
            for index, model_id in enumerate(MODEL_IDS)
        }
        costs[MODEL_IDS[1]] = max(costs[MODEL_IDS[1]], costs[MODEL_IDS[0]] * (1 + 1e-12))
        costs[MODEL_IDS[2]] = max(costs[MODEL_IDS[2]], costs[MODEL_IDS[1]] * (1 + 1e-12))
        score_rows.append(scores)
        cost_rows.append(costs)
    return score_rows, cost_rows


def _legacy_predictions(inputs: InputBatch) -> np.ndarray:
    artifact_path = Path(__file__).resolve().parents[1] / "src" / "ossp_router" / "resources" / "hash-regex-public.v1.json"
    artifact = legacy_hash_regex.load_artifact(artifact_path)
    rows = []
    for episode in inputs.episodes:
        scores, costs = legacy_hash_regex.predict_episode(episode, artifact)
        rows.append(
            [scores[model_id] for model_id in MODEL_IDS]
            + [math.log(costs[model_id]) for model_id in MODEL_IDS]
        )
    return np.asarray(rows, dtype=np.float64)


def _blend_predictions(
    learned: np.ndarray,
    legacy: np.ndarray,
    inputs: InputBatch,
    *,
    blend_weight: float,
    context_limit: int,
) -> np.ndarray:
    result = np.asarray(learned, dtype=np.float64).copy()
    result[:, :] = (
        (1.0 - blend_weight) * result + blend_weight * np.asarray(legacy, dtype=np.float64)
    )
    if blend_weight > 0.0:
        for index, episode in enumerate(inputs.episodes):
            if len(episode_text(episode)) >= context_limit:
                result[index] = legacy[index]
    return result


def _submission(inputs: InputBatch, policy: RoutingPolicy, tier: str, selected: Sequence[str]) -> Submission:
    return Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, model_id)
            for episode, model_id in zip(inputs.episodes, selected)
        ),
    )


def _score_tier(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    tier: str,
    selected: Sequence[str],
) -> Mapping[str, Any]:
    light = tuple(policy.light_model_id for _ in inputs.episodes)
    submissions = [
        _submission(inputs, policy, candidate, selected if candidate == tier else light)
        for candidate in TIERS
    ]
    return score_submissions(inputs, outcomes, submissions, policy)["tiers"][tier]


def _calibrate(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    predictions: np.ndarray,
    grid_size: int,
) -> Tuple[Mapping[str, float], Mapping[str, Any], Mapping[str, Tuple[str, ...]]]:
    scores, costs = _prediction_rows(predictions)
    ratios: Dict[str, float] = {}
    reports: Dict[str, Any] = {}
    selections: Dict[str, Tuple[str, ...]] = {}
    for tier in TIERS:
        multiplier = float(policy.tiers[tier].budget_multiplier)
        minimum = 1.0 / multiplier
        best = None
        for index in range(grid_size):
            safety = minimum + (1.0 - minimum) * index / max(1, grid_size - 1)
            selected, predicted_ratio = learned_router.select_models(
                scores,
                costs,
                budget_multiplier=multiplier,
                safety_ratio=safety,
            )
            report = _score_tier(inputs, outcomes, policy, tier, selected)
            actual_ratio = float(report["budget_ratio"])
            if not report["budget_passed"] or actual_ratio > multiplier * GUARD_RATIOS[tier]:
                continue
            rank = (float(report["tier_score"]), -actual_ratio, -safety)
            if best is None or rank > best[0]:
                best = (rank, safety, predicted_ratio, report, selected)
        if best is None:
            selected = tuple(MODEL_IDS[0] for _ in inputs.episodes)
            best = ((0.0, -1.0, -minimum), minimum, 1.0, _score_tier(inputs, outcomes, policy, tier, selected), selected)
        ratios[tier] = best[1]
        reports[tier] = {
            "safety_ratio": best[1],
            "predicted_budget_ratio": best[2],
            "actual_budget_ratio": best[3]["budget_ratio"],
            "tier_score": best[3]["tier_score"],
            "model_counts": best[3]["model_counts"],
            "guard_ratio": GUARD_RATIOS[tier],
        }
        selections[tier] = best[4]
    return ratios, reports, selections


def _head_dict(head: Tuple[float, np.ndarray]) -> Mapping[str, Any]:
    return {"intercept": head[0], "coefficients": [float(value) for value in head[1]]}


def _artifact_dict(
    *,
    word_bins: int,
    char_bins: int,
    dense_mean: np.ndarray,
    dense_scale: np.ndarray,
    heads: Sequence[Tuple[float, np.ndarray]],
    safety: Mapping[str, float],
    policy: RoutingPolicy,
    summary: Mapping[str, Any],
    blend_weight: float,
    context_limit: int,
) -> Mapping[str, Any]:
    count = len(MODEL_IDS)
    return {
        "artifact_type": learned_router.ARTIFACT_TYPE,
        "schema_version": 1,
        "feature_version": learned_router.FEATURE_VERSION,
        "hash_algorithm": "fnv1a64-signed-word12-char345",
        "word_hash_bins": word_bins,
        "char_hash_bins": char_bins,
        "dense_feature_names": list(learned_router.DENSE_FEATURE_NAMES),
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "dense_mean": [float(value) for value in dense_mean],
        "dense_scale": [float(value) for value in dense_scale],
        "score_heads": {model_id: _head_dict(heads[index]) for index, model_id in enumerate(MODEL_IDS)},
        "log_cost_heads": {model_id: _head_dict(heads[count + index]) for index, model_id in enumerate(MODEL_IDS)},
        "tier_safety_ratios": {tier: float(safety[tier]) for tier in TIERS},
        "legacy_blend_weight": float(blend_weight),
        "learned_context_limit_chars": int(context_limit),
        "legacy_artifact": "hash-regex-public.v1.json",
        "training_summary": dict(summary),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> Mapping[str, Any]:
    policy = load_bundled_policy()
    train_inputs, train_outcomes = load_input(args.input), load_outcomes(args.outcomes)
    dev_inputs, dev_outcomes = load_input(args.validation_input), load_outcomes(args.validation_outcomes)
    print("[features] building train matrix", flush=True)
    train_matrix, dense_mean, dense_scale = _feature_matrix(
        train_inputs, word_bins=args.word_bins, char_bins=args.char_bins
    )
    print("[features] building dev matrix", flush=True)
    dev_matrix, _, _ = _feature_matrix(
        dev_inputs,
        word_bins=args.word_bins,
        char_bins=args.char_bins,
        dense_mean=dense_mean,
        dense_scale=dense_scale,
    )
    targets = _targets(train_inputs, train_outcomes, policy)
    legacy_dev_predictions = _legacy_predictions(dev_inputs)
    gpu = _load_gpu()
    cp = gpu[0]
    device = cp.cuda.runtime.getDeviceProperties(0)
    device_name = device["name"].decode() if isinstance(device["name"], bytes) else str(device["name"])
    candidates = []
    for alpha in args.alphas:
        diagnostics, heads = [], []
        for index in range(targets.shape[1]):
            label = ("score-" if index < len(MODEL_IDS) else "log-cost-") + MODEL_IDS[index % len(MODEL_IDS)]
            intercept, coefficients, diagnostic = _fit_head_gpu(
                train_matrix,
                targets[:, index],
                alpha=alpha,
                label=label,
                gpu=gpu,
                maxiter=args.gpu_maxiter,
                tolerance=args.gpu_tolerance,
            )
            heads.append((intercept, coefficients))
            diagnostics.append(diagnostic)
        learned_dev_predictions = _predict(dev_matrix, heads)
        for blend_weight in args.blend_weights:
            for context_limit in args.context_limits:
                if blend_weight == 0.0 and context_limit != args.context_limits[0]:
                    continue  # limit is irrelevant without a legacy blend
                dev_predictions = _blend_predictions(
                    learned_dev_predictions,
                    legacy_dev_predictions,
                    dev_inputs,
                    blend_weight=blend_weight,
                    context_limit=context_limit,
                )
                safety, calibration, selections = _calibrate(
                    dev_inputs, dev_outcomes, policy, dev_predictions, args.safety_grid_size
                )
                submissions = [_submission(dev_inputs, policy, tier, selections[tier]) for tier in TIERS]
                dev_report = score_submissions(dev_inputs, dev_outcomes, submissions, policy)
                print(
                    f"[candidate] alpha={alpha} blend={blend_weight} limit={context_limit} "
                    f"final={dev_report['final_score']} "
                    f"fast={dev_report['tiers']['fast']['tier_score']} "
                    f"balanced={dev_report['tiers']['balanced']['tier_score']}",
                    flush=True,
                )
                candidates.append((
                    float(dev_report["final_score"]),
                    -alpha,
                    alpha,
                    heads,
                    diagnostics,
                    safety,
                    calibration,
                    dev_report,
                    blend_weight,
                    context_limit,
                ))
    best = max(candidates, key=lambda item: item[:2])
    _, _, alpha, heads, diagnostics, safety, calibration, dev_report, blend_weight, context_limit = best
    summary = {
        "num_train_episodes": len(train_inputs.episodes),
        "num_dev_episodes": len(dev_inputs.episodes),
        "training_backend": "gpu",
        "solver": "cupyx-lsmr",
        "gpu_device": device_name,
        "cupy_version": cp.__version__,
        "ridge_alpha": alpha,
        "alpha_candidates": list(args.alphas),
        "train_input_sha256": _sha256(args.input),
        "train_outcomes_sha256": _sha256(args.outcomes),
        "dev_input_sha256": _sha256(args.validation_input),
        "dev_outcomes_sha256": _sha256(args.validation_outcomes),
        "content_features_only": True,
        "uses_episode_id_or_order": False,
        "guard_ratios": GUARD_RATIOS,
        "legacy_blend_weight": blend_weight,
        "learned_context_limit_chars": context_limit,
        "blend_weight_candidates": list(args.blend_weights),
        "context_limit_candidates": list(args.context_limits),
    }
    artifact = _artifact_dict(
        word_bins=args.word_bins,
        char_bins=args.char_bins,
        dense_mean=dense_mean,
        dense_scale=dense_scale,
        heads=heads,
        safety=safety,
        policy=policy,
        summary=summary,
        blend_weight=blend_weight,
        context_limit=context_limit,
    )
    learned_router.parse_artifact(artifact)
    report = {
        "report_type": "ossp-learned-router-gpu-training-v1",
        "training_summary": summary,
        "feature_dimension": train_matrix.shape[1],
        "train_nnz": train_matrix.nnz,
        "gpu_fit_diagnostics": diagnostics,
        "validation_safety_calibration": calibration,
        "validation_score": dev_report,
        "candidate_scores": {
            f"alpha={item[2]:g},blend={item[8]:g},limit={item[9]}": str(item[7]["final_score"])
            for item in candidates
        },
    }
    _write_json(args.artifact, artifact)
    _write_json(args.report, report)
    return report


def _float_list(value: str) -> Tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if not result or any(not math.isfinite(item) or item <= 0 for item in result):
        raise argparse.ArgumentTypeError("alphas must be positive finite numbers")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--validation-outcomes", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--word-bins", type=int, default=4096)
    parser.add_argument("--char-bins", type=int, default=4096)
    parser.add_argument("--alphas", type=_float_list, default=_float_list("1,10,100"))
    parser.add_argument(
        "--blend-weights",
        type=lambda value: tuple(float(item) for item in value.split(",")),
        default=(DEFAULT_LEGACY_BLEND_WEIGHT,),
    )
    parser.add_argument(
        "--context-limits",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        default=(DEFAULT_CONTEXT_LIMIT,),
    )
    parser.add_argument("--gpu-maxiter", type=int, default=300)
    parser.add_argument("--gpu-tolerance", type=float, default=1e-5)
    parser.add_argument("--safety-grid-size", type=int, default=121)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        report = train(_parser().parse_args(argv))
    except Exception as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 2
    print(f"OK: GPU-trained artifact, Dev score={report['validation_score']['final_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
