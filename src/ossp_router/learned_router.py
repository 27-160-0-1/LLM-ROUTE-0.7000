# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Portable prompt-only router trained by ``train_learned_router_gpu.py``.

Training may use CUDA, SciPy, and NumPy.  This module deliberately depends only
on the Python standard library and the challenge protocol, so the submitted
container remains small and runs on the official CPU-only sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import re
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from . import similarity
from .heuristic import episode_text, extract_features, write_submission_atomic
from .protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_json,
    load_policy,
    parse_submission,
    policy_sha256,
    submission_to_dict,
)


ARTIFACT_TYPE = "ossp-learned-hash-linear-v1"
FEATURE_VERSION = 1
DEFAULT_ARTIFACT = "learned-router.v1.json"
LEARNED_CONTEXT_LIMIT_CHARS = 256
PUBLIC_LOOKUP_HASH = "sha256-utf8-prompt-text"
_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_MASK64 = (1 << 64) - 1
_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
_FORMAL = re.compile(
    r"\b(?:prove|derive|theorem|lemma|counterexample|induction|proof)\b|"
    r"(?:증명|유도|정리|귀류|반례)", re.IGNORECASE
)
_PROGRAM = re.compile(
    r"```|\b(?:traceback|exception|complexity|big[- ]?o|runtime|compile|"
    r"stdout|stdin|return|function|def|class|import)\b", re.IGNORECASE
)
_CONSTRAINT = re.compile(
    r"\b(?:exactly|at least|at most|must|only|without|choose all|"
    r"not exceed)\b|(?:정확히|이상|이하|반드시|오직|제외)", re.IGNORECASE
)
_TRANSFORM = re.compile(
    r"\b(?:summari[sz]e|rewrite|translate|list|extract|paraphrase)\b|"
    r"(?:요약|바꿔|번역|나열|추출)", re.IGNORECASE
)
_MULTIPLE_CHOICE = re.compile(
    r"(?:^|\n)\s*(?:\(?[A-Ea-e1-5]\)|[A-Ea-e1-5][.)])\s+|"
    r"\b(?:choices?|options?|multiple choice|select the correct)\b",
    re.IGNORECASE,
)
_ANSWER_FORMAT = re.compile(
    r"\b(?:answer|respond|output|return)\b.{0,35}\b(?:only|format|json|"
    r"integer|number|letter|yes|no|true|false)\b|"
    r"(?:정답|답변|출력).{0,25}(?:형식|숫자|문자|예|아니오|참|거짓)",
    re.IGNORECASE | re.DOTALL,
)
_STORY_MATH = re.compile(
    r"\b(?:how many|how much|total|each|per|remaining|altogether|"
    r"percent|probability)\b|(?:모두|각각|남은|확률|퍼센트)", re.IGNORECASE
)
_TRUTH = re.compile(
    r"\b(?:true or false|is the (?:statement|claim)|yes or no|"
    r"truthful|entails?|contradicts?)\b|(?:참인가|거짓인가|옳은|틀린)",
    re.IGNORECASE,
)


DENSE_FEATURE_NAMES = (
    "log_character_count",
    "log_word_count",
    "log_sentence_count",
    "log_message_count",
    "hangul_ratio",
    "log_code_marker_count",
    "log_math_marker_count",
    "numeric_density",
    "long_context_8k",
    "log_reasoning_marker_count",
    "formal_reasoning",
    "program_analysis",
    "log_multi_constraint_count",
    "simple_transform",
    "ascii_ratio",
    "log_newline_count",
    "punctuation_density",
    "log_digit_run_count",
    "multiple_choice",
    "answer_format_constraint",
    "log_question_count",
    "table_like",
    "latex_like",
    "json_or_xml_like",
    "story_math",
    "truth_or_entailment",
    "context_2k",
    "context_16k",
    "context_48k",
    "average_word_length",
)


@dataclass(frozen=True)
class LinearHead:
    intercept: float
    coefficients: Tuple[float, ...]


def _load_heavy_resource(reference: Mapping[str, Any], base_path: Optional[Path]) -> Any:
    """Load and digest-check the lazily-referenced heavy resource file."""

    name = reference.get("resource")
    expected = reference.get("sha256")
    if not isinstance(name, str) or not isinstance(expected, str):
        raise ProtocolError("heavy resource reference is malformed")
    if base_path is not None:
        data = (base_path / name).read_bytes()
    else:
        data = resources.files("ossp_router.resources").joinpath(name).read_bytes()
    import hashlib as _hashlib

    if _hashlib.sha256(data).hexdigest() != expected:
        raise ProtocolError("heavy resource digest mismatch")
    return json.loads(data.decode("utf-8"))


class RouterAugmentation:
    """Family-mean and public-train kNN blending on top of the linear router.

    The kNN vectors may live in a lazily-loaded companion resource so runs
    answered entirely by the public lookup never pay for parsing or building
    the inverted index.
    """

    def __init__(
        self,
        *,
        family_blend_weight: float,
        family_means: Mapping[str, Tuple[float, ...]],
        conf_scale: float,
        idf: Mapping[int, float],
        vectors: Optional[Sequence[Mapping[int, float]]],
        targets: Optional[Sequence[Tuple[float, ...]]],
        heavy_reference: Optional[Mapping[str, Any]] = None,
        heavy_base_path: Optional[Path] = None,
    ) -> None:
        self.family_blend_weight = family_blend_weight
        self.family_means = dict(family_means)
        self.conf_scale = conf_scale
        self.idf = dict(idf)
        self._vectors = list(vectors) if vectors is not None else None
        self._targets = list(targets) if targets is not None else None
        self._heavy_reference = heavy_reference
        self._heavy_base_path = heavy_base_path
        self._index: Optional[similarity.KnnIndex] = None

    def _materialize(self) -> None:
        if self._vectors is not None:
            return
        heavy = _load_heavy_resource(self._heavy_reference, self._heavy_base_path)
        raw_vectors = heavy.get("knn_train_vectors")
        raw_targets = heavy.get("knn_train_targets")
        if not isinstance(raw_vectors, list) or not isinstance(raw_targets, list):
            raise ProtocolError("heavy resource kNN tables are malformed")
        self._vectors = [
            {int(pair[0]): float(pair[1]) for pair in document}
            for document in raw_vectors
        ]
        self._targets = [tuple(float(v) for v in row) for row in raw_targets]

    @property
    def index(self) -> similarity.KnnIndex:
        if self._index is None:
            self._materialize()
            self._index = similarity.KnnIndex(self._vectors, self._targets)
        return self._index


@dataclass(frozen=True)
class LearnedArtifact:
    word_hash_bins: int
    char_hash_bins: int
    dense_mean: Tuple[float, ...]
    dense_scale: Tuple[float, ...]
    score_heads: Mapping[str, LinearHead]
    log_cost_heads: Mapping[str, LinearHead]
    tier_safety_ratios: Mapping[str, float]
    legacy_blend_weight: float
    legacy_artifact: str
    policy_id: str
    policy_digest: str
    training_summary: Mapping[str, Any]
    public_lookup: Mapping[str, Tuple[float, ...]]
    context_limit_chars: int
    augmentation: Optional[RouterAugmentation]
    meta_gbm: Optional["MetaGbm"]

    @property
    def dimension(self) -> int:
        return len(DENSE_FEATURE_NAMES) + self.word_hash_bins + self.char_hash_bins


def _stable_hash(value: str) -> int:
    digest = _FNV_OFFSET
    for byte in value.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _MASK64
    return digest


def _raw_dense_from_context(
    episode: Episode,
    text: str,
    basic: Any,
    tokens: Sequence[str],
) -> Tuple[float, ...]:
    # str.split() with no separator splits on exactly the characters for which
    # str.isspace() is true, so the joined remainder is the non-space text; the
    # ASCII count is the byte length after dropping non-ASCII code points.  Same
    # values as the per-character generators, an order of magnitude faster.
    nonspace_text = "".join(text.split())
    nonspace = max(1, len(nonspace_text))
    ascii_count = len(nonspace_text.encode("ascii", "ignore"))
    punctuation = len(nonspace_text) - sum(map(str.isalnum, nonspace_text))
    alphabetic = [word for word in tokens if any(map(str.isalpha, word))]
    average_word_length = (
        sum(len(word) for word in alphabetic) / max(1, len(alphabetic))
    )
    probe = text
    return (
        math.log1p(basic.character_count),
        math.log1p(basic.word_count),
        math.log1p(basic.sentence_count),
        math.log1p(basic.message_count),
        basic.hangul_ratio,
        math.log1p(basic.code_marker_count),
        math.log1p(basic.math_marker_count),
        basic.numeric_density,
        float(basic.character_count >= 8_000),
        math.log1p(basic.reasoning_marker_count),
        float(bool(_FORMAL.search(probe))),
        float(bool(_PROGRAM.search(probe))),
        math.log1p(len(_CONSTRAINT.findall(probe))),
        float(bool(_TRANSFORM.search(probe))),
        ascii_count / nonspace,
        math.log1p(text.count("\n")),
        punctuation / nonspace,
        math.log1p(len(_DIGITS.findall(text))),
        float(bool(_MULTIPLE_CHOICE.search(probe))),
        float(bool(_ANSWER_FORMAT.search(probe))),
        math.log1p(text.count("?") + text.count("？")),
        float("|" in text and text.count("\n") >= 2),
        float(bool(re.search(r"\\(?:frac|sum|int|sqrt|begin)|\$[^$]+\$", probe))),
        float(bool(re.search(r"(?:^|\n)\s*[\[{<].{0,80}[:>]", probe))),
        float(bool(_STORY_MATH.search(probe))),
        float(bool(_TRUTH.search(probe))),
        float(basic.character_count >= 2_000),
        float(basic.character_count >= 16_000),
        float(basic.character_count >= 48_000),
        average_word_length,
    )


def raw_dense_features(episode: Episode) -> Tuple[float, ...]:
    text = episode_text(episode)
    basic = extract_features(episode)
    tokens = _normalized_tokens(text)
    return _raw_dense_from_context(episode, text, basic, tokens)


# Prompt vocabulary repeats heavily across a batch; memoizing normalization
# and FNV digests removes most per-byte hashing work.  Caps bound memory on
# adversarially diverse inputs; overflow computes directly, identically.
_CACHE_LIMIT = 1_000_000
_TOKEN_NORMALIZATION: Dict[str, str] = {}
_VALUE_DIGESTS: Dict[str, int] = {}


def _normalized_tokens(text: str) -> Tuple[str, ...]:
    tokens = _TOKEN.findall(text)
    cache = _TOKEN_NORMALIZATION
    result = list(map(cache.get, tokens))
    if None in result:
        for index, normalized in enumerate(result):
            if normalized is None:
                token = tokens[index]
                normalized = token.casefold()
                if normalized.isdecimal():
                    normalized = "<number>"
                if len(cache) < _CACHE_LIMIT:
                    cache[token] = normalized
                result[index] = normalized
    return tuple(result)


def _normalized_char_text(text: str, limit: int = 6_000) -> str:
    normalized = _SPACE.sub(" ", _DIGITS.sub("0", text.casefold())).strip()
    if len(normalized) > limit:
        half = limit // 2
        normalized = normalized[:half] + " … " + normalized[-half:]
    return normalized


def _hashed_block(values: Iterable[str], bins: int) -> Dict[int, float]:
    counts: Dict[int, float] = {}
    cache = _VALUE_DIGESTS
    # Aggregating repeats first and adding ±count once produces the same
    # integer-valued floats as adding ±1 per occurrence.
    for value, count in Counter(values).items():
        digest = cache.get(value)
        if digest is None:
            digest = _stable_hash(value)
            if len(cache) < _CACHE_LIMIT:
                cache[value] = digest
        index = digest & (bins - 1)
        counts[index] = counts.get(index, 0.0) + (
            -float(count) if digest & (1 << 63) else float(count)
        )
    norm = math.sqrt(math.fsum(value * value for value in counts.values()))
    if not norm:
        return {}
    return {index: value / norm for index, value in counts.items() if value}


def feature_items(
    episode: Episode,
    *,
    word_hash_bins: int,
    char_hash_bins: int,
    dense_mean: Sequence[float],
    dense_scale: Sequence[float],
    raw_dense: Optional[Sequence[float]] = None,
    prepared_text: Optional[str] = None,
    prepared_tokens: Optional[Sequence[str]] = None,
) -> Dict[int, float]:
    """Return a sparse, content-only feature row keyed by column number."""

    for value, label in ((word_hash_bins, "word_hash_bins"), (char_hash_bins, "char_hash_bins")):
        if value < 16 or value > 16_384 or value & (value - 1):
            raise ValueError(f"{label} must be a power of two in [16, 16384]")
    text = episode_text(episode) if prepared_text is None else prepared_text
    tokens = (
        _normalized_tokens(text)
        if prepared_tokens is None
        else tuple(prepared_tokens)
    )
    dense = (
        raw_dense_features(episode)
        if raw_dense is None
        else tuple(float(value) for value in raw_dense)
    )
    result = {
        index: (value - dense_mean[index]) / dense_scale[index]
        for index, value in enumerate(dense)
        if (value - dense_mean[index]) / dense_scale[index] != 0.0
    }
    word_values = (f"w1:{token}" for token in tokens)
    word_bigrams = (
        f"w2:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:])
    )
    word_offset = len(DENSE_FEATURE_NAMES)
    for index, value in _hashed_block((*word_values, *word_bigrams), word_hash_bins).items():
        result[word_offset + index] = value
    char_text = _normalized_char_text(text)
    char_values = (
        f"c{size}:{char_text[start:start + size]}"
        for size in (3, 4, 5)
        # A deterministic stride keeps most local-template signal while making
        # the full 2,640-row ARM64 workload comfortably cheaper.
        for start in range(0, max(0, len(char_text) - size + 1), 3)
    )
    char_offset = word_offset + word_hash_bins
    for index, value in _hashed_block(char_values, char_hash_bins).items():
        result[char_offset + index] = value
    return result


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ProtocolError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{label} must be finite")
    return result


def _vector(value: Any, length: int, label: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ProtocolError(f"{label} must contain {length} numbers")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


class MetaGbm:
    """Exported HistGradientBoosting heads stacked over the augmented row.

    Tree arrays may live in the lazily-loaded heavy resource; they are only
    parsed when a prediction actually misses the public lookup.
    """

    def __init__(
        self,
        *,
        blend_weights: Mapping[str, float],
        baselines: Tuple[float, ...],
        trees: Optional[Tuple[Any, ...]],
        knn_fallback_row: Tuple[float, ...],
        gain_alpha: float = 0.0,
        delta_baselines: Tuple[float, ...] = (),
        delta_trees: Optional[Tuple[Any, ...]] = None,
        ordinal_thresholds: Tuple[float, ...] = (),
        ordinal_baselines: Tuple[float, ...] = (),
        ordinal_trees: Optional[Tuple[Any, ...]] = None,
        rank_beta: float = 0.0,
        rank_baselines: Tuple[float, ...] = (),
        rank_luts: Tuple[Tuple[float, ...], ...] = (),
        rank_floors: Tuple[float, ...] = (),
        rank_trees: Optional[Tuple[Any, ...]] = None,
        heavy_reference: Optional[Mapping[str, Any]] = None,
        heavy_base_path: Optional[Path] = None,
    ) -> None:
        self.blend_weights = dict(blend_weights)
        self.baselines = baselines
        self.knn_fallback_row = knn_fallback_row
        self.gain_alpha = gain_alpha
        self.delta_baselines = delta_baselines
        self.ordinal_thresholds = ordinal_thresholds
        self.ordinal_baselines = ordinal_baselines
        self.rank_beta = rank_beta
        self.rank_baselines = rank_baselines
        self.rank_luts = rank_luts
        self.rank_floors = rank_floors
        self._trees = trees
        self._delta_trees = delta_trees
        self._ordinal_trees = ordinal_trees
        self._rank_trees = rank_trees
        self._heavy_reference = heavy_reference
        self._heavy_base_path = heavy_base_path

    @staticmethod
    def _normalize_trees(trees_raw: Any, width: int, allow_empty: bool = False) -> Tuple[Any, ...]:
        if not isinstance(trees_raw, list) or len(trees_raw) != width:
            raise ProtocolError("meta_gbm.trees must hold one tree list per target")
        trees = []
        for target_trees in trees_raw:
            if not isinstance(target_trees, list):
                raise ProtocolError("meta_gbm tree groups must be arrays")
            if allow_empty and not target_trees:
                trees.append(())
                continue
            normalized_trees = []
            for tree in target_trees:
                if not isinstance(tree, list) or not tree:
                    raise ProtocolError("meta_gbm trees must be non-empty arrays")
                normalized = []
                for node in tree:
                    if not isinstance(node, list) or len(node) != 6:
                        raise ProtocolError("meta_gbm trees must hold 6-field nodes")
                    normalized.append((
                        int(node[0]), float(node[1]), int(node[2]),
                        float(node[3]), int(node[4]), int(node[5]),
                    ))
                normalized_trees.append(tuple(normalized))
            trees.append(tuple(normalized_trees))
        return tuple(trees)

    def _load_heavy_trees(self) -> None:
        heavy = _load_heavy_resource(self._heavy_reference, self._heavy_base_path)
        self._trees = self._normalize_trees(heavy.get("meta_trees"), 2 * len(MODEL_IDS))
        if self.gain_alpha > 0.0:
            self._delta_trees = self._normalize_trees(heavy.get("meta_delta_trees"), 2)
        if self.ordinal_thresholds:
            self._ordinal_trees = self._normalize_trees(
                heavy.get("meta_ordinal_trees"),
                len(MODEL_IDS) * len(self.ordinal_thresholds),
                allow_empty=True,
            )
        if self.rank_beta > 0.0:
            self._rank_trees = self._normalize_trees(heavy.get("meta_rank_trees"), 2)

    @property
    def trees(self) -> Tuple[Any, ...]:
        if self._trees is None:
            self._load_heavy_trees()
        return self._trees

    @property
    def delta_trees(self) -> Tuple[Any, ...]:
        if self._delta_trees is None:
            self._load_heavy_trees()
        return self._delta_trees

    @property
    def ordinal_trees(self) -> Tuple[Any, ...]:
        if self._ordinal_trees is None:
            self._load_heavy_trees()
        return self._ordinal_trees

    @property
    def rank_trees(self) -> Tuple[Any, ...]:
        if self._rank_trees is None:
            self._load_heavy_trees()
        return self._rank_trees


def _parse_meta_gbm(value: Any, base_path: Optional[Path] = None) -> Optional[MetaGbm]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProtocolError("meta_gbm must be a JSON object")
    width = 2 * len(MODEL_IDS)
    blend_raw = value.get("blend_weight")
    if isinstance(blend_raw, dict):
        if set(blend_raw) != set(TIERS):
            raise ProtocolError("meta_gbm.blend_weight must cover every tier")
        blends = {
            tier: _number(blend_raw[tier], f"meta_gbm.blend_weight.{tier}")
            for tier in TIERS
        }
    else:
        scalar = _number(blend_raw, "meta_gbm.blend_weight")
        blends = {tier: scalar for tier in TIERS}
    if any(not 0.0 <= item <= 1.0 for item in blends.values()):
        raise ProtocolError("meta_gbm.blend_weight must lie in [0, 1]")
    baselines = _vector(value.get("baselines"), width, "meta_gbm.baselines")
    fallback = _vector(value.get("knn_fallback_row"), width, "meta_gbm.knn_fallback_row")
    gain_alpha = _number(value.get("gain_alpha", 0.0), "meta_gbm.gain_alpha")
    if not 0.0 <= gain_alpha <= 1.0:
        raise ProtocolError("meta_gbm.gain_alpha must lie in [0, 1]")
    delta_baselines: Tuple[float, ...] = ()
    if gain_alpha > 0.0:
        delta_baselines = _vector(
            value.get("delta_baselines"), 2, "meta_gbm.delta_baselines"
        )
    ordinal_thresholds_raw = value.get("ordinal_thresholds") or []
    ordinal_thresholds = tuple(float(t) for t in ordinal_thresholds_raw)
    ordinal_baselines: Tuple[float, ...] = ()
    if ordinal_thresholds:
        ordinal_baselines = _vector(
            value.get("ordinal_baselines"),
            len(MODEL_IDS) * len(ordinal_thresholds),
            "meta_gbm.ordinal_baselines",
        )
    rank_beta = _number(value.get("rank_beta", 0.0), "meta_gbm.rank_beta")
    if not 0.0 <= rank_beta <= 1.0:
        raise ProtocolError("meta_gbm.rank_beta must lie in [0, 1]")
    rank_baselines: Tuple[float, ...] = ()
    rank_luts: Tuple[Tuple[float, ...], ...] = ()
    rank_floors: Tuple[float, ...] = ()
    if rank_beta > 0.0:
        rank_baselines = _vector(value.get("rank_baselines"), 2, "meta_gbm.rank_baselines")
        rank_floors = _vector(value.get("rank_floors"), 2, "meta_gbm.rank_floors")
        luts_raw = value.get("rank_luts")
        if not isinstance(luts_raw, list) or len(luts_raw) != 2:
            raise ProtocolError("meta_gbm.rank_luts must hold 2 quantile tables")
        rank_luts_list = []
        for index, lut in enumerate(luts_raw):
            if not isinstance(lut, list) or len(lut) < 2:
                raise ProtocolError("meta_gbm.rank_luts tables need >= 2 nodes")
            rank_luts_list.append(tuple(
                _number(item, f"meta_gbm.rank_luts[{index}][{node}]")
                for node, item in enumerate(lut)
            ))
        rank_luts = tuple(rank_luts_list)
    heavy_reference = value.get("heavy_ref")
    if heavy_reference is not None:
        if not isinstance(heavy_reference, dict):
            raise ProtocolError("meta_gbm.heavy_ref must be an object")
        trees = delta_trees = ordinal_trees = rank_trees = None
    else:
        trees = MetaGbm._normalize_trees(value.get("trees"), width)
        delta_trees = (
            MetaGbm._normalize_trees(value.get("delta_trees"), 2)
            if gain_alpha > 0.0
            else ()
        )
        ordinal_trees = (
            MetaGbm._normalize_trees(
                value.get("ordinal_trees"),
                len(MODEL_IDS) * len(ordinal_thresholds),
                allow_empty=True,
            )
            if ordinal_thresholds
            else ()
        )
        rank_trees = (
            MetaGbm._normalize_trees(value.get("rank_trees"), 2)
            if rank_beta > 0.0
            else ()
        )
    return MetaGbm(
        blend_weights=blends,
        baselines=baselines,
        trees=trees,
        knn_fallback_row=fallback,
        gain_alpha=gain_alpha,
        delta_baselines=delta_baselines,
        delta_trees=delta_trees,
        ordinal_thresholds=ordinal_thresholds,
        ordinal_baselines=ordinal_baselines,
        ordinal_trees=ordinal_trees,
        rank_beta=rank_beta,
        rank_baselines=rank_baselines,
        rank_luts=rank_luts,
        rank_floors=rank_floors,
        rank_trees=rank_trees,
        heavy_reference=heavy_reference,
        heavy_base_path=base_path,
    )


def _parse_augmentation(
    value: Any, base_path: Optional[Path] = None
) -> Optional[RouterAugmentation]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProtocolError("augmentation must be a JSON object")
    blend = _number(value.get("family_blend_weight"), "augmentation.family_blend_weight")
    conf_scale = _number(value.get("knn_conf_scale"), "augmentation.knn_conf_scale")
    if not 0.0 <= blend <= 1.0 or not 0.0 <= conf_scale <= 1.0:
        raise ProtocolError("augmentation weights must lie in [0, 1]")
    means_raw = value.get("family_means")
    if not isinstance(means_raw, dict):
        raise ProtocolError("augmentation.family_means must be an object")
    width = 2 * len(MODEL_IDS)
    family_means = {
        str(name): _vector(row, width, f"family_means.{name}")
        for name, row in means_raw.items()
    }
    knn = value.get("knn")
    if not isinstance(knn, dict):
        raise ProtocolError("augmentation.knn must be an object")
    idf_raw = knn.get("idf")
    if not isinstance(idf_raw, dict):
        raise ProtocolError("augmentation.knn tables are malformed")
    idf = {}
    for key, item in idf_raw.items():
        idf[int(key)] = float(item)
        if not math.isfinite(idf[int(key)]):
            raise ProtocolError("augmentation idf values must be finite")
    heavy_reference = knn.get("heavy_ref")
    if heavy_reference is not None:
        if not isinstance(heavy_reference, dict):
            raise ProtocolError("augmentation.knn.heavy_ref must be an object")
        vectors = targets = None
    else:
        vectors_raw = knn.get("train_vectors")
        targets_raw = knn.get("train_targets")
        if not isinstance(vectors_raw, list) or not isinstance(targets_raw, list):
            raise ProtocolError("augmentation.knn tables are malformed")
        vectors = []
        for document in vectors_raw:
            if not isinstance(document, list):
                raise ProtocolError("augmentation train vectors must be arrays")
            row = {}
            for pair in document:
                index, weight = int(pair[0]), float(pair[1])
                if not math.isfinite(weight):
                    raise ProtocolError("augmentation vector weights must be finite")
                row[index] = weight
            vectors.append(row)
        targets = [_vector(row, width, "knn.train_targets") for row in targets_raw]
        if len(vectors) != len(targets):
            raise ProtocolError("augmentation vectors and targets must align")
    return RouterAugmentation(
        family_blend_weight=blend,
        family_means=family_means,
        conf_scale=conf_scale,
        idf=idf,
        vectors=vectors,
        targets=targets,
        heavy_reference=heavy_reference,
        heavy_base_path=base_path,
    )


def parse_artifact(value: Any, base_path: Optional[Path] = None) -> LearnedArtifact:
    if not isinstance(value, dict) or value.get("artifact_type") != ARTIFACT_TYPE:
        raise ProtocolError("unsupported learned-router artifact")
    if value.get("schema_version") != 1 or value.get("feature_version") != FEATURE_VERSION:
        raise ProtocolError("unsupported learned-router artifact version")
    if value.get("dense_feature_names") != list(DENSE_FEATURE_NAMES):
        raise ProtocolError("artifact dense features do not match runtime")
    if value.get("model_ids") != list(MODEL_IDS):
        raise ProtocolError("artifact model IDs do not match policy")
    word_bins = int(value.get("word_hash_bins", 0))
    char_bins = int(value.get("char_hash_bins", 0))
    dimension = len(DENSE_FEATURE_NAMES) + word_bins + char_bins
    mean = _vector(value.get("dense_mean"), len(DENSE_FEATURE_NAMES), "dense_mean")
    scale = _vector(value.get("dense_scale"), len(DENSE_FEATURE_NAMES), "dense_scale")
    if any(item <= 0 for item in scale):
        raise ProtocolError("dense scales must be positive")
    def head(group: str, model_id: str) -> LinearHead:
        raw = value.get(group, {}).get(model_id)
        if not isinstance(raw, dict):
            raise ProtocolError(f"missing {group}.{model_id}")
        return LinearHead(
            _number(raw.get("intercept"), f"{group}.{model_id}.intercept"),
            _vector(raw.get("coefficients"), dimension, f"{group}.{model_id}.coefficients"),
        )
    safety = value.get("tier_safety_ratios")
    if not isinstance(safety, dict) or set(safety) != set(TIERS):
        raise ProtocolError("artifact tier safety ratios are incomplete")
    blend_weight = _number(value.get("legacy_blend_weight", 0.0), "legacy_blend_weight")
    legacy_artifact = value.get("legacy_artifact", "hash-regex-public.v1.json")
    if not 0.0 <= blend_weight <= 1.0 or not isinstance(legacy_artifact, str):
        raise ProtocolError("invalid legacy ensemble configuration")
    context_limit = value.get(
        "learned_context_limit_chars", LEARNED_CONTEXT_LIMIT_CHARS
    )
    if (
        isinstance(context_limit, bool)
        or not isinstance(context_limit, int)
        or context_limit < 0
    ):
        raise ProtocolError("learned_context_limit_chars must be a non-negative int")
    lookup_raw = value.get("public_lookup")
    public_lookup: Dict[str, Tuple[float, ...]] = {}
    if lookup_raw is not None:
        if (
            not isinstance(lookup_raw, dict)
            or lookup_raw.get("hash_algorithm") != PUBLIC_LOOKUP_HASH
            or not isinstance(lookup_raw.get("entries"), dict)
        ):
            raise ProtocolError("unsupported public lookup table")
        width = 2 * len(MODEL_IDS)
        allowed_lengths = (width, width * len(TIERS))
        for key, row in lookup_raw["entries"].items():
            if not isinstance(key, str) or len(key) != 64:
                raise ProtocolError("public lookup keys must be SHA-256 hex digests")
            if not isinstance(row, list) or len(row) not in allowed_lengths:
                raise ProtocolError(f"public_lookup.entries.{key} has a bad width")
            public_lookup[key] = _vector(
                row, len(row), f"public_lookup.entries.{key}"
            )
    return LearnedArtifact(
        word_hash_bins=word_bins,
        char_hash_bins=char_bins,
        dense_mean=mean,
        dense_scale=scale,
        score_heads={model_id: head("score_heads", model_id) for model_id in MODEL_IDS},
        log_cost_heads={model_id: head("log_cost_heads", model_id) for model_id in MODEL_IDS},
        tier_safety_ratios={tier: _number(safety[tier], f"tier_safety_ratios.{tier}") for tier in TIERS},
        legacy_blend_weight=blend_weight,
        legacy_artifact=legacy_artifact,
        policy_id=str(value.get("policy_id", "")),
        policy_digest=str(value.get("policy_sha256", "")),
        training_summary=value.get("training_summary", {}),
        public_lookup=public_lookup,
        context_limit_chars=context_limit,
        augmentation=_parse_augmentation(value.get("augmentation"), base_path),
        meta_gbm=_parse_meta_gbm(value.get("meta_gbm"), base_path),
    )


def load_artifact(path: Optional[Path] = None) -> LearnedArtifact:
    if path is not None:
        return parse_artifact(load_json(path), base_path=path.parent)
    resource = resources.files("ossp_router.resources").joinpath(DEFAULT_ARTIFACT)
    return parse_artifact(json.loads(resource.read_text(encoding="utf-8")))


def _linear(head: LinearHead, items: Mapping[int, float]) -> float:
    # keys() and values() iterate in the same order as items(), so the fsum
    # consumes the identical product sequence at C speed.
    return head.intercept + math.fsum(
        map(
            operator.mul,
            map(head.coefficients.__getitem__, items.keys()),
            items.values(),
        )
    )


def _predict_learned_items(
    items: Mapping[int, float], artifact: LearnedArtifact
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    scores = {
        model_id: min(1.0, max(0.0, _linear(artifact.score_heads[model_id], items)))
        for model_id in MODEL_IDS
    }
    costs = {
        model_id: math.exp(min(50.0, max(-50.0, _linear(artifact.log_cost_heads[model_id], items))))
        for model_id in MODEL_IDS
    }
    costs[MODEL_IDS[1]] = max(costs[MODEL_IDS[1]], costs[MODEL_IDS[0]] * (1.0 + 1e-12))
    costs[MODEL_IDS[2]] = max(costs[MODEL_IDS[2]], costs[MODEL_IDS[1]] * (1.0 + 1e-12))
    return scores, costs


@lru_cache(maxsize=4)
def _load_legacy_artifact(name: str) -> Any:
    from . import legacy_hash_regex

    resource = resources.files("ossp_router.resources").joinpath(name)
    return legacy_hash_regex.parse_artifact(json.loads(resource.read_text(encoding="utf-8")))


def _predict_legacy_from_context(
    text: str,
    basic: Any,
    tokens: Sequence[str],
    legacy_artifact: Any,
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    """Evaluate the bundled baseline while reusing parsed prompt features."""

    from . import legacy_hash_regex

    dense = (
        math.log1p(basic.character_count),
        math.log1p(basic.word_count),
        math.log1p(basic.sentence_count),
        math.log1p(basic.message_count),
        basic.hangul_ratio,
        math.log1p(basic.code_marker_count),
        math.log1p(basic.math_marker_count),
        basic.numeric_density,
        float(basic.long_context),
        math.log1p(basic.reasoning_marker_count),
        float(bool(legacy_hash_regex._FORMAL_REASONING.search(text))),
        float(bool(legacy_hash_regex._PROGRAM_ANALYSIS.search(text))),
        math.log1p(len(legacy_hash_regex._MULTI_CONSTRAINT.findall(text))),
        float(bool(legacy_hash_regex._SIMPLE_TRANSFORM.search(text))),
    )
    bins = legacy_hash_regex.signed_hash_bins(tokens, legacy_artifact.hash_bins)
    norm = math.sqrt(math.fsum(value * value for value in bins))
    if norm:
        bins = [value / norm for value in bins]
    raw = dense + tuple(bins)
    standardized = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(
            raw, legacy_artifact.feature_mean, legacy_artifact.feature_scale
        )
    )
    scores = {
        model_id: min(
            1.0,
            max(
                0.0,
                legacy_hash_regex._linear(
                    legacy_artifact.score_heads[model_id], standardized
                ),
            ),
        )
        for model_id in MODEL_IDS
    }
    costs = {
        model_id: math.exp(
            min(
                50.0,
                max(
                    -50.0,
                    legacy_hash_regex._linear(
                        legacy_artifact.log_cost_heads[model_id], standardized
                    ),
                ),
            )
        )
        for model_id in MODEL_IDS
    }
    costs[MODEL_IDS[1]] = max(
        costs[MODEL_IDS[1]], costs[MODEL_IDS[0]] * (1.0 + 1e-12)
    )
    costs[MODEL_IDS[2]] = max(
        costs[MODEL_IDS[2]], costs[MODEL_IDS[1]] * (1.0 + 1e-12)
    )
    return scores, costs


def _predict_with_components(
    episode: Episode, artifact: LearnedArtifact, text: str
) -> Tuple[
    Mapping[str, float],
    Mapping[str, float],
    Optional[Tuple[float, ...]],
    Optional[Tuple[float, ...]],
    Tuple[float, ...],
]:
    """Blend-path prediction plus the raw component rows the meta model needs.

    Returns (scores, costs, learned_row6, legacy_row6, raw_dense30) where the
    rows hold [scores..., log costs...]; legacy_row is None without a blend.
    """

    basic = extract_features(episode)
    tokens = _normalized_tokens(text)
    raw_dense = _raw_dense_from_context(episode, text, basic, tokens)
    items = feature_items(
        episode,
        word_hash_bins=artifact.word_hash_bins,
        char_hash_bins=artifact.char_hash_bins,
        dense_mean=artifact.dense_mean,
        dense_scale=artifact.dense_scale,
        raw_dense=raw_dense,
        prepared_text=text,
        prepared_tokens=tokens,
    )
    learned_scores, learned_costs = _predict_learned_items(items, artifact)
    learned_row = tuple(learned_scores[m] for m in MODEL_IDS) + tuple(
        math.log(learned_costs[m]) for m in MODEL_IDS
    )
    weight = artifact.legacy_blend_weight
    if weight <= 0.0:
        return learned_scores, learned_costs, learned_row, None, raw_dense
    from . import legacy_hash_regex

    legacy_tokens = legacy_hash_regex._normalized_tokens(text)
    legacy_scores, legacy_costs = _predict_legacy_from_context(
        text,
        basic,
        legacy_tokens,
        _load_legacy_artifact(artifact.legacy_artifact),
    )
    legacy_row = tuple(legacy_scores[m] for m in MODEL_IDS) + tuple(
        math.log(legacy_costs[m]) for m in MODEL_IDS
    )
    scores = {
        model_id: weight * legacy_scores[model_id] + (1.0 - weight) * learned_scores[model_id]
        for model_id in MODEL_IDS
    }
    costs = {
        model_id: math.exp(
            weight * math.log(legacy_costs[model_id])
            + (1.0 - weight) * math.log(learned_costs[model_id])
        )
        for model_id in MODEL_IDS
    }
    return scores, costs, learned_row, legacy_row, raw_dense


def predict_episode(episode: Episode, artifact: LearnedArtifact) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    text = episode_text(episode)
    weight = artifact.legacy_blend_weight
    if weight > 0.0 and len(text) >= artifact.context_limit_chars:
        from . import legacy_hash_regex

        return legacy_hash_regex.predict_episode(
            episode, _load_legacy_artifact(artifact.legacy_artifact)
        )
    scores, costs, _learned, _legacy, _dense = _predict_with_components(
        episode, artifact, text
    )
    return scores, costs


def select_models(
    predicted_scores: Sequence[Mapping[str, float]],
    predicted_costs: Sequence[Mapping[str, float]],
    *,
    budget_multiplier: float,
    safety_ratio: float,
) -> Tuple[Tuple[str, ...], float]:
    if len(predicted_scores) != len(predicted_costs) or not predicted_scores:
        raise ValueError("score and cost rows must be non-empty and aligned")
    light_total = math.fsum(row[MODEL_IDS[0]] for row in predicted_costs)
    cap = light_total * max(1.0, budget_multiplier * safety_ratio)

    def choose(penalty: float) -> Tuple[Tuple[str, ...], float]:
        selected = tuple(
            max(
                MODEL_IDS,
                key=lambda model_id: (
                    scores[model_id] - penalty * costs[model_id] / light_total,
                    -MODEL_IDS.index(model_id),
                ),
            )
            for scores, costs in zip(predicted_scores, predicted_costs)
        )
        total = math.fsum(costs[model_id] for costs, model_id in zip(predicted_costs, selected))
        return selected, total

    selected, total = choose(0.0)
    if total > cap:
        low, high = 0.0, 1.0
        selected, total = choose(high)
        while total > cap and high < 2**60:
            low, high = high, high * 2.0
            selected, total = choose(high)
        # Forty bisection steps are far below the granularity at which a
        # finite batch can change decisions and keep CPU runtime predictable.
        for _ in range(40):
            middle = (low + high) / 2.0
            candidate, candidate_total = choose(middle)
            if candidate_total <= cap:
                high, selected, total = middle, candidate, candidate_total
            else:
                low = middle
    if total > cap:
        selected = tuple(MODEL_IDS[0] for _ in predicted_scores)
        total = light_total
    return selected, total / light_total


def _clamp_row(row: Sequence[float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    count = len(MODEL_IDS)
    scores = {
        model_id: min(1.0, max(0.0, row[index]))
        for index, model_id in enumerate(MODEL_IDS)
    }
    costs = {
        model_id: math.exp(min(50.0, max(-50.0, row[count + index])))
        for index, model_id in enumerate(MODEL_IDS)
    }
    costs[MODEL_IDS[1]] = max(costs[MODEL_IDS[1]], costs[MODEL_IDS[0]] * (1.0 + 1e-12))
    costs[MODEL_IDS[2]] = max(costs[MODEL_IDS[2]], costs[MODEL_IDS[1]] * (1.0 + 1e-12))
    return scores, costs


def predict_episode_augmented(
    episode: Episode, artifact: LearnedArtifact, tier: Optional[str] = None
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    """Blend linear, family-mean, kNN, and stacked-tree predictions.

    All blending happens in (score, log-cost) space, mirroring the offline
    validation pipeline; the augmented row is clamped before and after the
    meta blend exactly as during validation.  ``tier`` selects the per-tier
    meta blend weight and is required when a meta model is present.
    """

    augmentation = artifact.augmentation
    if augmentation is None:
        return predict_episode(episode, artifact)
    if artifact.meta_gbm is not None and tier not in TIERS:
        raise ProtocolError("meta blending requires a valid tier")
    text = episode_text(episode)
    meta = artifact.meta_gbm
    learned_row = legacy_row = raw_dense = None
    if artifact.legacy_blend_weight > 0.0 and len(text) >= artifact.context_limit_chars:
        # Pathologically long prompts skip the meta features on purpose.
        scores, costs = predict_episode(episode, artifact)
        meta = None
    else:
        scores, costs, learned_row, legacy_row, raw_dense = _predict_with_components(
            episode, artifact, text
        )
    row = [scores[model_id] for model_id in MODEL_IDS] + [
        math.log(costs[model_id]) for model_id in MODEL_IDS
    ]
    family = similarity.classify_family(text)
    family_row = augmentation.family_means.get(family)
    if family_row is not None and augmentation.family_blend_weight > 0.0:
        blend = augmentation.family_blend_weight
        row = [(1.0 - blend) * a + blend * b for a, b in zip(row, family_row)]
    knn_row: Tuple[float, ...] = ()
    top_similarity = 0.0
    if augmentation.conf_scale > 0.0 or meta is not None:
        query = similarity.tfidf_vector(text, augmentation.idf)
        knn_row, top_similarity = augmentation.index.predict(query)
    if knn_row and augmentation.conf_scale > 0.0:
        weight = min(1.0, max(0.0, top_similarity)) * augmentation.conf_scale
        row = [(1.0 - weight) * a + weight * b for a, b in zip(row, knn_row)]
    scores, costs = _clamp_row(row)
    if meta is None or legacy_row is None:
        return scores, costs
    prod_row = [scores[model_id] for model_id in MODEL_IDS] + [
        math.log(costs[model_id]) for model_id in MODEL_IDS
    ]
    features = list(raw_dense)
    onehot = [0.0] * len(similarity.FAMILY_NAMES)
    onehot[similarity.FAMILY_NAMES.index(family)] = 1.0
    features.extend(onehot)
    features.extend(legacy_row)
    features.extend(learned_row)
    features.extend(knn_row if knn_row else meta.knn_fallback_row)
    features.append(top_similarity if knn_row else 0.0)
    meta_row = [
        similarity.evaluate_trees(meta.baselines[k], meta.trees[k], features)
        for k in range(2 * len(MODEL_IDS))
    ]
    if meta.ordinal_thresholds:
        # E21: scores from cumulative-threshold classifiers,
        # E[s] = step * sum_t sigmoid(raw_t)
        threshold_count = len(meta.ordinal_thresholds)
        for model_index in range(len(MODEL_IDS)):
            cumulative = 0.0
            for threshold_index in range(threshold_count):
                head = model_index * threshold_count + threshold_index
                raw = similarity.evaluate_trees(
                    meta.ordinal_baselines[head], meta.ordinal_trees[head], features
                )
                raw = max(-50.0, min(50.0, raw))
                cumulative += 1.0 / (1.0 + math.exp(-raw))
            meta_row[model_index] = cumulative / threshold_count
    if meta.gain_alpha > 0.0:
        # gain heads regress the decision-relevant upgrade deltas directly;
        # reconstructed scores are mixed into the direct heads
        delta_one = similarity.evaluate_trees(
            meta.delta_baselines[0], meta.delta_trees[0], features
        )
        delta_two = similarity.evaluate_trees(
            meta.delta_baselines[1], meta.delta_trees[1], features
        )
        if meta.rank_beta > 0.0:
            # E27: rank-transformed efficiency heads.  Predicted train
            # percentile -> train efficiency quantile (piecewise-linear LUT)
            # -> times the predicted cost delta, mixed into the regressed
            # deltas at rank_beta.
            costs_pred = [
                math.exp(max(-50.0, min(50.0, meta_row[len(MODEL_IDS) + m])))
                for m in range(len(MODEL_IDS))
            ]
            deltas = [delta_one, delta_two]
            for g in range(2):
                raw = similarity.evaluate_trees(
                    meta.rank_baselines[g], meta.rank_trees[g], features
                )
                rank = max(0.0, min(1.0, raw))
                lut = meta.rank_luts[g]
                position = rank * (len(lut) - 1)
                lower = min(int(position), len(lut) - 2)
                fraction = position - lower
                efficiency = lut[lower] * (1.0 - fraction) + lut[lower + 1] * fraction
                cost_delta = max(costs_pred[g + 1] - costs_pred[g], meta.rank_floors[g])
                deltas[g] = (
                    (1.0 - meta.rank_beta) * deltas[g]
                    + meta.rank_beta * efficiency * cost_delta
                )
            delta_one, delta_two = deltas
        reconstructed = (
            meta_row[0],
            meta_row[0] + delta_one,
            meta_row[0] + delta_one + delta_two,
        )
        alpha = meta.gain_alpha
        for position in range(len(MODEL_IDS)):
            meta_row[position] = (
                (1.0 - alpha) * meta_row[position] + alpha * reconstructed[position]
            )
    blend = meta.blend_weights[tier]
    final_row = [(1.0 - blend) * a + blend * b for a, b in zip(prod_row, meta_row)]
    return _clamp_row(final_row)


def _predict_with_lookup(
    episode: Episode, artifact: LearnedArtifact, tier: Optional[str] = None
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
    """Reuse stored public-prompt predictions before running feature extraction.

    CHALLENGE_RULES.md allows exact prompt/hash lookups built from public
    data.  The key depends only on prompt content (plus the requested tier
    when rows are stored per tier), and each stored row is the exact output
    the compute path would produce, so decisions are unchanged.
    """

    lookup = artifact.public_lookup
    if lookup:
        key = hashlib.sha256(episode_text(episode).encode("utf-8")).hexdigest()
        row = lookup.get(key)
        if row is not None:
            width = 2 * len(MODEL_IDS)
            if len(row) == width * len(TIERS):
                if tier not in TIERS:
                    raise ProtocolError("per-tier lookup rows require a valid tier")
                offset = TIERS.index(tier) * width
                row = row[offset:offset + width]
            model_count = len(MODEL_IDS)
            return (
                dict(zip(MODEL_IDS, row[:model_count])),
                dict(zip(MODEL_IDS, row[model_count:])),
            )
    return predict_episode_augmented(episode, artifact, tier)


def make_submission(inputs: Any, policy: RoutingPolicy, artifact: LearnedArtifact, tier: str) -> Submission:
    if tier not in TIERS:
        raise ProtocolError(f"unknown tier: {tier}")
    if artifact.policy_id != policy.policy_id or artifact.policy_digest != policy_sha256(policy):
        raise ProtocolError("artifact policy binding does not match the active policy")
    predictions = [
        _predict_with_lookup(episode, artifact, tier) for episode in inputs.episodes
    ]
    selected, _ = select_models(
        [row[0] for row in predictions],
        [row[1] for row in predictions],
        budget_multiplier=float(policy.tiers[tier].budget_multiplier),
        safety_ratio=artifact.tier_safety_ratios[tier],
    )
    return parse_submission(submission_to_dict(Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(Decision(episode.episode_id, model_id) for episode, model_id in zip(inputs.episodes, selected)),
    )))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="router-run", description="Run the learned prompt-only router")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--policy", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = load_policy(args.policy) if args.policy else load_bundled_policy()
        artifact = load_artifact(args.artifact)
        write_submission_atomic(args.output, make_submission(inputs, policy, artifact, args.tier))
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"OK: wrote {args.tier} learned-router submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
