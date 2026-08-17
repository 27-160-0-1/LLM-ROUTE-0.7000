# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python hashed character tf-idf and public-train kNN prediction.

CHALLENGE_RULES.md allows classifiers, vocabularies/IDF tables, and search
indexes built from public data inside the submission image.  This module
matches prompts against the public Train prompts by hashed character-n-gram
cosine similarity; the training tool and the runtime share this exact
implementation, so offline validation reflects container behavior.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

HASH_BITS = 15
HASH_BINS = 1 << HASH_BITS
NGRAM_SIZES = (3, 4, 5)
TEXT_LIMIT = 4_000
TOP_COMPONENTS = 256
NEIGHBORS = 16  # E20: k=16이 k=8 대비 CV EV +0.0014 (단봉 곡선 확인)

_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_MASK64 = (1 << 64) - 1
_SPACE = re.compile(r"\s+")

_GRAM_BINS: Dict[str, int] = {}
_GRAM_CACHE_LIMIT = 2_000_000

# Content-only source-family heuristics (validated on public Train/Dev).
_RULETAKER = re.compile(r"\bIf (?:something|someone|the \w+) ")
_TRUTHFULQA = re.compile(r"^Question: .{5,200}\nA\. ", re.DOTALL)
_HRMCR_AGE = re.compile(r"한국 나이|선/후배|존댓말")
_HRMCR_CAL = re.compile(r"음력|양력|윤년|3·1절|생일")
_DMMATH = re.compile(
    r"^(?:Solve|Calculate|Simplify|Differentiate|Factor|Expand|Evaluate|"
    r"What is|Let |Suppose|Divide|Multiply|Put|Sort|Round|Convert|"
    r"Two letters|Three letters|Four letters|Five letters|Six letters|"
    r"What comes next|List the prime factors|Is \d|Find the \w+ derivative)"
)
_CODE = re.compile(r"\bdef \w+\(|assert f\(")
_AIME = re.compile(r"\$[^$]+\$")

FAMILY_NAMES = (
    "code", "hrmcr", "ruletaker", "truthfulqa", "belebele",
    "longdoc", "aime", "dmmath", "gsm8k_or_other",
)


def classify_family(text: str) -> str:
    """Assign one of the public source-family labels from prompt content only."""

    head = text[:600]
    if _CODE.search(head):
        return "code"
    if _HRMCR_AGE.search(head) or _HRMCR_CAL.search(head[:200]):
        return "hrmcr"
    if _RULETAKER.search(head) and " is " in head:
        return "ruletaker"
    if _TRUTHFULQA.match(text):
        return "truthfulqa"
    if sum("가" <= character <= "힣" for character in head) > 40:
        return "belebele"
    if len(text) > 6_000:
        return "longdoc"
    if _AIME.search(head) and len(text) < 2_000:
        return "aime"
    if _DMMATH.match(head) and len(text) < 400:
        return "dmmath"
    return "gsm8k_or_other"


def _gram_bin(gram: str) -> int:
    cached = _GRAM_BINS.get(gram)
    if cached is not None:
        return cached
    digest = _FNV_OFFSET
    for byte in gram.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _MASK64
    result = digest & (HASH_BINS - 1)
    if len(_GRAM_BINS) < _GRAM_CACHE_LIMIT:
        _GRAM_BINS[gram] = result
    return result


def hashed_counts(text: str) -> Dict[int, int]:
    """Hashed character n-gram counts of the normalized prompt head."""

    normalized = _SPACE.sub(" ", text[:TEXT_LIMIT].casefold()).strip()
    padded = f" {normalized} "
    counts: Dict[int, int] = {}
    length = len(padded)
    for size in NGRAM_SIZES:
        for start in range(0, max(0, length - size + 1)):
            index = _gram_bin(padded[start:start + size])
            counts[index] = counts.get(index, 0) + 1
    return counts


def document_frequencies(texts: Iterable[str]) -> Tuple[Dict[int, int], int]:
    frequencies: Dict[int, int] = {}
    total = 0
    for text in texts:
        total += 1
        for index in hashed_counts(text):
            frequencies[index] = frequencies.get(index, 0) + 1
    return frequencies, total


def idf_table(frequencies: Mapping[int, int], total_documents: int) -> Dict[int, float]:
    return {
        index: math.log((1 + total_documents) / (1 + df)) + 1.0
        for index, df in frequencies.items()
    }


def tfidf_vector(
    text: str,
    idf: Mapping[int, float],
    *,
    top_components: int = 0,
) -> Dict[int, float]:
    """L2-normalized sublinear tf-idf vector; optionally keep top components."""

    weights = {}
    for index, count in hashed_counts(text).items():
        factor = idf.get(index)
        if factor is None:
            continue
        weights[index] = (1.0 + math.log(count)) * factor
    if top_components and len(weights) > top_components:
        kept = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
        weights = dict(kept[:top_components])
    norm = math.sqrt(math.fsum(value * value for value in weights.values()))
    if not norm:
        return {}
    return {index: value / norm for index, value in weights.items()}


def evaluate_trees(baseline: float, trees: Sequence[Sequence[Sequence[float]]], features: Sequence[float]) -> float:
    """Evaluate exported HistGradientBoosting trees.

    Each node is [is_leaf, value, feature_idx, threshold, left, right]; the
    traversal reproduces sklearn's numeric-threshold predict exactly for
    finite inputs.
    """

    total = baseline
    for tree in trees:
        node = tree[0]
        while not node[0]:
            node = tree[int(node[4])] if features[int(node[2])] <= node[3] else tree[int(node[5])]
        total += node[1]
    return total


class KnnIndex:
    """Inverted index over stored train vectors for cosine kNN."""

    def __init__(
        self,
        vectors: Sequence[Mapping[int, float]],
        targets: Sequence[Sequence[float]],
    ) -> None:
        if len(vectors) != len(targets):
            raise ValueError("kNN vectors and targets must align")
        self.targets = [tuple(float(v) for v in row) for row in targets]
        postings: Dict[int, List[Tuple[int, float]]] = {}
        for document, vector in enumerate(vectors):
            for index, value in vector.items():
                postings.setdefault(index, []).append((document, value))
        self.postings = postings

    def predict(
        self, query: Mapping[int, float], neighbors: int = NEIGHBORS
    ) -> Tuple[Tuple[float, ...], float]:
        """Return (weighted neighbor target row, top-1 similarity)."""

        if not query:
            return (), 0.0
        scores: Dict[int, float] = {}
        for index, value in query.items():
            for document, stored in self.postings.get(index, ()):
                scores[document] = scores.get(document, 0.0) + value * stored
        if not scores:
            return (), 0.0
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:neighbors]
        total = math.fsum(similarity for _document, similarity in ranked)
        if total <= 1e-9:
            return (), 0.0
        width = len(self.targets[0])
        blended = [0.0] * width
        for document, similarity in ranked:
            weight = similarity / total
            row = self.targets[document]
            for position in range(width):
                blended[position] += weight * row[position]
        return tuple(blended), ranked[0][1]
