"""Choose which passages the model sees.

Small knowledge bases are sent whole. Larger ones are ranked on this machine
with BM25, so the default install does not need an embedding model or a vector
database. A 5-character prefix feature lets "refund" match "refunds".
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from loopkit.knowledge import Passage

_STOP = frozenset(
    """
    a an the of to and in on for is it with as at by from or be this that are was
    were your you we our their its if into over not can how what when where which
    who whom why does did do than then them these those
    """.split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Retrieval:
    passages: tuple[Passage, ...]
    weak: bool


def select(
    passages: Sequence[Passage],
    query: str,
    *,
    top_k: int,
    budget: int,
) -> Retrieval:
    """Return passages in reading order, capped by count and character budget."""

    if not passages:
        return Retrieval(passages=(), weak=True)
    total_chars = sum(len(passage.text) for passage in passages)
    if total_chars <= budget:
        return Retrieval(passages=tuple(passages), weak=False)

    ranked = _bm25(passages, query)
    weak = ranked[0][1] <= 0
    chosen: list[Passage] = []
    used = 0
    for passage, _score in ranked:
        if len(chosen) >= top_k:
            break
        if chosen and used + len(passage.text) > budget:
            continue
        chosen.append(passage)
        used += len(passage.text)
    if not chosen:
        chosen = [ranked[0][0]]
    order = {passage.id: index for index, passage in enumerate(passages)}
    chosen.sort(key=lambda passage: order[passage.id])
    return Retrieval(passages=tuple(chosen), weak=weak)


def _bm25(passages: Sequence[Passage], query: str) -> list[tuple[Passage, float]]:
    documents = [_features(_indexed(passage)) for passage in passages]
    query_features = _features(query)
    if not query_features or not documents:
        return [(passage, 0.0) for passage in passages]

    count = len(documents)
    document_frequency: Counter[str] = Counter()
    for features in documents:
        document_frequency.update(set(features))
    average_length = sum(len(features) for features in documents) / count
    k1 = 1.5
    b = 0.75
    scored: list[tuple[Passage, float]] = []
    for passage, features in zip(passages, documents, strict=True):
        frequency = Counter(features)
        length = len(features) or 1
        score = 0.0
        for term in query_features:
            seen = frequency.get(term, 0)
            if not seen:
                continue
            idf = math.log(1 + (count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = seen + k1 * (1 - b + b * length / average_length)
            score += idf * (seen * (k1 + 1)) / denominator
        scored.append((passage, score))
    scored.sort(key=lambda item: (-item[1], _id_sort_key(item[0])))
    return scored


def _indexed(passage: Passage) -> str:
    label = " ".join(part for part in (passage.heading, passage.source) if part)
    # Repeat the label so a heading match can outrank a passing mention in a long body.
    return f"{label}\n{label}\n{passage.text}"


def _features(text: str) -> list[str]:
    features: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        if len(token) < 3 or token in _STOP:
            continue
        features.append(token)
        if len(token) >= 5:
            features.append(token[:5])
    return features


def _id_sort_key(passage: Passage) -> tuple[int, int | str]:
    if passage.id.startswith("p") and passage.id[1:].isdigit():
        return (0, int(passage.id[1:]))
    return (1, passage.id)
