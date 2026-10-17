"""Local RAG query.

Drafting and checking call a model. A RAG query does not. It ranks passages
that are already in memory, using BM25, and returns a score for each one.
Nothing in this module sends a request or decides when the loop should stop.
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
class RagHit:
    """One passage scored by a local RAG query.

    ``score`` is a BM25 value. Zero means the query text shares no indexed
    features with the passage.
    """

    passage: Passage
    score: float


def query(passages: Sequence[Passage], text: str) -> list[RagHit]:
    """Rank ``passages`` for a RAG query and return the best match first.

    ``text`` is the question or task. It is not a prompt, and this function
    does not call a model. Tokens shorter than 3 characters and a small stop
    list are dropped. A 5-character prefix is indexed as well, so ``refund``
    can match ``refunds``. The heading and the source name are repeated in
    the indexed text so a heading match can outrank a passing mention in a
    long body. A hit with score 0 shares no features with ``text``.
    """

    documents = [_features(_indexed(passage)) for passage in passages]
    query_features = _features(text)
    if not query_features or not documents:
        return [RagHit(passage, 0.0) for passage in passages]

    count = len(documents)
    document_frequency: Counter[str] = Counter()
    for features in documents:
        document_frequency.update(set(features))
    average_length = sum(len(features) for features in documents) / count
    k1 = 1.5
    b = 0.75
    hits: list[RagHit] = []
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
        hits.append(RagHit(passage, score))
    hits.sort(key=lambda hit: (-hit.score, _id_sort_key(hit.passage)))
    return hits


def _indexed(passage: Passage) -> str:
    """Text BM25 indexes for one passage, with the heading repeated.

    Repeating the heading and the source gives those words more weight than
    the same word appearing once in a long body.
    """

    label = " ".join(part for part in (passage.heading, passage.source) if part)
    return f"{label}\n{label}\n{passage.text}"


def _features(text: str) -> list[str]:
    """Lowercased tokens used as BM25 terms, plus a prefix of longer tokens.

    Stop words and tokens under 3 characters are left out. A token of 5
    characters or more also contributes its first 5 characters, so a singular
    query can meet a plural in the passage.
    """

    features: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        if len(token) < 3 or token in _STOP:
            continue
        features.append(token)
        if len(token) >= 5:
            features.append(token[:5])
    return features


def _id_sort_key(passage: Passage) -> tuple[int, int | str]:
    """Order equal scores by passage id, with ``p1``, ``p2``, ... as numbers.

    Non-numeric ids sort after numeric ones, then by the id string, so a tie
    does not depend on dict order.
    """

    if passage.id.startswith("p") and passage.id[1:].isdigit():
        return (0, int(passage.id[1:]))
    return (1, passage.id)
