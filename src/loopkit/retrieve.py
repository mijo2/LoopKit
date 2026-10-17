"""Choose which passages the model sees.

Small knowledge bases are sent whole. Larger ones are ranked by a local RAG
query (:func:`loopkit.rag.query`), so the default install does not need an
embedding model or a vector database. This module does not call a model and
does not score passages itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from loopkit.knowledge import Passage
from loopkit.rag import query as rag_query


@dataclass(frozen=True)
class Retrieval:
    """Passages selected for one question.

    ``is_weak`` is true when a RAG query ran and its best score was zero:
    the question shares no indexed words with the passages. It is false when
    the whole knowledge base fit in the budget, and when the best passage
    matched.
    """

    passages: tuple[Passage, ...]
    is_weak: bool


def select(
    passages: Sequence[Passage],
    query: str,
    *,
    top_k: int,
    budget: int,
) -> Retrieval:
    """Pick passages for one question, in reading order.

    When the passages fit in ``budget``, every passage is returned and
    ``is_weak`` is false: nothing was dropped, so there is no weak match.
    Otherwise :func:`loopkit.rag.query` ranks them. This keeps the best
    matches up to ``top_k``, skipping a later passage when it would pass
    ``budget``, and always keeps at least the best hit. The chosen passages
    are then sorted back into the order they had in ``passages``.
    """

    if not passages:
        return Retrieval(passages=(), is_weak=True)
    total_chars = sum(len(passage.text) for passage in passages)
    if total_chars <= budget:
        return Retrieval(passages=tuple(passages), is_weak=False)

    ranked = rag_query(passages, query)
    is_weak = ranked[0].score <= 0
    chosen: list[Passage] = []
    used = 0
    for hit in ranked:
        if len(chosen) >= top_k:
            break
        if chosen and used + len(hit.passage.text) > budget:
            continue
        chosen.append(hit.passage)
        used += len(hit.passage.text)
    if not chosen:
        chosen = [ranked[0].passage]
    order = {passage.id: index for index, passage in enumerate(passages)}
    chosen.sort(key=lambda passage: order[passage.id])
    return Retrieval(passages=tuple(chosen), is_weak=is_weak)
