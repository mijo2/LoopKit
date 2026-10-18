from loopkit.knowledge import Passage
from loopkit.rag import query
from loopkit.retrieve import select


def _passage(pid: str, text: str, heading: str = "") -> Passage:
    return Passage(id=pid, source=f"{pid}.md", heading=heading, text=text)


def test_small_corpus_is_sent_whole() -> None:
    passages = (
        _passage("p1", "API keys are shown once."),
        _passage("p2", "Refunds last 14 days."),
    )
    found = select(passages, "something unrelated", top_k=1, budget=10_000)
    assert [item.id for item in found.passages] == ["p1", "p2"]
    assert found.is_weak is False


def test_plural_query_ranks_the_matching_passage() -> None:
    filler = "The status page lists uptime for each region. " * 40
    passages = (
        _passage("p1", filler, heading="Status"),
        _passage("p2", "Refunds are available for 14 days after the charge.", heading="Refunds"),
        _passage("p3", filler + " Keys are shown once.", heading="Keys"),
    )
    found = select(passages, "What is the refund window?", top_k=1, budget=200)
    assert [item.id for item in found.passages] == ["p2"]
    assert found.is_weak is False


def test_no_overlap_sets_is_weak() -> None:
    passages = (
        _passage("p1", "harbor lanes and shipping notes " * 30),
        _passage("p2", "copper wire and voltage notes " * 30),
        _passage("p3", "cedar shingles and roof pitch " * 30),
    )
    found = select(passages, "zzzzquantum", top_k=2, budget=2000)
    assert found.is_weak is True
    assert [item.id for item in found.passages] == ["p1", "p2"]


def test_rag_query_ranks_on_this_machine() -> None:
    hits = query(
        (
            _passage("p1", "harbor lanes and shipping notes"),
            _passage("p2", "Refunds are available for 14 days after the charge.", heading="Refunds"),
        ),
        "What is the refund window?",
    )
    assert [hit.passage.id for hit in hits] == ["p2", "p1"]
    assert hits[0].score > 0
    assert hits[1].score == 0


def test_higher_score_does_not_reorder_the_prompt() -> None:
    passages = (
        _passage("p1", "alpha " * 200),
        _passage("p2", "refunds " * 200),
        _passage("p3", "beta " * 200),
    )
    found = select(passages, "refund", top_k=2, budget=3000)
    assert [item.id for item in found.passages] == ["p1", "p2"]
