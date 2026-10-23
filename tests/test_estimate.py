"""The reply-size rules: deterministic, from the input first and the latest reply after."""

from __future__ import annotations

import pytest

from loopkit.estimate import (
    CHARS_PER_TOKEN,
    DEFAULT_MAX_RESPONSE_CHARS,
    EMPTY_SHAPE,
    MIN_RESPONSE_CHARS,
    AskShape,
    EstimationStrategy,
    coerce_strategy,
    expected_output,
    first_draft,
    first_output,
    length_compliant,
    max_response_chars,
    response_length_target,
    words_for,
)


def test_the_first_draft_grows_with_the_passages_within_bounds() -> None:
    small = AskShape(task_tokens=20, instruction_count=4, passage_tokens=200)
    large = AskShape(task_tokens=20, instruction_count=4, passage_tokens=4_000)
    huge = AskShape(task_tokens=20, instruction_count=4, passage_tokens=100_000)
    assert first_draft(EMPTY_SHAPE) == 120  # the floor
    assert 120 <= first_draft(small) < first_draft(large) <= first_draft(huge) == 1_500
    # The same input always gives the same number.
    assert first_draft(large) == first_draft(AskShape(20, 4, 4_000))


def test_book_and_open_mode_have_their_own_rules() -> None:
    shape = AskShape(task_tokens=20, instruction_count=4, passage_tokens=2_000)
    book = AskShape(task_tokens=20, instruction_count=4, passage_tokens=2_000, response="book")
    assert first_draft(book) > first_draft(shape)
    assert first_draft(book) >= 1_000
    open_normal = AskShape(task_tokens=20, instruction_count=3, passage_tokens=0, mode="open")
    open_book = AskShape(
        task_tokens=20, instruction_count=3, passage_tokens=0, mode="open", response="book"
    )
    assert 120 <= first_draft(open_normal) <= 1_500
    assert first_draft(open_book) > first_draft(open_normal)
    # In open mode the task is the only size signal.
    longer = AskShape(task_tokens=200, instruction_count=3, passage_tokens=0, mode="open")
    assert first_draft(longer) > first_draft(open_normal)


def test_first_output_by_kind() -> None:
    shape = AskShape(task_tokens=20, instruction_count=5, passage_tokens=1_000)
    assert first_output("Draft", shape) == first_draft(shape)
    assert first_output("OneCheck", shape) < first_output("GroundingCheck", shape)
    # The combined check carries one verdict per instruction.
    fewer = AskShape(task_tokens=20, instruction_count=1, passage_tokens=1_000)
    assert first_output("DraftCheck", shape) > first_output("DraftCheck", fewer)
    assert first_output("RolePrefix", shape) > 0
    assert first_output("Something", shape) > 0


def test_last_turn_strategy_uses_the_latest_reply_once_there_is_one() -> None:
    shape = AskShape(task_tokens=20, instruction_count=4, passage_tokens=1_000)
    strategy = EstimationStrategy.LAST_TURN
    assert expected_output(strategy, "Draft", shape, {}) == first_draft(shape)
    assert expected_output(strategy, "Draft", shape, {"Draft": 777}) == 777
    assert expected_output(strategy, "Draft", shape, {"OneCheck": 777}) == first_draft(shape)
    assert expected_output(strategy, "OneCheck", shape, {"OneCheck": 9}) == 9


def test_strategy_values_and_coercion() -> None:
    assert [strategy.value for strategy in EstimationStrategy] == ["last_turn"]
    assert coerce_strategy(None) is EstimationStrategy.LAST_TURN
    assert coerce_strategy(EstimationStrategy.LAST_TURN) is EstimationStrategy.LAST_TURN
    assert coerce_strategy("last_turn") is EstimationStrategy.LAST_TURN
    assert coerce_strategy(" LAST_TURN ") is EstimationStrategy.LAST_TURN
    with pytest.raises(ValueError, match="estimation_strategy must be one of: last_turn"):
        coerce_strategy("guess")
    with pytest.raises(ValueError, match="estimation_strategy must be one of"):
        coerce_strategy(3)
    with pytest.raises(ValueError, match="Unknown estimation strategy"):
        expected_output("other", "Draft", EMPTY_SHAPE, {})  # type: ignore[arg-type]


def test_model_max_response_chars_by_name_prefix_and_client_cap() -> None:
    assert max_response_chars("grok-4.7") == 128_000 * CHARS_PER_TOKEN
    assert max_response_chars("gpt-6.1-sol") == 128_000 * CHARS_PER_TOKEN
    assert max_response_chars("Claude-Sonnet-5-5") == 64_000 * CHARS_PER_TOKEN
    assert max_response_chars("llama3.1") == DEFAULT_MAX_RESPONSE_CHARS
    assert max_response_chars("openchat") == DEFAULT_MAX_RESPONSE_CHARS
    # The client's own output cap wins when it is lower, less the JSON around the answer.
    assert max_response_chars("claude-sonnet-5-5", 8192) == (8192 - 30) * CHARS_PER_TOKEN
    assert max_response_chars("claude-sonnet-5-5", 1_000_000) == 64_000 * CHARS_PER_TOKEN
    # Never below the floor, whatever the cap says.
    assert max_response_chars("tiny", 10) == MIN_RESPONSE_CHARS


def test_response_length_target_from_the_estimate_or_the_caller() -> None:
    shape = AskShape(task_tokens=20, instruction_count=4, passage_tokens=600)
    auto = response_length_target(shape, 512_000)
    assert auto == (first_draft(shape) - 30) * CHARS_PER_TOKEN
    assert MIN_RESPONSE_CHARS <= auto <= 512_000
    # An explicit length is used as given, inside the bounds.
    wanted = AskShape(task_tokens=20, instruction_count=4, passage_tokens=600, response_chars=900)
    assert response_length_target(wanted, 512_000) == 900
    tiny = AskShape(task_tokens=20, instruction_count=4, passage_tokens=600, response_chars=5)
    assert response_length_target(tiny, 512_000) == MIN_RESPONSE_CHARS
    huge = AskShape(task_tokens=20, instruction_count=4, passage_tokens=600, response_chars=10**9)
    assert response_length_target(huge, 256_000) == 256_000
    # And it feeds the reply estimate, so the budget and the length agree.
    assert first_draft(wanted) == 225 + 30
    assert first_output("Draft", wanted) == first_draft(wanted)


def test_length_compliance_has_a_tolerance() -> None:
    assert length_compliant(800, 800)
    assert length_compliant(880, 800)
    assert not length_compliant(881, 800)
    assert length_compliant(0, 800)
    assert words_for(840) == 140
    assert words_for(1) == 1
