"""The token estimator, provider usage parsing, and the meter that enforces a budget."""

from __future__ import annotations

import pytest

from loopkit.estimate import AskShape, EstimationStrategy, first_output
from loopkit.tokens import TokenMeter, Usage, estimate_tokens, usage_from_response

PROSE = (
    "Northwind refunds a subscription cancelled within 14 days of the charge. "
    "The refund returns to the original payment method within 5 business days."
)


def test_estimate_is_zero_for_empty_and_grows_with_text() -> None:
    assert estimate_tokens("") == 0
    assert 1 <= estimate_tokens("hi") <= 2
    short = estimate_tokens(PROSE)
    # About 4 characters per token for English prose: 143 characters is in the 30s.
    assert 28 <= short <= 45
    assert estimate_tokens(PROSE + " " + PROSE) > short


def test_estimate_counts_dense_text_by_words_and_non_ascii_by_character() -> None:
    # Long identifiers: word-based count is the floor, so this is at least 1.3 per word.
    ids = " ".join(f"[p{n}]" for n in range(1, 21))
    assert estimate_tokens(ids) >= 26
    # Each CJK character is at least one token.
    assert estimate_tokens("東京都渋谷区") >= 6


def test_usage_from_each_provider_shape() -> None:
    responses = usage_from_response({"usage": {"input_tokens": 120, "output_tokens": 30}})
    assert responses == Usage(120, 30)
    chat = usage_from_response(
        {"usage": {"prompt_tokens": 50, "completion_tokens": 7, "total_tokens": 57}}
    )
    assert chat == Usage(50, 7)
    anthropic = usage_from_response(
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 40,
            }
        }
    )
    assert anthropic == Usage(150, 5)
    # A bare usage dict works too, as does a dict a custom client might store.
    assert usage_from_response({"input_tokens": 3, "output_tokens": 4}) == Usage(3, 4)


def test_usage_trusts_total_when_reasoning_tokens_are_not_in_completion() -> None:
    usage = usage_from_response(
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 620}}
    )
    assert usage == Usage(100, 520)


def test_usage_from_an_sdk_object() -> None:
    class Sampling:
        prompt_tokens = 70
        completion_tokens = 9
        reasoning_tokens = 0
        total_tokens = 79

    class Response:
        usage = Sampling()

    assert usage_from_response(Response()) == Usage(70, 9)
    assert usage_from_response(Sampling()) == Usage(70, 9)


def test_usage_is_none_when_missing_or_zero() -> None:
    assert usage_from_response(None) is None
    assert usage_from_response({"choices": []}) is None
    assert usage_from_response({"usage": None}) is None
    assert usage_from_response({"usage": {"input_tokens": 0, "output_tokens": 0}}) is None
    assert usage_from_response({"usage": {"input_tokens": "many"}}) is None
    assert usage_from_response("text") is None

    class Empty:
        prompt_tokens = 0
        completion_tokens = 0

    assert usage_from_response(Empty()) is None


def test_meter_without_a_limit_only_counts() -> None:
    meter = TokenMeter()
    assert meter.limit is None
    assert meter.remaining is None
    assert meter.fits(10**9)
    usage = meter.record(
        system_text="rules", user_text="task", output_text='{"a": 1}', kind="Draft", reported=None
    )
    assert usage.estimated is True
    assert usage.input_tokens == estimate_tokens("rules") + estimate_tokens("task")
    assert usage.output_tokens == estimate_tokens('{"a": 1}')
    assert meter.used == usage.total
    assert meter.calls == 1
    assert meter.estimated is True
    assert meter.exceeded is False
    assert meter.describe() == f"about {meter.used:,} tokens used"


def test_meter_sizes_a_reply_from_the_shape_until_one_is_observed() -> None:
    shape = AskShape(task_tokens=40, instruction_count=4, passage_tokens=2_000)
    meter = TokenMeter(10_000, estimator=lambda text: len(text), shape=shape)
    assert meter.strategy is EstimationStrategy.LAST_TURN
    assert meter.expected_output("Draft") == first_output("Draft", shape)
    assert meter.estimate_call("ab", "cde", "Draft") == 5 + first_output("Draft", shape)
    assert meter.estimate_call("ab", "cde", "OneCheck") == 5 + first_output("OneCheck", shape)
    assert meter.estimate_call("ab", "cde", "Unknown") == 5 + first_output("Unknown", shape)
    # Text that is not in the prompt yet, such as the draft a check will read.
    assert meter.estimate_call(
        "ab", "cde", "OneCheck", extra_prompt_tokens=70
    ) == 75 + first_output("OneCheck", shape)
    meter.record(
        system_text="ab", user_text="cde", output_text="", kind="Draft", reported=Usage(5, 42)
    )
    assert meter.expected_output("Draft") == 42
    assert meter.estimate_call("ab", "cde", "Draft") == 5 + 42
    # The latest reply of a kind is what the next one is expected to be.
    meter.record(
        system_text="ab", user_text="cde", output_text="", kind="Draft", reported=Usage(5, 30)
    )
    assert meter.expected_output("Draft") == 30
    assert meter.expected_output("OneCheck") == first_output("OneCheck", shape)
    assert meter.used == 82
    assert meter.estimated is False


def test_book_drafts_are_expected_to_be_larger() -> None:
    shape = AskShape(task_tokens=40, instruction_count=4, passage_tokens=2_000, response="book")
    meter = TokenMeter(None, estimator=lambda text: 0, shape=shape)
    assert meter.estimate_call("", "", "Draft") == first_output("Draft", shape)
    assert meter.estimate_call("", "", "Draft") > TokenMeter(
        None, estimator=lambda text: 0, shape=AskShape(40, 4, 2_000)
    ).estimate_call("", "", "Draft")


def test_meter_calibrates_prompt_estimates_against_reported_usage() -> None:
    meter = TokenMeter(1_000, estimator=lambda text: len(text))
    # The provider counted twice what the estimator did: later estimates double.
    meter.record(
        system_text="aaaa", user_text="bbbb", output_text="", kind="OneCheck", reported=Usage(16, 4)
    )
    assert meter.estimate_call("aaaa", "bbbb", "OneCheck") == 16 + 4
    # The correction is clamped so one strange report cannot run away.
    meter.record(
        system_text="aaaa",
        user_text="bbbb",
        output_text="",
        kind="OneCheck",
        reported=Usage(8000, 4),
    )
    assert meter.estimate_call("aaaa", "bbbb", "OneCheck") == 8 * 4 + 4
    meter.record(
        system_text="aaaa", user_text="bbbb", output_text="", kind="OneCheck", reported=Usage(1, 4)
    )
    assert meter.estimate_call("aaaa", "bbbb", "OneCheck") == 4 + 4


def test_meter_fits_remaining_and_exceeded() -> None:
    meter = TokenMeter(100, estimator=lambda text: 0)
    assert meter.fits(100)
    assert not meter.fits(101)
    meter.record(system_text="", user_text="", output_text="", kind="Draft", reported=Usage(60, 30))
    assert meter.remaining == 10
    assert meter.fits(10)
    assert not meter.fits(11)
    before = meter.exceeded
    meter.record(system_text="", user_text="", output_text="", kind="Draft", reported=Usage(20, 5))
    after = meter.exceeded
    assert (before, after) == (False, True)
    assert meter.used == 115
    assert meter.remaining == 0
    # Over the limit, nothing more fits.
    assert not meter.fits(1)
    assert not meter.fits(0)
    assert meter.describe() == "115 of 100 tokens used"


def test_meter_rejects_a_limit_below_one() -> None:
    with pytest.raises(ValueError):
        TokenMeter(0)
