"""Structured shapes for one draft step and one audit step."""

from pydantic import BaseModel, Field


class Draft(BaseModel):
    """What the drafter returns. The caller never sees this type."""

    answer: str = Field(
        description="The answer the end user should read. Cite passage ids inline as [p1]."
    )
    cited_ids: list[str] = Field(
        description="Passage ids that support the answer, without brackets. Example: p1."
    )


class DraftCheck(BaseModel):
    """What the auditor returns. Code uses ``revise`` as the stop condition."""

    follows_instructions: bool = Field(
        description="True only when the draft does what the user instructions require."
    )
    grounded: bool = Field(
        description=(
            "True only when every factual claim is supported by the passages and every "
            "cited id was provided. An honest statement that the answer is not in the "
            "passages is grounded."
        )
    )
    revise: bool = Field(
        description=(
            "True only when another draft could materially fix a real problem using these "
            "passages. False when the draft is acceptable, including an honest refusal."
        )
    )
    notes: str = Field(
        description="Concrete edits for the next draft. Empty when revise is false."
    )
    gaps: list[str] = Field(
        description=(
            "Omissions a revision can fix from these passages. Empty when nothing in the "
            "passages would improve the draft."
        )
    )
