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


class InstructionVerdict(BaseModel):
    """Whether the draft follows one numbered instruction."""

    number: int = Field(description="The instruction number from the prompt, starting at 1.")
    followed: bool = Field(
        description="True only when the draft follows this instruction. Judge this instruction alone."
    )
    note: str = Field(
        description="What to change so the draft follows this instruction. Empty when followed is true."
    )


class OneCheck(BaseModel):
    """Whether a draft follows the single instruction in the prompt."""

    followed: bool = Field(
        description="True only when the draft follows this one instruction. Judge this instruction alone."
    )
    note: str = Field(
        description="What to change so the draft follows this instruction. Empty when followed is true."
    )


class GroundingCheck(BaseModel):
    """Whether a draft stays inside the passages. Instructions are checked separately."""

    grounded: bool = Field(
        description=(
            "True only when every factual claim is supported by the passages and every "
            "cited id was provided. An honest statement that the answer is not in the "
            "passages is grounded."
        )
    )
    revise: bool = Field(
        description=(
            "True only when another draft could materially fix a grounding problem or an "
            "omission using these passages. False when the draft is acceptable on the passages, "
            "including an honest refusal. Do not set this because of length or the instructions."
        )
    )
    notes: str = Field(
        description="Concrete grounding edits for the next draft. Empty when revise is false."
    )
    gaps: list[str] = Field(
        description=(
            "Omissions a revision can fix from these passages. Empty when nothing in the "
            "passages would improve the draft."
        )
    )


class DraftCheck(BaseModel):
    """What the auditor returns when every instruction is judged in one call."""

    instructions: list[InstructionVerdict] = Field(
        description=(
            "One verdict for every numbered instruction, using those numbers. "
            "Judge each instruction on its own."
        )
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
