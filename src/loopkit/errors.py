"""Errors a caller can fix by changing configuration or input."""


class LoopError(Exception):
    """The knowledge base, task, or local configuration stopped the loop."""


class LoopWarning(UserWarning):
    """The loop runs, but in a way the caller probably did not intend.

    Raised with :func:`warnings.warn` when, for example, no knowledge base
    was given and the loop answers from no passages.
    """
