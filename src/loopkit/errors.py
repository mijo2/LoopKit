"""Errors a caller can fix by changing configuration or input."""


class LoopError(Exception):
    """The knowledge base, task, or local configuration stopped the loop."""
