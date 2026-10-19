"""Shared test setup.

The optimize step is on by default for users. Tests that script the model
turn it off here so their reply lists stay one draft plus its checks; the
step has its own tests in ``test_optimize.py``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_auto_optimize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOPKIT_AUTO_OPTIMIZE", "0")
