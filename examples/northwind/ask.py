"""Ask the Northwind knowledge base.

export XAI_API_KEY="your_api_key"  # or OPENAI_API_KEY, or ANTHROPIC_API_KEY
python examples/northwind/ask.py
python examples/northwind/ask.py How do I rotate an API key?
"""

from __future__ import annotations

import sys
from pathlib import Path

from loopkit import Loop

ROOT = Path(__file__).parent
DEFAULT_TASK = "I was charged 10 days ago. Can I get a refund?"


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    instructions = [
        line.strip()
        for line in (ROOT / "instructions.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    loop = Loop(knowledge=ROOT / "knowledge", instructions=instructions)
    print(loop.ask(question))


if __name__ == "__main__":
    main()
