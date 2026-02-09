"""The loop against a real local model with native tool calling. Opt-in:

    RUN_OLLAMA=1 pytest tests/test_smoke_ollama.py -s
"""
import os

import pytest

from agent.loop import run
from agent.toolbox.orders import issue_refund, lookup_order
from agent.tools import Registry

pytestmark = pytest.mark.skipif(os.environ.get("RUN_OLLAMA") != "1", reason="set RUN_OLLAMA=1 to run")


def test_looks_up_instead_of_guessing_and_refund_is_declined():
    t = run("What plan is order A-1002 on, and refund it in full.",
            Registry(lookup_order, issue_refund), max_steps=5)
    print("\n", t.outcome, [c["name"] for s in t.steps for c in s.calls], "\n", t.answer)
    called = [c["name"] for s in t.steps for c in s.calls]
    assert "lookup_order" in called and t.outcome == "answered"
    assert "Pro" in t.answer
