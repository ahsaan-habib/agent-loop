"""Offline by default: the model is a script of replies. test_smoke_ollama.py
runs the loop against a real local model and is opt-in."""
import pytest

from agent import cli, trace
from agent.model import Reply, ToolCall
from agent.toolbox import orders


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(cli, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(orders, "OUTBOX", tmp_path / "outbox")
    return tmp_path


class ScriptedModel:
    model = "scripted"

    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def call(name, **args):
    return Reply("", [ToolCall("c", name, args)], input_tokens=100, output_tokens=10)


def answer(text):
    return Reply(text, [], input_tokens=120, output_tokens=30)
