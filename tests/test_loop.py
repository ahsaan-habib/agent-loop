import json
import sys

import httpx
import pytest

from agent import cli, loop, trace
from agent.budget import Budget
from agent.context import compact, count_tokens
from agent.model import Model
from agent.toolbox.orders import lookup_order
from agent.tools import Registry
from agent.trace import render
from conftest import ScriptedModel, answer, call


def tools():
    return Registry(lookup_order)


def test_answers_after_one_lookup():
    m = ScriptedModel(call("lookup_order", order_id="A-1003"), answer("Order A-1003 is past due."))
    t = loop.run("why is A-1003 blocked?", tools(), model=m)
    assert t.outcome == "answered" and len(t.steps) == 2 and t.tokens == 260
    tool_msg = m.seen[1][-1]
    assert tool_msg["role"] == "tool" and "card declined" in tool_msg["content"]
    assert t.steps[0].calls[0]["observation"].startswith('{"order_id": "A-1003"')
    assert "outcome  answered" in render(trace.RUNS / f"{t.run_id}.json")


def test_repeat_is_named_then_stopped():
    same = call("lookup_order", order_id="A-1001")
    m = ScriptedModel(same, same, same)
    t = loop.run("x", tools(), model=m)
    assert t.outcome == "repetition" and "already ran that exact call" in m.seen[2][-1]["content"]
    assert t.steps[1].note.startswith("repetition guard")


def test_step_budget_and_cost_cap():
    m = ScriptedModel(*[call("lookup_order", order_id=f"A-100{i}") for i in range(3)])
    assert loop.run("x", tools(), model=m, max_steps=3).outcome == "step_budget"
    m = ScriptedModel(call("lookup_order", order_id="A-1001"), answer("never reached"))
    t = loop.run("x", tools(), model=m, budget=Budget(cap_tokens=50))
    assert t.outcome == "cost_cap" and "I couldn't finish this" in t.answer


def test_compaction_pins_system_and_goal():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "GOAL"}]
    msgs += [{"role": "tool", "content": "x" * 4000} for _ in range(12)]
    summariser = ScriptedModel(answer("looked up A-1001 and A-1002"))
    out, done = compact(msgs, summariser, limit=6000)
    assert done and out[:2] == msgs[:2] and out[2]["content"] == "[earlier steps] looked up A-1001 and A-1002"
    assert out[3:] == msgs[-6:] and count_tokens(out) < count_tokens(msgs)
    assert compact(msgs[:3], summariser) == (msgs[:3], False)


def test_model_parses_native_tool_calls():
    def handler(req):
        body = json.loads(req.content)
        assert body["tools"][0]["function"]["name"] == "lookup_order" and body["think"] is False
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "lookup_order", "arguments": {"order_id": "A-1001"}}}]},
            "prompt_eval_count": 200, "eval_count": 12})
    m = Model(base_url="http://x")
    m.http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    r = m.chat([{"role": "user", "content": "hi"}], tools=tools().schemas())
    assert r.tool_calls[0].arguments == {"order_id": "A-1001"} and r.input_tokens == 200
    assert r.as_message()["tool_calls"][0]["function"]["name"] == "lookup_order"


def test_cli_stats_and_trace(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["agent", "trace"])
    with pytest.raises(SystemExit, match="no runs yet"):
        cli.main()
    loop.run("x", tools(), model=ScriptedModel(call("lookup_order", order_id="A-1001"), answer("done")))
    for argv in (["stats"], ["trace"]):
        monkeypatch.setattr(sys, "argv", ["agent", *argv])
        cli.main()
    out = capsys.readouterr().out
    assert "answered" in out and "lookup_order(order_id='A-1001')" in out


def test_ask_human_declines_when_not_interactive(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda p: (_ for _ in ()).throw(EOFError()))
    assert cli.ask_human("issue_refund", {"order_id": "A-1"}) is False
