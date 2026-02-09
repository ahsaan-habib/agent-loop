import json

from agent.toolbox import docs
from agent.toolbox.orders import add_note, issue_refund, lookup_order
from agent.tools import Registry


def test_schema_comes_from_types_and_docstring():
    fn = docs.search_docs.schema["function"]
    assert fn["name"] == "search_docs" and "ONLY" in fn["description"]
    props = fn["parameters"]["properties"]
    assert props["section"]["enum"] == ["laravel", "eloquent", "filament", "livewire"]
    assert fn["parameters"]["required"] == ["query"] and "Full sentences" in props["query"]["description"]
    refund = issue_refund.schema["function"]
    assert refund["parameters"]["properties"]["amount_eur"]["type"] == "number"
    assert refund["description"].endswith("Requires human confirmation; may be declined.")


def test_bad_arguments_and_unknown_tools_become_observations():
    reg = Registry(lookup_order, add_note)
    assert "invalid arguments for lookup_order" in reg.call("lookup_order", {"order_id": 5, "x": 1})["error"]
    assert "no tool called 'delete_db'" in reg.call("delete_db", {})["error"]
    assert reg.call("lookup_order", {"order_id": "A-1001"})["customer"] == "Maria Lindqvist"
    assert reg.call("lookup_order", {"order_id": "A-9999"}) == {"error": "no order A-9999"}


def test_irreversible_needs_a_human(sandbox):
    asked = []
    deny = Registry(issue_refund)
    assert "declined" in deny.call("issue_refund", {"order_id": "A-1002", "amount_eur": 29, "reason": "r"})
    assert not (sandbox / "outbox").exists()
    allow = Registry(issue_refund, approve=lambda n, a: asked.append(n) or True)
    assert allow.call("issue_refund", {"order_id": "A-1002", "amount_eur": 99, "reason": "r"})["error"].startswith("amount")
    ok = allow.call("issue_refund", {"order_id": "A-1002", "amount_eur": "29", "reason": "duplicate charge"})
    assert ok["refunded"]["amount_eur"] == 29.0 and asked == ["issue_refund", "issue_refund"]
    assert json.loads((sandbox / "outbox" / "refunds.jsonl").read_text())["order_id"] == "A-1002"


def test_reversible_writes_are_allowed_without_asking(sandbox):
    reg = Registry(add_note)        # default approver denies everything
    assert reg.call("add_note", {"order_id": "A-1001", "note": "checked"})["added"]["note"] == "checked"


def test_search_docs_filters_section_and_low_scores(monkeypatch):
    from rag_grounded_stub import Chunk, Hybrid, Rerank

    monkeypatch.setattr(docs, "_retrieval", lambda: (Hybrid(), Rerank()))
    hits = docs.search_docs(query="global scope", section="eloquent")
    assert [h["source"] for h in hits] == ["laravel/eloquent.md"]
    assert docs.search_docs(query="global scope", section="filament") == []
