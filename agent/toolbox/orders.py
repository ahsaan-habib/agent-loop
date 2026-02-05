"""Demo order tools over data/orders.json — stands in for a real billing API."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..tools import tool

DATA = Path(__file__).resolve().parents[2] / "data" / "orders.json"
OUTBOX = Path("outbox")   # stands in for side effects; nothing leaves this machine


def _append(name: str, row: dict) -> dict:
    OUTBOX.mkdir(exist_ok=True)
    row = {"id": uuid.uuid4().hex[:8], "ts": time.time(), **row}
    with (OUTBOX / f"{name}.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _load() -> dict:
    return json.loads(DATA.read_text())


@tool
def lookup_order(order_id: str) -> dict:
    """Look up one order by id, e.g. "A-1001". Returns customer, plan, status,
    amount and any notes. Returns {"error": ...} if the id doesn't exist.

    Args:
        order_id: the order id, format "A-" followed by four digits
    """
    order = _load().get(order_id)
    return {"order_id": order_id, **order} if order else {"error": f"no order {order_id}"}


@tool(consequence="write_reversible")
def add_note(order_id: str, note: str) -> dict:
    """Attach an internal note to an order (visible to support staff only).
    Reversible: notes can be removed. Use it to record what you found or did.

    Args:
        order_id: the order id, e.g. "A-1001"
        note: one or two sentences
    """
    if order_id not in _load():
        return {"error": f"no order {order_id}"}
    return {"added": _append("notes", {"order_id": order_id, "note": note})}


@tool(consequence="write_irreversible")
def issue_refund(order_id: str, amount_eur: float, reason: str) -> dict:
    """Refund money to the customer on an order. Irreversible.
    Only for amounts up to the order amount. Do not call it to "check" a refund.

    Args:
        order_id: the order id, e.g. "A-1001"
        amount_eur: amount in euros, greater than 0 and at most the order amount
        reason: why, in one sentence — the reviewer reads this
    """
    order = _load().get(order_id)
    if not order:
        return {"error": f"no order {order_id}"}
    if not 0 < amount_eur <= order["amount_eur"]:
        return {"error": f"amount must be between 0 and {order['amount_eur']}"}
    return {"refunded": _append("refunds", {"order_id": order_id, "amount_eur": amount_eur, "reason": reason})}
