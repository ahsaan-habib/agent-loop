"""Demo order tools over data/orders.json — stands in for a real billing API."""
from __future__ import annotations

import json
from pathlib import Path

from ..tools import tool

DATA = Path(__file__).resolve().parents[2] / "data" / "orders.json"


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
