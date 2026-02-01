"""The context window is a budget, not a memory. Every turn re-sends
everything, and observations are the biggest messages in a run."""
from __future__ import annotations

import json


def count_tokens(messages: list[dict]) -> int:
    # ~4 chars/token is close enough to decide *when* to trim
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 4


def trim(messages: list[dict], limit: int = 6000, keep: int = 8) -> list[dict]:
    if count_tokens(messages) < limit:
        return messages
    return messages[:1] + messages[-keep:]
