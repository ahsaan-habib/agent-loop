"""agent/context.py — keep the ends, summarise the middle.

The context window is a budget, not a memory. Every turn re-sends everything,
and observations are the biggest messages in a run.

This replaced a sliding window that dropped the oldest messages — which on
turn 7 meant dropping the original goal. The agent then (reasonably, given
what it could see) answered a different question. Nothing errored.
"""
from __future__ import annotations

import json

SUMMARISE_PROMPT = {"role": "system", "content": (
    "Summarise these earlier agent steps for the agent itself. Keep: which tools "
    "were called with what arguments, the key facts each returned (ids, numbers, "
    "names, sources), and anything ruled out. Drop pleasantries and raw text. "
    "Plain prose, under 200 words.")}


def count_tokens(messages: list[dict]) -> int:
    # ~4 chars/token is close enough to decide *when* to compact
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 4


def render(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        if m.get("tool_calls"):
            calls = "; ".join(f"{c['function']['name']}({json.dumps(c['function']['arguments'])})"
                              for c in m["tool_calls"])
            lines.append(f"assistant called: {calls}")
        elif m["content"]:
            lines.append(f"{m['role']}: {m['content'][:2000]}")
    return "\n".join(lines)


def compact(messages: list[dict], model, limit: int = 6000) -> tuple[list[dict], bool]:
    if count_tokens(messages) < limit:
        return messages, False
    head = messages[:2]     # system prompt + the original goal. Never drop these.
    tail = messages[-6:]    # the last few turns verbatim — the model is mid-thought
    middle = messages[2:-6]
    if not middle:
        return messages, False
    summary = model.chat([SUMMARISE_PROMPT, {"role": "user", "content": render(middle)}]).content
    bridge = {"role": "assistant", "content": f"[earlier steps] {summary}"}
    return head + [bridge] + tail, True
