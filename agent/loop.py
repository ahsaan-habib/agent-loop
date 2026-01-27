"""agent/loop.py — the entire concept."""
from __future__ import annotations

import json

from .model import Model
from .tools import Registry

SYSTEM = {"role": "system", "content": "You are a helpful assistant. Use tools when you need information."}


def serialise(result) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


def run(goal: str, tools: Registry, model: Model | None = None, max_steps: int = 8) -> str:
    model = model or Model()
    messages = [SYSTEM, {"role": "user", "content": goal}]
    for step in range(max_steps):
        reply = model.chat(messages, tools=tools.schemas())          # THINK
        messages.append(reply.as_message())
        if not reply.tool_calls:                             # STOP?
            return reply.content                             # ...answered
        for call in reply.tool_calls:                        # ACT
            result = tools[call.name](**call.arguments)
            messages.append({"role": "tool", "content": serialise(result)})  # OBSERVE
    return "Could not finish within the step budget."         # GIVE UP
