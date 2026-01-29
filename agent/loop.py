"""agent/loop.py — the entire concept."""
from __future__ import annotations

import json

from .model import Model
from .tools import Registry
from .trace import RunTrace, Step, summarise_observation

SYSTEM = {"role": "system", "content": "You are a helpful assistant. Use tools when you need information."}


def serialise(result) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


def run(goal: str, tools: Registry, model: Model | None = None, max_steps: int = 8) -> RunTrace:
    model = model or Model()
    trace = RunTrace(goal)
    messages = [SYSTEM, {"role": "user", "content": goal}]
    for n in range(1, max_steps + 1):
        reply = model.chat(messages, tools=tools.schemas())          # THINK
        step = Step(n, reply.input_tokens, reply.output_tokens)
        trace.steps.append(step)
        messages.append(reply.as_message())
        if not reply.tool_calls:                                     # STOP?
            trace.finish("answered", reply.content)                  # ...answered
            return trace
        for call in reply.tool_calls:                                # ACT
            result = tools[call.name](**call.arguments)
            step.calls.append({"name": call.name, "arguments": call.arguments,
                               "observation": summarise_observation(result)})
            messages.append({"role": "tool", "content": serialise(result)})   # OBSERVE
    trace.finish("step_budget", "Could not finish within the step budget.")  # GIVE UP
    return trace
