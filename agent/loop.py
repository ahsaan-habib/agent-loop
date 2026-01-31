"""agent/loop.py — the entire concept."""
from __future__ import annotations

import json

from .model import Model
from .tools import Registry
from .trace import RunTrace, Step, summarise_observation

SYSTEM = {"role": "system", "content": "You are a helpful assistant. Use tools when you need information."}


def serialise(result) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


def nudge(text: str) -> dict:
    return {"role": "user", "content": f"[agent runtime] {text}"}


def run(goal: str, tools: Registry, model: Model | None = None, max_steps: int = 8) -> RunTrace:
    model = model or Model()
    trace = RunTrace(goal)
    messages = [SYSTEM, {"role": "user", "content": goal}]
    seen: set[str] = set()
    for n in range(1, max_steps + 1):
        reply = model.chat(messages, tools=tools.schemas())          # THINK
        step = Step(n, reply.input_tokens, reply.output_tokens)
        trace.steps.append(step)
        messages.append(reply.as_message())
        if not reply.tool_calls:                                     # STOP?
            trace.finish("answered", reply.content)                  # ...answered
            return trace
        for call in reply.tool_calls:                                # ACT
            signature = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
            if signature in seen:
                # Don't just break — tell the model WHY, or the next turn repeats it.
                messages.append(nudge("You already ran that exact call. "
                                      "Use what you have, or say you cannot answer."))
                step.note = f"repetition guard: {call.name} with identical arguments"
                break
            seen.add(signature)
            result = tools[call.name](**call.arguments)
            step.calls.append({"name": call.name, "arguments": call.arguments,
                               "observation": summarise_observation(result)})
            messages.append({"role": "tool", "content": serialise(result)})   # OBSERVE
    trace.finish("step_budget", "Could not finish within the step budget.")  # GIVE UP
    return trace
