"""agent/loop.py — the entire concept."""
from __future__ import annotations

import json
import time

from .budget import Budget
from .context import compact
from .model import Model
from .tools import Registry
from .trace import RunTrace, Step, summarise_observation

SYSTEM = {"role": "system", "content": "You are a helpful assistant. Use tools when you need information."}


def serialise(result) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


def nudge(text: str) -> dict:
    return {"role": "user", "content": f"[agent runtime] {text}"}


# Four exits, and only one of them is good:
#   answered      model replied with no tool calls
#   step_budget   max_steps reached              -> rising rate: tools too vague
#   cost_cap      spend over the per-run cap     -> should be ~0: a runaway
#   repetition    looped again after being told  -> the cheapest guard there is


def give_up(trace: RunTrace, outcome: str, why: str) -> RunTrace:
    trace.finish(outcome, f"I couldn't finish this ({why}). What I found so far is in the steps above, "
                          "but I don't have a complete answer.")
    return trace


def run(goal: str, tools: Registry, model: Model | None = None, max_steps: int = 8,
        budget: Budget | None = None) -> RunTrace:
    model = model or Model()
    budget = budget or Budget()
    trace = RunTrace(goal)
    messages = [SYSTEM, {"role": "user", "content": goal}]
    seen: set[str] = set()
    repeats = 0
    for n in range(1, max_steps + 1):
        if budget.exceeded():
            return give_up(trace, "cost_cap", f"hit the ${budget.cap_usd} / {budget.cap_tokens} token cap")
        t0 = time.perf_counter()
        messages, compacted = compact(messages, model)
        reply = model.chat(messages, tools=tools.schemas())          # THINK
        budget.add(time.perf_counter() - t0, reply.input_tokens, reply.output_tokens)
        trace.cost_usd = round(budget.usd, 6)
        step = Step(n, reply.input_tokens, reply.output_tokens, note="context compacted" if compacted else "")
        trace.steps.append(step)
        messages.append(reply.as_message())
        if not reply.tool_calls:                                     # STOP?
            trace.finish("answered", reply.content)                  # ...answered
            return trace
        for call in reply.tool_calls:                                # ACT
            signature = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
            if signature in seen:
                repeats += 1
                if repeats > 1:
                    return give_up(trace, "repetition", "kept repeating the same lookup")
                # Don't just break — tell the model WHY, or the next turn repeats it.
                messages.append(nudge("You already ran that exact call. "
                                      "Use what you have, or say you cannot answer."))
                step.note = f"repetition guard: {call.name} with identical arguments"
                break
            seen.add(signature)
            result = tools.call(call.name, call.arguments)
            step.calls.append({"name": call.name, "arguments": call.arguments,
                               "observation": summarise_observation(result)})
            messages.append({"role": "tool", "tool_name": call.name, "content": serialise(result)})  # OBSERVE
    return give_up(trace, "step_budget", f"ran out of steps after {max_steps}")    # GIVE UP
