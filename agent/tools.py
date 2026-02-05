"""@tool turns a typed, documented function into a tool schema.

The schema is not plumbing: the model chooses tools almost entirely from
names, descriptions and parameter docs. It's the highest-leverage prompt
surface in the system, so it's written by hand in the docstring.
"""
from __future__ import annotations

import inspect
import re
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import ValidationError, validate_call

_JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _json_type(tp) -> dict:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return _json_type(args[0]) if len(args) == 1 else {}
    if origin is Literal:
        return {"type": "string", "enum": list(typing.get_args(tp))}
    if origin in (list, typing.List):
        (item,) = typing.get_args(tp) or (str,)
        return {"type": "array", "items": _json_type(item)}
    return {"type": _JSON.get(tp, "string")}


def _parse_doc(doc: str) -> tuple[str, dict[str, str]]:
    doc = inspect.cleandoc(doc or "")
    desc, _, args_block = doc.partition("Args:")
    params: dict[str, str] = {}
    current = None
    for line in args_block.splitlines():
        m = re.match(r"\s*(\w+):\s*(.*)", line)
        if m:
            current = m.group(1)
            params[current] = m.group(2).strip()
        elif current and line.strip():
            params[current] += " " + line.strip()
    return " ".join(desc.split()), params


# Consequence classes, fixed at registration:
#   read              search, fetch, look up         free to call, inside the budget
#   write_reversible  draft, tag, add a note         allowed, logged, undoable
#   write_irreversible send, refund, delete          a human confirms. Always.
# No "auto-approve above 0.9 confidence": a model's stated confidence is a
# token sequence, not a calibrated probability.
Consequence = Literal["read", "write_reversible", "write_irreversible"]
Approver = Callable[[str, dict], bool]


def deny_all(name: str, arguments: dict) -> bool:
    return False


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    schema: dict
    consequence: Consequence = "read"

    def __post_init__(self):
        self._validated = validate_call(self.fn)

    def __call__(self, **kw):
        """Never raises into the loop: a bad call becomes an observation the
        model can read and recover from. A silent drop it would just repeat."""
        try:
            return self._validated(**kw)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            return {"error": f"invalid arguments for {self.name}: {problems}"}
        except Exception as e:  # the tool itself failed
            return {"error": f"{self.name} failed: {type(e).__name__}: {e}"}


def tool(fn: Callable[..., Any] | None = None, *, consequence: Consequence = "read"):
    if fn is None:
        return lambda f: _build(f, consequence)
    return _build(fn, consequence)


def _build(fn: Callable[..., Any], consequence: Consequence) -> Tool:
    hints = typing.get_type_hints(fn)
    desc, param_docs = _parse_doc(fn.__doc__)
    props, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        props[name] = {**_json_type(hints.get(name, str)), "description": param_docs.get(name, "")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    schema = {"type": "function", "function": {
        "name": fn.__name__, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}
    if consequence == "write_irreversible":
        schema["function"]["description"] += " Requires human confirmation; may be declined."
    return Tool(fn.__name__, fn, schema, consequence)


class Registry:
    def __init__(self, *tools: Tool, approve: Approver = deny_all):
        self.tools = {t.name: t for t in tools}
        self.approve = approve

    def schemas(self) -> list[dict]:
        return [t.schema for t in self.tools.values()]

    def call(self, name: str, arguments: dict):
        tool = self.tools.get(name)
        if tool is None:   # tool hallucination: allowlist, answered as an observation
            return {"error": f"there is no tool called {name!r}. Available: {', '.join(self.tools)}"}
        if tool.consequence == "write_irreversible" and not self.approve(name, arguments):
            # a refusal the model can read is one it can recover from
            return {"declined": f"A human reviewed this {name} call and declined it. "
                                "Do not retry it; tell the user it needs manual handling."}
        return tool(**arguments)
