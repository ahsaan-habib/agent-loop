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


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    schema: dict

    def __call__(self, **kw):
        return self.fn(**kw)


def tool(fn: Callable[..., Any]) -> Tool:
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
    return Tool(fn.__name__, fn, schema)


class Registry:
    def __init__(self, *tools: Tool):
        self.tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict]:
        return [t.schema for t in self.tools.values()]

    def __getitem__(self, name: str) -> Tool:
        return self.tools[name]
