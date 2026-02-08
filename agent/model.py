"""Ollama /api/chat with native tool calling."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("AGENT_MODEL", "qwen3:4b-instruct")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    def as_message(self) -> dict:
        msg = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}}
                                 for c in self.tool_calls]
        return msg


class Model:
    def __init__(self, model: str = MODEL, base_url: str = OLLAMA_URL, timeout: float = 120):
        self.model = model
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Reply:
        body = {"model": self.model, "messages": messages, "stream": False, "think": False,
                "options": {"temperature": 0.0}}
        if tools:
            body["tools"] = tools
        r = self.http.post("/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        msg = data["message"]
        calls = [ToolCall(id=f"call_{i}", name=c["function"]["name"],
                          arguments=c["function"].get("arguments") or {})
                 for i, c in enumerate(msg.get("tool_calls") or [])]
        return Reply(content=msg.get("content", ""), tool_calls=calls,
                     input_tokens=data.get("prompt_eval_count", 0),
                     output_tokens=data.get("eval_count", 0))
