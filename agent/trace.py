"""One trace per run: every step, token count and outcome. Agent bugs are
quiet — nothing throws — so without this they aren't visible at all."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

RUNS = Path("runs")


@dataclass
class Step:
    n: int
    input_tokens: int
    output_tokens: int
    calls: list[dict] = field(default_factory=list)   # {name, arguments, observation}
    note: str = ""


@dataclass
class RunTrace:
    goal: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:6])
    started: float = field(default_factory=time.time)
    steps: list[Step] = field(default_factory=list)
    outcome: str = ""
    answer: str = ""
    seconds: float = 0.0
    cost_usd: float = 0.0

    @property
    def tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.steps)

    def finish(self, outcome: str, answer: str) -> None:
        self.outcome, self.answer = outcome, answer
        self.seconds = round(time.time() - self.started, 2)
        RUNS.mkdir(exist_ok=True)
        (RUNS / f"{self.run_id}.json").write_text(json.dumps(asdict(self), indent=2, default=str))


def summarise_observation(result) -> str:
    if isinstance(result, list):
        return f"{len(result)} hits"
    if isinstance(result, dict) and "error" in result:
        return f"error: {result['error']}"
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return text[:80] + ("…" if len(text) > 80 else "")


def render(path: Path) -> str:
    t = json.loads(path.read_text())
    out = [f"run {t['run_id']} · goal: {t['goal']!r}"]
    for s in t["steps"]:
        calls = s["calls"] or [{"name": "(answer)", "arguments": {}, "observation": ""}]
        for i, c in enumerate(calls):
            head = f"  step {s['n']}  think {s['input_tokens'] + s['output_tokens']:>6} tok" if i == 0 else " " * 27
            args = ", ".join(f"{k}={v!r}" for k, v in c["arguments"].items())
            out.append(f"{head}  →  {c['name']}({args})")
            if c.get("observation"):
                out.append(f"{'':>17}observe  {c['observation']}")
        if s.get("note"):
            out.append(f"{'':>17}^^ {s['note']}")
    total = sum(s["input_tokens"] + s["output_tokens"] for s in t["steps"])
    out.append(f"  outcome  {t['outcome']} · {len(t['steps'])} steps · {total:,} tok · "
               f"${t.get('cost_usd', 0):.4f} · {t['seconds']}s")
    return "\n".join(out)
