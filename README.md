# agent-loop

An agent is a loop. This is that loop written by hand — no framework — on a
local model (`qwen3:4b-instruct` via Ollama's native tool calling), with a guardrail on
every line that has ever gone wrong.

```python
for step in range(max_steps):
    reply = model.chat(messages, tools=schemas)   # THINK
    if not reply.tool_calls:                      # STOP?
        return reply.content                      #   answered
    for call in reply.tool_calls:                 # ACT
        result = tools[call.name](**call.arguments)
        messages.append(observation(result))      # OBSERVE
return "could not finish"                         # GIVE UP
```

The model emits a structured request; your code runs a function; you hand the
result back. The intelligence is rented, the control flow is yours.
`agent/loop.py` is that loop plus the guards below.

## Guards

| Problem | Guard | Where |
|---|---|---|
| polite infinite loop | repetition guard: same tool + same args → tell the model it looped; twice → stop | `loop.py` |
| semantic loop (rephrased searches) | tool descriptions that say what the tool *can't* do and when to stop | `toolbox/docs.py` |
| context drift | compaction that pins system prompt + goal, summarises the middle, keeps the last turns verbatim | `context.py` |
| tool hallucination | allowlist; unknown tools and bad arguments come back as observations, not exceptions | `tools.py` |
| silent cost blowout | per-run cost and token cap that stops the loop | `budget.py` |
| irreversible actions | consequence classes; `write_irreversible` always needs a human yes | `tools.py`, `cli.py` |
| "did it work?" | named termination reason on every run + a trace per run | `trace.py` |

Termination reasons: `answered`, `step_budget`, `cost_cap`, `repetition`.
Only the first is good; the others give up visibly instead of returning
nothing.

## The tool schema is the prompt

The model picks tools from names, descriptions and parameter docs. `@tool`
builds the JSON schema from the type hints and the docstring, so the docstring
is where behaviour gets written:

```python
@tool
def search_docs(query: str, section: Literal["laravel", "eloquent", "filament", "livewire"] | None = None):
    """Search the Laravel, Filament and Livewire documentation.
    Returns up to 6 passages ...
    Covers official framework documentation ONLY — not Stack Overflow, ...
    If the first call returns nothing relevant, do not rephrase and retry:
    the answer is very likely not in this corpus. Say so instead.
    """
```

## Demo tools

| Tool | Class | Does |
|---|---|---|
| `search_docs` | read | hybrid retrieval + rerank from [rag-grounded](https://github.com/ahsaan-habib/rag-grounded) |
| `lookup_order` | read | `data/orders.json` fixture |
| `add_note` | write_reversible | appends to `outbox/notes.jsonl` |
| `issue_refund` | write_irreversible | asks you in the terminal, then appends to `outbox/refunds.jsonl` |

Nothing leaves the machine; the outbox stands in for real side effects.

## Run it

```bash
make install model && source .venv/bin/activate
agent run "Does Filament support nested resources?"
agent run "Order A-1003 was charged twice in January, refund the duplicate"
agent trace            # step-by-step trace of the last run
agent stats            # termination reasons across runs/
```

`AGENT_COST_CAP_USD`, `AGENT_TOKEN_CAP` and `AGENT_COMPUTE_USD_PER_HOUR` set
the budget; local runs are costed by wall time.
