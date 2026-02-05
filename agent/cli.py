from __future__ import annotations

import argparse
from pathlib import Path

from .loop import run
from .toolbox.docs import search_docs
from .toolbox.orders import add_note, issue_refund, lookup_order
from .tools import Registry
from .trace import RUNS, render


def ask_human(name: str, arguments: dict) -> bool:
    print(f"\n  ⚠ the agent wants to call {name}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})")
    try:
        return input("  allow? [y/N] ").strip().lower() == "y"
    except EOFError:      # non-interactive: irreversible actions are declined
        return False


def main() -> None:
    ap = argparse.ArgumentParser(prog="agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("goal")
    r.add_argument("--max-steps", type=int, default=8)
    t = sub.add_parser("trace")
    t.add_argument("run_id", nargs="?", help="default: latest run")
    args = ap.parse_args()

    if args.cmd == "run":
        tools = Registry(search_docs, lookup_order, add_note, issue_refund, approve=ask_human)
        trace = run(args.goal, tools, max_steps=args.max_steps)
        print(trace.answer)
        print(f"\n[{trace.outcome} · {len(trace.steps)} steps · {trace.tokens:,} tok · ${trace.cost_usd:.4f} · run {trace.run_id}]")
    else:
        path = RUNS / f"{args.run_id}.json" if args.run_id else max(RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime)
        print(render(Path(path)))


if __name__ == "__main__":
    main()
