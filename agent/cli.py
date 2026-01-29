from __future__ import annotations

import argparse
from pathlib import Path

from .loop import run
from .toolbox.docs import search
from .toolbox.orders import lookup_order
from .tools import Registry
from .trace import RUNS, render


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
        trace = run(args.goal, Registry(search, lookup_order), max_steps=args.max_steps)
        print(trace.answer)
        print(f"\n[{trace.outcome} · {len(trace.steps)} steps · {trace.tokens:,} tok · run {trace.run_id}]")
    else:
        path = RUNS / f"{args.run_id}.json" if args.run_id else max(RUNS.glob("*.json"), key=lambda p: p.stat().st_mtime)
        print(render(Path(path)))


if __name__ == "__main__":
    main()
