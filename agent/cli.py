from __future__ import annotations

import argparse

from .loop import run
from .toolbox.docs import search
from .toolbox.orders import lookup_order
from .tools import Registry


def main() -> None:
    ap = argparse.ArgumentParser(prog="agent")
    ap.add_argument("goal")
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()
    print(run(args.goal, Registry(search, lookup_order), max_steps=args.max_steps))


if __name__ == "__main__":
    main()
