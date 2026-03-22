from __future__ import annotations

import argparse
from pathlib import Path

from chart_extractor.pipeline.orchestrator import run_pipeline, run_task2, run_task3, run_task4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bar chart extraction CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("run-pipeline", "run-task2", "run-task3", "run-task4"):
        p = sub.add_parser(cmd)
        p.add_argument("--input-dir", type=Path, default=None, help="Override input images directory")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "run-pipeline":
        run_pipeline(input_dir=args.input_dir)
    elif args.command == "run-task2":
        run_task2(input_dir=args.input_dir)
    elif args.command == "run-task3":
        run_task3(input_dir=args.input_dir)
    elif args.command == "run-task4":
        run_task4(input_dir=args.input_dir)


if __name__ == "__main__":
    main()

