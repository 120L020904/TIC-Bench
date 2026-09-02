from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import run_evaluation
from .inference import run_answers
from .openai_template import create_openai_caller
from .report import build_report


KINDS = ("spatial", "visual", "temporal")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="ticbench")
    commands = root.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="generate benchmark answers")
    ask.add_argument("dataset_root", type=Path)
    ask.add_argument("--model", required=True)
    ask.add_argument("--kind", choices=KINDS)
    ask.add_argument("--mode", default="full", choices=("full", "text_only", "images_only", "question_images_only"))
    ask.add_argument("--dry-run", action="store_true")

    evaluate = commands.add_parser("evaluate", help="judge generated answers")
    evaluate.add_argument("dataset_root", type=Path)
    evaluate.add_argument(
        "--judge-model", required=True, nargs=3, metavar=("JUDGE_1", "JUDGE_2", "JUDGE_3"),
        help="three unique judge models used for majority voting",
    )
    evaluate.add_argument("--target", required=True)
    evaluate.add_argument("--kind", choices=KINDS)

    report = commands.add_parser("report", help="summarize evaluation files")
    report.add_argument("dataset_root", type=Path)
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "ask":
        caller = None if args.dry_run else create_openai_caller(args.model)
        result = run_answers(args.dataset_root, args.model, caller, kind=args.kind, mode=args.mode, dry_run=args.dry_run)
    elif args.command == "evaluate":
        if len(set(args.judge_model)) != 3:
            parser().error("--judge-model requires three unique model names")
        judges = {name: create_openai_caller(name) for name in args.judge_model}
        result = run_evaluation(args.dataset_root, args.target, judges, kind=args.kind)
    else:
        result = build_report(args.dataset_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
