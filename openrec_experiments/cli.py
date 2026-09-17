import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="OpenRec reproducible experiment runner")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "run"]:
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        if name == "run":
            sub.add_argument("--model", choices=["popularity", "lr", "fm"])
            sub.add_argument("--seed", type=int)
    summary = commands.add_parser("report")
    summary.add_argument("--runs", nargs="+", required=True, type=Path)
    summary.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "report":
        from .report import report
        report(args.runs, args.output)
        return
    config = json.loads(args.config.read_text())
    if args.command == "prepare":
        from .data import prepare
        prepare(config, args.output)
    else:
        from .runner import run
        if args.model is not None:
            config["model"] = args.model
        if args.seed is not None:
            config["seed"] = args.seed
        print(json.dumps(run(config, args.output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
