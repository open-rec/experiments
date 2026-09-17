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
            sub.add_argument("--model", choices=["popularity", "lr", "fm", "transformer"])
            sub.add_argument("--seed", type=int)
    summary = commands.add_parser("report")
    summary.add_argument("--runs", nargs="+", required=True, type=Path)
    summary.add_argument("--output", required=True, type=Path)
    embedding = commands.add_parser("embed-titles")
    embedding.add_argument("--articles", required=True, type=Path)
    embedding.add_argument("--output", required=True, type=Path)
    embedding.add_argument("--model", default="intfloat/multilingual-e5-small")
    embedding.add_argument("--batch-size", type=int, default=256)
    content_embedding = commands.add_parser("embed-content")
    content_embedding.add_argument("--articles", required=True, type=Path)
    content_embedding.add_argument("--output", required=True, type=Path)
    content_embedding.add_argument("--column", default="body")
    content_embedding.add_argument("--model", default="intfloat/multilingual-e5-small")
    content_embedding.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.command == "report":
        from .report import report
        report(args.runs, args.output)
        return
    if args.command == "embed-titles":
        from .semantic import embed_titles
        embed_titles(args.articles, args.output, args.model, args.batch_size)
        return
    if args.command == "embed-content":
        from .semantic import embed_article_text
        embed_article_text(
            args.articles, args.output, args.model, args.column, args.batch_size
        )
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
