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
            sub.add_argument("--model", choices=["popularity", "lr", "fm", "lightgbm", "transformer"])
            sub.add_argument("--seed", type=int)
    summary = commands.add_parser("report")
    summary.add_argument("--runs", nargs="+", required=True, type=Path)
    summary.add_argument("--output", required=True, type=Path)
    embedding = commands.add_parser("embed-titles")
    embedding.add_argument("--articles", required=True, type=Path)
    embedding.add_argument("--output", required=True, type=Path)
    embedding.add_argument("--model", default="intfloat/multilingual-e5-small")
    embedding.add_argument("--batch-size", type=int, default=256)
    official_music = commands.add_parser("official-music")
    official_music.add_argument("--config", required=True, type=Path)
    official_music.add_argument("--evaluator", required=True, type=Path)
    official_music.add_argument("--output", required=True, type=Path)
    official_synerise = commands.add_parser("official-synerise-profiles")
    official_synerise.add_argument("--config", required=True, type=Path)
    official_synerise.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "report":
        from .report import report
        report(args.runs, args.output)
        return
    if args.command == "embed-titles":
        from .datasets.ebnerd.semantic import embed_titles
        embed_titles(args.articles, args.output, args.model, args.batch_size)
        return
    if args.command == "official-music":
        from .datasets.music_crs_2026.official import run
        config = json.loads(args.config.read_text())
        print(json.dumps(run(config, args.evaluator, args.output), indent=2))
        return
    if args.command == "official-synerise-profiles":
        from .datasets.synerise_2025.official import create_profiles
        config = json.loads(args.config.read_text())
        print(json.dumps(create_profiles(config, args.output), indent=2))
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
