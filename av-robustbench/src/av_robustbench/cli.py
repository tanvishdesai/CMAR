from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from av_robustbench.datasets import FakeAVCelebDataset, FeatureCacheDataset, LAVDFDataset
from av_robustbench.evaluate import benchmark, load_card, save_card_outputs
from av_robustbench.leaderboard import create_leaderboard_entry, update_leaderboard
from av_robustbench.models import list_models, load_model
from av_robustbench.utils.io import write_json
from av_robustbench.utils.seed import seed_everything


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="av-robustbench")
    parser.add_argument("--seed", type=int, default=2026)
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-models", help="List registered model adapters.")
    list_parser.set_defaults(func=_cmd_list_models)

    eval_parser = sub.add_parser("evaluate", help="Run the full robustness benchmark.")
    _add_model_dataset_args(eval_parser)
    eval_parser.add_argument("--attacks", nargs="*", default=[])
    eval_parser.add_argument("--certify", action="store_true")
    eval_parser.add_argument("--sigma", nargs="*", type=float, default=[0.25])
    eval_parser.add_argument("--degrade", action="store_true")
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--max-samples", type=int, default=None)
    eval_parser.add_argument("--device", type=str, default=None)
    eval_parser.add_argument("--n0", type=int, default=100)
    eval_parser.add_argument("--n", type=int, default=1000)
    eval_parser.add_argument("--alpha", type=float, default=0.001)
    eval_parser.set_defaults(func=_cmd_evaluate)

    cert_parser = sub.add_parser("certify", help="Run certification only.")
    _add_model_dataset_args(cert_parser)
    cert_parser.add_argument("--sigma", nargs="+", type=float, required=True)
    cert_parser.add_argument("--output", type=Path, required=True)
    cert_parser.add_argument("--max-samples", type=int, default=None)
    cert_parser.add_argument("--device", type=str, default=None)
    cert_parser.add_argument("--n0", type=int, default=100)
    cert_parser.add_argument("--n", type=int, default=1000)
    cert_parser.add_argument("--alpha", type=float, default=0.001)
    cert_parser.set_defaults(func=_cmd_certify)

    attack_parser = sub.add_parser("attack", help="Run attacks only.")
    _add_model_dataset_args(attack_parser)
    attack_parser.add_argument("--attacks", nargs="+", required=True)
    attack_parser.add_argument("--output", type=Path, required=True)
    attack_parser.add_argument("--max-samples", type=int, default=None)
    attack_parser.add_argument("--device", type=str, default=None)
    attack_parser.set_defaults(func=_cmd_attack)

    card_parser = sub.add_parser("card", help="Generate Markdown/LaTeX card outputs.")
    card_parser.add_argument("--results", type=Path, required=True)
    card_parser.add_argument("--output", type=Path, required=True)
    card_parser.set_defaults(func=_cmd_card)

    submit_parser = sub.add_parser("submit", help="Create or update leaderboard JSON.")
    submit_parser.add_argument("--card", type=Path, required=True)
    submit_parser.add_argument("--leaderboard", type=Path, required=True)
    submit_parser.add_argument("--model-name", type=str, default=None)
    submit_parser.add_argument("--paper-url", type=str, default=None)
    submit_parser.add_argument("--code-url", type=str, default=None)
    submit_parser.set_defaults(func=_cmd_submit)

    args = parser.parse_args(argv)
    seed_everything(args.seed)
    args.func(args)


def _add_model_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default="feature_cache", choices=["feature_cache", "fakeavceleb", "lavdf"])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--condition", type=str, default="clean")


def _cmd_list_models(args: argparse.Namespace) -> None:
    del args
    for item in list_models():
        print(f"{item['name']}\t{item.get('dataset') or '-'}\t{item.get('architecture') or '-'}")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    model = _load_model_from_args(args)
    dataset = _load_dataset_from_args(args)
    card = benchmark(
        model,
        dataset,
        model_name=args.model,
        dataset_name=args.dataset,
        attacks=args.attacks,
        certify=args.certify,
        sigmas=args.sigma,
        degrade=args.degrade,
        cache_dir=args.cache_dir,
        output_dir=args.output,
        max_samples=args.max_samples,
        device=args.device,
        certification_kwargs={"n0": args.n0, "n": args.n, "alpha": args.alpha},
    )
    save_card_outputs(card, args.output)
    print(args.output / "robustness_card.json")


def _cmd_certify(args: argparse.Namespace) -> None:
    model = _load_model_from_args(args)
    dataset = _load_dataset_from_args(args)
    card = benchmark(
        model,
        dataset,
        model_name=args.model,
        dataset_name=args.dataset,
        attacks=[],
        certify=True,
        sigmas=args.sigma,
        output_dir=args.output.parent,
        max_samples=args.max_samples,
        device=args.device,
        certification_kwargs={"n0": args.n0, "n": args.n, "alpha": args.alpha},
    )
    write_json(card.certification, args.output)
    print(args.output)


def _cmd_attack(args: argparse.Namespace) -> None:
    model = _load_model_from_args(args)
    dataset = _load_dataset_from_args(args)
    card = benchmark(
        model,
        dataset,
        model_name=args.model,
        dataset_name=args.dataset,
        attacks=args.attacks,
        certify=False,
        output_dir=args.output.parent,
        max_samples=args.max_samples,
        device=args.device,
    )
    write_json(card.attacks, args.output)
    print(args.output)


def _cmd_card(args: argparse.Namespace) -> None:
    card = load_card(args.results)
    paths = save_card_outputs(card, args.output)
    print(paths["markdown"])


def _cmd_submit(args: argparse.Namespace) -> None:
    card = load_card(args.card)
    entry = create_leaderboard_entry(
        card,
        model_name=args.model_name,
        paper_url=args.paper_url,
        code_url=args.code_url,
    )
    update_leaderboard(args.leaderboard, entry)
    print(args.leaderboard)


def _load_model_from_args(args: argparse.Namespace) -> Any:
    return load_model(args.model, checkpoint_path=args.checkpoint)


def _load_dataset_from_args(args: argparse.Namespace) -> FeatureCacheDataset:
    cls: type[FeatureCacheDataset]
    if args.dataset == "fakeavceleb":
        cls = FakeAVCelebDataset
    elif args.dataset == "lavdf":
        cls = LAVDFDataset
    else:
        cls = FeatureCacheDataset
    return cls(
        args.cache_dir,
        args.manifest,
        split=args.split,
        condition=args.condition,
        allow_partial_cache=True,
    )


if __name__ == "__main__":
    main()

