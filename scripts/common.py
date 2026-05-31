from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cmar.config import CacheConfig, ModelConfig, TrainConfig

T = TypeVar("T")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dataclass_from_dict(cls: Type[T], data: dict[str, Any]) -> T:
    values = {}
    field_map = {f.name: f for f in fields(cls)}
    for name, field in field_map.items():
        if name not in data:
            continue
        value = data[name]
        if field.type is ModelConfig or name == "model":
            value = dataclass_from_dict(ModelConfig, value)
        values[name] = value
    return cls(**values)


def load_cache_config(path: str | Path) -> CacheConfig:
    return dataclass_from_dict(CacheConfig, load_json(path))


def load_train_config(path: str | Path) -> TrainConfig:
    return dataclass_from_dict(TrainConfig, load_json(path))


def add_path_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)


def apply_train_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if getattr(args, "cache_dir", None):
        config.cache_dir = args.cache_dir
    if getattr(args, "output_dir", None):
        config.output_dir = args.output_dir
    return config
