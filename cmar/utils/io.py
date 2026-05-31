from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_tensor(tensor: torch.Tensor, path: str | Path, dtype: str = "float16") -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tensor = tensor.detach().cpu()
    if dtype == "float16":
        tensor = tensor.half()
    elif dtype == "float32":
        tensor = tensor.float()
    else:
        raise ValueError(f"Unsupported feature dtype: {dtype}")
    torch.save(tensor, path)


def load_tensor(path: str | Path, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.load(Path(path), map_location="cpu")
    tensor = tensor.to(dtype=dtype)
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
