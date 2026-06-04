from __future__ import annotations

import pytest

from av_robustbench.models import list_models, load_model
from av_robustbench.models.registry import ModelNotAvailableError


def test_builtin_registry_has_certav_entries() -> None:
    names = {item["name"] for item in list_models()}
    assert "certav_sigma025" in names
    assert "certav_sigma100" in names


def test_load_without_checkpoint_is_explicit() -> None:
    with pytest.raises(ModelNotAvailableError):
        load_model("certav_sigma100")

