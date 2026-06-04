from __future__ import annotations

from pathlib import Path

from av_robustbench.datasets.feature_cache import FeatureCacheDataset


class FakeAVCelebDataset(FeatureCacheDataset):
    """FakeAVCeleb cached-feature dataset."""

    dataset_name = "fakeavceleb"

    def __init__(
        self,
        cache_dir: str | Path,
        manifest_csv: str | Path | None = None,
        *,
        split: str = "test",
        condition: str = "clean",
        allow_partial_cache: bool = False,
    ) -> None:
        super().__init__(
            cache_dir,
            manifest_csv,
            split=split,
            condition=condition,
            allow_partial_cache=allow_partial_cache,
        )

