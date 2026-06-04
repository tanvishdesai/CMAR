from __future__ import annotations

import numpy as np
import pytest

from av_robustbench.core import RobustnessCard
from av_robustbench.degradations.audio import degrade_audio
from av_robustbench.degradations.specs import DEGRADATION_SPECS, get_degradation_spec
from av_robustbench.degradations.visual import degrade_frames
from av_robustbench.leaderboard import (
    LeaderboardEntry,
    create_leaderboard_entry,
    update_leaderboard,
    validate_entry,
)
from av_robustbench.utils.io import read_json


def test_degradation_specs_cover_expected_battery() -> None:
    assert len(DEGRADATION_SPECS) == 12
    social = get_degradation_spec("d12_social")
    assert social.visual
    assert social.audio
    assert social.requires_ffmpeg
    with pytest.raises(KeyError):
        get_degradation_spec("missing_condition")


def test_array_degradations_are_deterministic() -> None:
    frame = np.full((8, 8, 3), 128, dtype=np.uint8)
    first_frame = degrade_frames([frame], "d5_vnoise001", seed=123)[0]
    second_frame = degrade_frames([frame], "d5_vnoise001", seed=123)[0]
    assert first_frame.shape == frame.shape
    assert first_frame.dtype == np.uint8
    assert np.array_equal(first_frame, second_frame)
    assert not np.array_equal(first_frame, frame)

    waveform = np.linspace(-0.5, 0.5, 160, dtype=np.float32)
    first_audio = degrade_audio(waveform, 16_000, "d9_anoise_30db", seed=123)
    second_audio = degrade_audio(waveform, 16_000, "d9_anoise_30db", seed=123)
    assert first_audio.shape == waveform.shape
    assert first_audio.dtype == np.float32
    assert np.array_equal(first_audio, second_audio)
    assert not np.array_equal(first_audio, waveform)


def test_leaderboard_entry_from_card_and_replace(tmp_path) -> None:
    card = RobustnessCard(
        "certav_sigma100",
        "fakeavceleb",
        clean_metrics={"auc": 0.95, "accuracy": 0.9},
        attacks={"pgd_linf": {"eps": 0.1, "adversarial_accuracy": 0.82}},
        certification={
            "sigma_1.00": {
                "summary": {
                    "certified_accuracy_at_radii": {"r_0.25": 0.86, "r_1.00": 0.71},
                    "mean_certified_radius": 1.2,
                }
            }
        },
        degradations={"conditions": {"d1_jpeg75": {"auc": 0.8}, "d2_jpeg50": {"auc": 0.7}}},
        cross_dataset={"accuracy": 0.62},
    )
    entry = create_leaderboard_entry(card, paper_url="https://example.com/paper")
    assert entry.clean_auc == 0.95
    assert entry.robust_acc_eps010 == 0.82
    assert entry.cert_acc_r025 == 0.86
    assert entry.cert_acc_r100 == 0.71
    assert entry.degradation_avg_auc == 0.75
    assert entry.cross_dataset_acc == 0.62

    path = tmp_path / "leaderboard.json"
    update_leaderboard(path, entry)
    update_leaderboard(
        path,
        LeaderboardEntry("certav_sigma100", "fakeavceleb", clean_auc=0.99),
        replace=True,
    )
    data = read_json(path)
    assert len(data["models"]) == 1
    assert data["models"][0]["clean_auc"] == 0.99

    with pytest.raises(ValueError):
        validate_entry(LeaderboardEntry("bad", "fakeavceleb", clean_accuracy=1.1))
