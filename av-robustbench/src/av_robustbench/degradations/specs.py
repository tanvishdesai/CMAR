from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DegradationSpec:
    name: str
    description: str
    visual: bool
    audio: bool
    params: dict[str, Any] = field(default_factory=dict)
    requires_ffmpeg: bool = False


DEGRADATION_SPECS: dict[str, DegradationSpec] = {
    "d1_jpeg75": DegradationSpec(
        "d1_jpeg75", "JPEG compression with quality factor 75", True, False, {"quality": 75}
    ),
    "d2_jpeg50": DegradationSpec(
        "d2_jpeg50", "JPEG compression with quality factor 50", True, False, {"quality": 50}
    ),
    "d3_resize075": DegradationSpec(
        "d3_resize075", "Downsample to 75 percent and upsample back", True, False, {"scale": 0.75}
    ),
    "d4_resize050": DegradationSpec(
        "d4_resize050", "Downsample to 50 percent and upsample back", True, False, {"scale": 0.50}
    ),
    "d5_vnoise001": DegradationSpec(
        "d5_vnoise001", "Add Gaussian visual noise with sigma 0.01", True, False, {"sigma": 0.01}
    ),
    "d6_vnoise002": DegradationSpec(
        "d6_vnoise002", "Add Gaussian visual noise with sigma 0.02", True, False, {"sigma": 0.02}
    ),
    "d7_mp3_128k": DegradationSpec(
        "d7_mp3_128k", "MP3 audio roundtrip at 128 kbps", False, True, {"bitrate": "128k"}, True
    ),
    "d8_mp3_64k": DegradationSpec(
        "d8_mp3_64k", "MP3 audio roundtrip at 64 kbps", False, True, {"bitrate": "64k"}, True
    ),
    "d9_anoise_30db": DegradationSpec(
        "d9_anoise_30db", "Add audio noise at 30 dB SNR", False, True, {"snr_db": 30.0}
    ),
    "d10_anoise_20db": DegradationSpec(
        "d10_anoise_20db", "Add audio noise at 20 dB SNR", False, True, {"snr_db": 20.0}
    ),
    "d11_h264_crf28": DegradationSpec(
        "d11_h264_crf28", "H.264 video re-encoding at CRF 28 with AAC audio", True, True, {"crf": 28}, True
    ),
    "d12_social": DegradationSpec(
        "d12_social",
        "Social-media style chain: resize 0.75x, JPEG 75, H.264 CRF 28, AAC 128k",
        True,
        True,
        {"quality": 75, "scale": 0.75, "crf": 28, "audio_bitrate": "128k"},
        True,
    ),
}


def get_degradation_spec(name: str | DegradationSpec) -> DegradationSpec:
    if isinstance(name, DegradationSpec):
        return name
    try:
        return DEGRADATION_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown degradation `{name}`. Available: {', '.join(DEGRADATION_SPECS)}") from exc

