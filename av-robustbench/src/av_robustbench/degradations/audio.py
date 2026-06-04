from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from av_robustbench.degradations.specs import get_degradation_spec


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def add_audio_noise_snr(
    waveform: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    wave = waveform.astype(np.float32)
    signal_power = float(np.mean(wave**2))
    if signal_power <= 1e-12:
        return wave
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, math.sqrt(noise_power), size=wave.shape).astype(np.float32)
    return np.clip(wave + noise, -1.0, 1.0).astype(np.float32)


def codec_roundtrip(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    codec: str,
    bitrate: str,
) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("soundfile is required for audio codec degradations.") from exc
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required for audio codec degradations.")
    suffix = ".mp3" if codec == "mp3" else ".m4a"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        wav_path = tmp_dir / "input.wav"
        encoded_path = tmp_dir / f"encoded{suffix}"
        decoded_path = tmp_dir / "decoded.wav"
        sf.write(wav_path, waveform.astype(np.float32), sample_rate)
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav_path),
        ]
        if codec == "mp3":
            cmd.extend(["-b:a", bitrate])
        else:
            cmd.extend(["-c:a", "aac", "-b:a", bitrate])
        cmd.extend(["-threads", "1", str(encoded_path)])
        subprocess.run(cmd, check=True, timeout=60)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(encoded_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-threads",
                "1",
                str(decoded_path),
            ],
            check=True,
            timeout=60,
        )
        decoded, sr = sf.read(decoded_path, dtype="float32")
    if decoded.ndim > 1:
        decoded = decoded.mean(axis=1)
    if int(sr) != int(sample_rate):
        try:
            import librosa
        except ImportError as exc:
            raise ImportError("librosa is required when codec output sample rate changes.") from exc
        decoded = librosa.resample(decoded, orig_sr=sr, target_sr=sample_rate)
    return decoded.astype(np.float32)


def degrade_audio(
    waveform: np.ndarray,
    sample_rate: int,
    condition: str,
    *,
    seed: int = 2026,
) -> np.ndarray:
    spec = get_degradation_spec(condition)
    if spec.name in {"d7_mp3_128k", "d8_mp3_64k"}:
        return codec_roundtrip(waveform, sample_rate, codec="mp3", bitrate=str(spec.params["bitrate"]))
    if spec.name in {"d9_anoise_30db", "d10_anoise_20db"}:
        return add_audio_noise_snr(
            waveform,
            float(spec.params["snr_db"]),
            np.random.default_rng(seed),
        )
    if spec.name == "d11_h264_crf28":
        return codec_roundtrip(waveform, sample_rate, codec="aac", bitrate="128k")
    if spec.name == "d12_social":
        return codec_roundtrip(waveform, sample_rate, codec="aac", bitrate=str(spec.params["audio_bitrate"]))
    if not spec.audio:
        return waveform.astype(np.float32)
    raise ValueError(f"{condition} is not implemented as an audio degradation.")

