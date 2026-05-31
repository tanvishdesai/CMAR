from __future__ import annotations

import io
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DegradationSpec:
    condition: str
    display_name: str
    visual: bool
    audio: bool
    description: str


DEGRADATION_SPECS: Dict[str, DegradationSpec] = {
    "d1_jpeg75": DegradationSpec("d1_jpeg75", "JPEG QF=75", True, False, "visual JPEG compression"),
    "d2_jpeg50": DegradationSpec("d2_jpeg50", "JPEG QF=50", True, False, "heavy visual JPEG compression"),
    "d3_resize075": DegradationSpec("d3_resize075", "Resize 0.75x", True, False, "visual down/up scaling"),
    "d4_resize050": DegradationSpec("d4_resize050", "Resize 0.50x", True, False, "heavy visual down/up scaling"),
    "d5_vnoise001": DegradationSpec("d5_vnoise001", "Visual noise 0.01", True, False, "additive visual Gaussian noise"),
    "d6_vnoise002": DegradationSpec("d6_vnoise002", "Visual noise 0.02", True, False, "strong visual Gaussian noise"),
    "d7_mp3_128k": DegradationSpec("d7_mp3_128k", "MP3 128k", False, True, "MP3 audio compression"),
    "d8_mp3_64k": DegradationSpec("d8_mp3_64k", "MP3 64k", False, True, "low bitrate MP3 audio compression"),
    "d9_anoise_30db": DegradationSpec("d9_anoise_30db", "Audio noise 30dB", False, True, "audio noise at 30dB SNR"),
    "d10_anoise_20db": DegradationSpec("d10_anoise_20db", "Audio noise 20dB", False, True, "audio noise at 20dB SNR"),
    "d11_h264_crf28": DegradationSpec("d11_h264_crf28", "H.264 CRF=28", True, True, "full video re-encoding"),
    "d12_social": DegradationSpec("d12_social", "Social sim", True, True, "JPEG75 + resize0.75 + MP3 128k"),
}


def jpeg_frame(frame: np.ndarray, quality: int) -> np.ndarray:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    w, h = image.size
    down = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    up = down.resize((w, h), Image.BICUBIC)
    return np.asarray(up.convert("RGB"))


def noise_frame(frame: np.ndarray, sigma: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    arr = frame.astype(np.float32) / 255.0
    arr = np.clip(arr + rng.normal(0.0, sigma, size=arr.shape), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def degrade_frames(
    frames: Iterable[np.ndarray],
    condition: Optional[str],
    seed: int = 2026,
) -> List[np.ndarray]:
    if condition in (None, "clean"):
        return [f.astype(np.uint8) for f in frames]
    rng = np.random.default_rng(seed)
    out = []
    for frame in frames:
        if condition == "d1_jpeg75":
            frame = jpeg_frame(frame, 75)
        elif condition == "d2_jpeg50":
            frame = jpeg_frame(frame, 50)
        elif condition == "d3_resize075":
            frame = resize_frame(frame, 0.75)
        elif condition == "d4_resize050":
            frame = resize_frame(frame, 0.50)
        elif condition == "d5_vnoise001":
            frame = noise_frame(frame, 0.01, rng)
        elif condition == "d6_vnoise002":
            frame = noise_frame(frame, 0.02, rng)
        elif condition == "d12_social":
            frame = resize_frame(jpeg_frame(frame, 75), 0.75)
        elif condition == "d11_h264_crf28":
            pass
        else:
            raise ValueError(f"Condition {condition} is not a visual frame degradation")
        out.append(frame.astype(np.uint8))
    return out


def add_audio_noise_snr(
    waveform: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    wave = waveform.astype(np.float32)
    signal_power = float(np.mean(wave**2))
    if signal_power <= 1e-12:
        return wave
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, math.sqrt(noise_power), size=wave.shape).astype(np.float32)
    return np.clip(wave + noise, -1.0, 1.0)


def mp3_roundtrip(waveform: np.ndarray, sample_rate: int, bitrate: str) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("soundfile is required for MP3 degradations") from exc
    if not ffmpeg_available():
        return waveform.astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "input.wav"
        mp3_path = Path(tmp) / "compressed.mp3"
        out_path = Path(tmp) / "output.wav"
        sf.write(wav_path, waveform, sample_rate)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(wav_path),
                "-b:a",
                bitrate,
                "-threads",
                "1",
                str(mp3_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=60,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(mp3_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-threads",
                "1",
                str(out_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=60,
        )
        decoded, sr = sf.read(out_path, dtype="float32")
        if decoded.ndim > 1:
            decoded = decoded.mean(axis=1)
        if sr != sample_rate:
            import librosa

            decoded = librosa.resample(decoded, orig_sr=sr, target_sr=sample_rate)
        return decoded.astype(np.float32)


def aac_roundtrip(waveform: np.ndarray, sample_rate: int, bitrate: str = "128k") -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("soundfile is required for AAC roundtrip degradations") from exc
    if not ffmpeg_available():
        return waveform.astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "input.wav"
        m4a_path = Path(tmp) / "compressed.m4a"
        out_path = Path(tmp) / "output.wav"
        sf.write(wav_path, waveform, sample_rate)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(wav_path),
                "-c:a",
                "aac",
                "-b:a",
                bitrate,
                "-threads",
                "1",
                str(m4a_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=60,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(m4a_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-threads",
                "1",
                str(out_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=60,
        )
        decoded, sr = sf.read(out_path, dtype="float32")
        if decoded.ndim > 1:
            decoded = decoded.mean(axis=1)
        if sr != sample_rate:
            import librosa

            decoded = librosa.resample(decoded, orig_sr=sr, target_sr=sample_rate)
        return decoded.astype(np.float32)


def degrade_audio(
    waveform: np.ndarray,
    sample_rate: int,
    condition: Optional[str],
    seed: int = 2026,
) -> np.ndarray:
    if condition in (None, "clean"):
        return waveform.astype(np.float32)
    rng = np.random.default_rng(seed)
    if condition == "d7_mp3_128k":
        return mp3_roundtrip(waveform, sample_rate, "128k")
    if condition == "d8_mp3_64k":
        return mp3_roundtrip(waveform, sample_rate, "64k")
    if condition == "d9_anoise_30db":
        return add_audio_noise_snr(waveform, 30.0, rng)
    if condition == "d10_anoise_20db":
        return add_audio_noise_snr(waveform, 20.0, rng)
    if condition == "d12_social":
        return mp3_roundtrip(waveform, sample_rate, "128k")
    if condition == "d11_h264_crf28":
        return aac_roundtrip(waveform, sample_rate, "128k")
    raise ValueError(f"Condition {condition} is not an audio degradation")


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def h264_roundtrip_frames(
    frames: Iterable[np.ndarray],
    crf: int = 28,
    fps: int = 8,
    timeout_seconds: float = 60.0,
) -> List[np.ndarray]:
    """Compress only the sampled frames, not the full source video.

    Full-video re-encoding was the crash-prone path in earlier Kaggle runs.
    This keeps the degradation faithful enough for feature extraction while
    bounding ffmpeg memory, runtime, and temporary disk use.
    """

    frames = [frame.astype(np.uint8) for frame in frames]
    if not ffmpeg_available():
        return frames
    try:
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for index, frame in enumerate(frames):
                Image.fromarray(frame, mode="RGB").save(tmp_dir / f"frame_{index:05d}.png")
            encoded = tmp_dir / "h264.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(tmp_dir / "frame_%05d.png"),
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(crf),
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "1",
                    str(encoded),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=timeout_seconds,
            )
            capture = cv2.VideoCapture(str(encoded))
            decoded: List[np.ndarray] = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            finally:
                capture.release()
        if not decoded:
            return frames
        if len(decoded) > len(frames):
            indices = np.linspace(0, len(decoded) - 1, len(frames)).round().astype(int).tolist()
            decoded = [decoded[index] for index in indices]
        while len(decoded) < len(frames):
            decoded.append(decoded[-1].copy())
        return decoded[: len(frames)]
    except Exception:
        return frames


def h264_reencode(video_path: str | Path, crf: int = 28, tmp_dir: Optional[str | Path] = None) -> Path:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required for H.264 re-encoding but was not found on PATH")
    tmp_root = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="cmar_h264_"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    out_path = tmp_root / f"{Path(video_path).stem}_crf{crf}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def visual_conditions() -> List[str]:
    return [name for name, spec in DEGRADATION_SPECS.items() if spec.visual]


def audio_conditions() -> List[str]:
    return [name for name, spec in DEGRADATION_SPECS.items() if spec.audio]
