from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from av_robustbench.degradations.specs import get_degradation_spec


def jpeg_frame(frame: np.ndarray, quality: int) -> np.ndarray:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"))


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    width, height = image.size
    down = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BICUBIC)
    return np.asarray(down.resize((width, height), Image.BICUBIC).convert("RGB"))


def noise_frame(frame: np.ndarray, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    image = frame.astype(np.float32) / 255.0
    image = np.clip(image + rng.normal(0.0, sigma, size=image.shape), 0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def degrade_frames(
    frames: Iterable[np.ndarray],
    condition: str,
    *,
    seed: int = 2026,
) -> list[np.ndarray]:
    spec = get_degradation_spec(condition)
    rng = np.random.default_rng(seed)
    out = []
    for frame in frames:
        arr = frame.astype(np.uint8)
        if spec.name in {"d1_jpeg75", "d2_jpeg50"}:
            arr = jpeg_frame(arr, int(spec.params["quality"]))
        elif spec.name in {"d3_resize075", "d4_resize050"}:
            arr = resize_frame(arr, float(spec.params["scale"]))
        elif spec.name in {"d5_vnoise001", "d6_vnoise002"}:
            arr = noise_frame(arr, float(spec.params["sigma"]), rng)
        elif spec.name == "d12_social":
            arr = resize_frame(jpeg_frame(arr, int(spec.params["quality"])), float(spec.params["scale"]))
        elif spec.name == "d11_h264_crf28":
            out.append(arr)
            continue
        elif not spec.visual:
            out.append(arr)
            continue
        else:
            raise ValueError(f"{condition} is not implemented as a frame-level visual degradation.")
        out.append(arr.astype(np.uint8))
    if spec.name == "d11_h264_crf28":
        return h264_roundtrip_frames(out, crf=int(spec.params["crf"]))
    return out


def h264_roundtrip_frames(
    frames: Iterable[np.ndarray],
    *,
    crf: int = 28,
    fps: int = 8,
    timeout_seconds: float = 60.0,
) -> list[np.ndarray]:
    frames = [frame.astype(np.uint8) for frame in frames]
    if not frames or shutil.which("ffmpeg") is None:
        return frames
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for frame H.264 roundtrips.") from exc
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for index, frame in enumerate(frames):
            Image.fromarray(frame, mode="RGB").save(tmp_dir / f"frame_{index:05d}.png")
        encoded = tmp_dir / "encoded.mp4"
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
            check=True,
            timeout=timeout_seconds,
        )
        cap = cv2.VideoCapture(str(encoded))
        decoded: list[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
    if not decoded:
        return frames
    if len(decoded) > len(frames):
        indices = np.linspace(0, len(decoded) - 1, len(frames)).round().astype(int).tolist()
        decoded = [decoded[index] for index in indices]
    while len(decoded) < len(frames):
        decoded.append(decoded[-1].copy())
    return decoded[: len(frames)]

