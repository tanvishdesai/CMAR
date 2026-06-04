from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from av_robustbench.degradations.specs import DegradationSpec, get_degradation_spec
from av_robustbench.utils.io import ensure_dir


def apply_degradation(
    video_path: str | Path | None,
    audio_path: str | Path | None,
    spec: str | DegradationSpec,
    output_dir: str | Path,
) -> tuple[Path | None, Path | None]:
    """Apply a file-level degradation and return degraded media paths.

    Frame-level JPEG/resize/noise conditions are supported for videos through
    OpenCV frame decoding/writing. Codec conditions use ffmpeg.
    """

    spec = get_degradation_spec(spec)
    output_dir = ensure_dir(output_dir)
    out_video = None
    out_audio = None
    if spec.visual and video_path is not None:
        out_video = output_dir / f"{Path(video_path).stem}_{spec.name}.mp4"
        if spec.name in {"d11_h264_crf28", "d12_social"}:
            _ffmpeg_video_chain(Path(video_path), out_video, spec)
        else:
            _opencv_video_frame_chain(Path(video_path), out_video, spec.name)
    if spec.audio and audio_path is not None:
        out_audio = output_dir / f"{Path(audio_path).stem}_{spec.name}.wav"
        if spec.name in {"d11_h264_crf28", "d12_social", "d7_mp3_128k", "d8_mp3_64k"}:
            _ffmpeg_audio_chain(Path(audio_path), out_audio, spec)
        else:
            _soundfile_audio_chain(Path(audio_path), out_audio, spec.name)
    return out_video, out_audio


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for this degradation but was not found on PATH.")


def _ffmpeg_video_chain(input_path: Path, output_path: Path, spec: DegradationSpec) -> None:
    _require_ffmpeg()
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(input_path)]
    if spec.name == "d12_social":
        cmd.extend(["-vf", f"scale=iw*{spec.params['scale']}:ih*{spec.params['scale']}"])
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-crf",
            str(spec.params.get("crf", 28)),
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-b:a",
            str(spec.params.get("audio_bitrate", "128k")),
            str(output_path),
        ]
    )
    subprocess.run(cmd, check=True, timeout=300)


def _ffmpeg_audio_chain(input_path: Path, output_path: Path, spec: DegradationSpec) -> None:
    _require_ffmpeg()
    bitrate = str(spec.params.get("bitrate", spec.params.get("audio_bitrate", "128k")))
    codec = "libmp3lame" if spec.name in {"d7_mp3_128k", "d8_mp3_64k"} else "aac"
    temp_encoded = output_path.with_suffix(".mp3" if codec == "libmp3lame" else ".m4a")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-ac",
            "1",
            "-c:a",
            codec,
            "-b:a",
            bitrate,
            str(temp_encoded),
        ],
        check=True,
        timeout=120,
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
            str(temp_encoded),
            "-ac",
            "1",
            str(output_path),
        ],
        check=True,
        timeout=120,
    )
    temp_encoded.unlink(missing_ok=True)


def _opencv_video_frame_chain(input_path: Path, output_path: Path, condition: str) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for file-level visual degradations.") from exc
    from av_robustbench.degradations.visual import degrade_frames

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            degraded = degrade_frames([frame_rgb], condition, seed=2026 + index)[0]
            writer.write(cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))
            index += 1
    finally:
        cap.release()
        writer.release()


def _soundfile_audio_chain(input_path: Path, output_path: Path, condition: str) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError("soundfile is required for file-level audio degradations.") from exc
    from av_robustbench.degradations.audio import degrade_audio

    waveform, sample_rate = sf.read(input_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    degraded = degrade_audio(waveform, int(sample_rate), condition)
    sf.write(output_path, degraded, int(sample_rate))

