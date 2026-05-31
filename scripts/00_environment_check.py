from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))


def version_or_missing(package: str) -> str:
    try:
        module = __import__(package)
        return getattr(module, "__version__", "installed")
    except Exception:  # noqa: BLE001
        return "missing"


def main() -> None:
    parser = argparse.ArgumentParser(description="CMAR Kaggle environment check")
    parser.add_argument("--load-models", action="store_true", help="Load DINOv2 and Whisper and run dummy forwards.")
    parser.add_argument("--output", default="/kaggle/working/cmar_environment.json")
    args = parser.parse_args()

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "packages": {
            "torchvision": version_or_missing("torchvision"),
            "torchaudio": version_or_missing("torchaudio"),
            "transformers": version_or_missing("transformers"),
            "timm": version_or_missing("timm"),
            "librosa": version_or_missing("librosa"),
            "cv2": version_or_missing("cv2"),
            "sklearn": version_or_missing("sklearn"),
            "pandas": version_or_missing("pandas"),
            "pydub": version_or_missing("pydub"),
            "torchattacks": version_or_missing("torchattacks"),
        },
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)

    if args.load_models:
        from transformers import WhisperFeatureExtractor

        from cmar.models.audio_encoder import WhisperTinyFeatureExtractor
        from cmar.models.visual_encoder import DINOv2FeatureExtractor

        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        visual = DINOv2FeatureExtractor(image_size=224).to(device).eval()
        audio = WhisperTinyFeatureExtractor().to(device).eval()
        processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        frames = torch.randn(8 * 16, 3, visual.image_size, visual.image_size, device=device)
        waveform = torch.zeros(16000 * 10).numpy()
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            v = visual(frames)
            a = audio(inputs)
        report["dummy_visual_shape"] = list(v.shape)
        report["dummy_audio_shape"] = list(a.shape)
        report["dinov2_image_size"] = visual.image_size
        report["dinov2_patch_embed_img_size"] = list(getattr(visual.model.patch_embed, "img_size", ()))
        if device.type == "cuda":
            report["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
