from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch

from av_robustbench.certification.core import (
    CertificationResult,
    certified_radius,
    lower_confidence_bound_exact,
    summarize_certification,
)

NoiseMode = str


class SmoothedAVClassifier:
    """Randomized smoothing wrapper for any audio-visual detector.

    The base detector may be an `AVDetector`, any `torch.nn.Module` returning a
    dict with `logits`, or a callable returning logits directly.
    """

    VALID_NOISE_MODES = {"joint", "visual_only", "audio_only"}

    def __init__(
        self,
        base: Any,
        sigma: float,
        *,
        noise_mode: NoiseMode = "joint",
        n_classes: int = 2,
        device: str | torch.device | None = None,
    ) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if noise_mode not in self.VALID_NOISE_MODES:
            raise ValueError(f"noise_mode must be one of {sorted(self.VALID_NOISE_MODES)}")
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.base = base
        self.sigma = float(sigma)
        self.noise_mode = noise_mode
        self.n_classes = int(n_classes)
        if device is None:
            try:
                first_param = next(base.parameters())
                device = first_param.device
            except Exception:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if hasattr(base, "to"):
            base.to(self.device)
        if hasattr(base, "eval"):
            base.eval()

    def _ensure_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim in {1, 2, 4}:
            return tensor.unsqueeze(0)
        return tensor

    def _add_noise(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.noise_mode in {"joint", "visual_only"}:
            visual = visual + torch.randn(
                visual.shape,
                generator=generator,
                device=visual.device,
                dtype=visual.dtype,
            ) * self.sigma
        if self.noise_mode in {"joint", "audio_only"}:
            audio = audio + torch.randn(
                audio.shape,
                generator=generator,
                device=audio.device,
                dtype=audio.dtype,
            ) * self.sigma
        return visual, audio

    def _call_base(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base, "predict"):
            out = self.base.predict(visual, audio)
        else:
            out = self.base(visual, audio)
        if isinstance(out, dict):
            if "logits" in out:
                logits = out["logits"]
            elif "probs" in out:
                probs = out["probs"]
                if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[-1] == 1):
                    probs = probs.reshape(-1).clamp(1e-7, 1 - 1e-7)
                    logits = torch.logit(probs)
                else:
                    logits = torch.log(probs.clamp_min(1e-12))
            else:
                raise ValueError("Model output dict must contain `logits` or `probs`.")
        else:
            logits = out
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits, device=self.device)
        return logits

    def _predictions_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 0:
            return (logits.reshape(1) >= 0).long()
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[-1] == 1):
            return (logits.reshape(-1) >= 0).long()
        return logits.argmax(dim=-1).long()

    @torch.inference_mode()
    def _sample_counts(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        n_samples: int,
        *,
        batch_size: int = 64,
        seed: int | None = None,
    ) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        visual = self._ensure_batch(visual).to(self.device)
        audio = self._ensure_batch(audio).to(self.device)
        if visual.shape[0] != 1 or audio.shape[0] != 1:
            raise ValueError("SmoothedAVClassifier.certify expects one sample at a time.")
        counts = np.zeros(self.n_classes, dtype=np.int64)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        for start in range(0, n_samples, batch_size):
            size = min(batch_size, n_samples - start)
            visual_batch = visual.expand(size, *visual.shape[1:]).clone()
            audio_batch = audio.expand(size, *audio.shape[1:]).clone()
            visual_noisy, audio_noisy = self._add_noise(visual_batch, audio_batch, generator=generator)
            logits = self._call_base(visual_noisy, audio_noisy)
            preds = self._predictions_from_logits(logits).detach().cpu().numpy().astype(int)
            bincount = np.bincount(preds, minlength=self.n_classes)
            counts[: len(bincount)] += bincount[: self.n_classes]
        return counts

    def predict(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        *,
        n_samples: int = 100,
        batch_size: int = 64,
        seed: int | None = None,
    ) -> int:
        counts = self._sample_counts(visual, audio, n_samples, batch_size=batch_size, seed=seed)
        return int(np.argmax(counts))

    def certify(
        self,
        visual: torch.Tensor,
        audio: torch.Tensor,
        true_label: int | torch.Tensor,
        *,
        n0: int = 100,
        n: int = 1000,
        alpha: float = 0.001,
        batch_size: int = 64,
        sample_id: str | None = None,
        seed: int | None = None,
    ) -> CertificationResult:
        label = int(true_label.item() if isinstance(true_label, torch.Tensor) else true_label)
        counts0 = self._sample_counts(visual, audio, n0, batch_size=batch_size, seed=seed)
        cA = int(np.argmax(counts0))
        cert_seed = None if seed is None else seed + 1
        counts = self._sample_counts(visual, audio, n, batch_size=batch_size, seed=cert_seed)
        nA = int(counts[cA])
        pA_lower = lower_confidence_bound_exact(nA, n, alpha)
        if pA_lower > 0.5:
            radius = certified_radius(self.sigma, pA_lower)
            predicted = cA
            abstained = False
            correct = predicted == label
        else:
            radius = 0.0
            predicted = -1
            abstained = True
            correct = False
        return CertificationResult(
            predicted_class=predicted,
            certified_radius=radius,
            correct=correct,
            pA_lower=pA_lower,
            counts_top=nA,
            counts_total=n,
            counts=counts.tolist(),
            abstained=abstained,
            true_label=label,
            sample_id=sample_id,
            sigma=self.sigma,
            noise_mode=self.noise_mode,
            alpha=alpha,
        )

    def certify_dataset(
        self,
        dataset: Iterable[Any],
        *,
        n0: int = 100,
        n: int = 1000,
        alpha: float = 0.001,
        batch_size: int = 64,
        max_samples: int | None = None,
        progress_callback: Callable[[int, int | None, CertificationResult], None] | None = None,
        seed: int | None = None,
    ) -> list[CertificationResult]:
        results: list[CertificationResult] = []
        total = len(dataset) if hasattr(dataset, "__len__") else None
        if max_samples is not None and total is not None:
            total = min(total, max_samples)
        for index, item in enumerate(dataset):
            if max_samples is not None and index >= max_samples:
                break
            visual, audio, label, sample_id = _unpack_dataset_item(item)
            result = self.certify(
                visual,
                audio,
                label,
                n0=n0,
                n=n,
                alpha=alpha,
                batch_size=batch_size,
                sample_id=sample_id,
                seed=None if seed is None else seed + index * 2,
            )
            results.append(result)
            if progress_callback is not None:
                progress_callback(index + 1, total, result)
        return results


def _unpack_dataset_item(item: Any) -> tuple[torch.Tensor, torch.Tensor, Any, str | None]:
    if isinstance(item, dict):
        return (
            item["visual"],
            item["audio"],
            item["label"],
            str(item.get("clip_id") or item.get("sample_id") or "") or None,
        )
    if isinstance(item, tuple | list) and len(item) >= 3:
        sample_id = str(item[3]) if len(item) >= 4 else None
        return item[0], item[1], item[2], sample_id
    raise TypeError("Dataset items must be dicts or tuples `(visual, audio, label[, sample_id])`.")


def certify_multi_sigma(
    base: Any,
    dataset: Iterable[Any],
    *,
    sigmas: list[float],
    noise_mode: str = "joint",
    n_classes: int = 2,
    device: str | torch.device | None = None,
    n0: int = 100,
    n: int = 1000,
    alpha: float = 0.001,
    batch_size: int = 64,
    max_samples: int | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for sigma in sigmas:
        smoothed = SmoothedAVClassifier(
            base,
            sigma=sigma,
            noise_mode=noise_mode,
            n_classes=n_classes,
            device=device,
        )
        results = smoothed.certify_dataset(
            dataset,
            n0=n0,
            n=n,
            alpha=alpha,
            batch_size=batch_size,
            max_samples=max_samples,
        )
        key = f"sigma_{sigma:g}"
        output[key] = {
            "sigma": sigma,
            "noise_mode": noise_mode,
            "summary": summarize_certification(results),
            "per_sample": [result.to_dict() for result in results],
        }
    return output
