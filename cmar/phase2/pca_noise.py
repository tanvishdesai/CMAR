"""PCA-guided anisotropic noise for Phase 2 CertAV experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import gammaln
from scipy.stats import norm


ANISOTROPIC_NOISE_MODES = {
    "anisotropic_strat1": "eigenvalue_proportional",
    "anisotropic_strat2": "subspace_projection",
    "anisotropic_strat3": "inverse_eigenvalue",
}


@dataclass
class PCAArtifact:
    """In-memory representation of a PCA basis saved by Phase 2 scripts."""

    feature_space: str
    mean: torch.Tensor
    components: torch.Tensor
    eigenvalues: torch.Tensor
    explained_variance_ratio: torch.Tensor
    dim_at_80pct: int
    dim_at_90pct: int
    dim_at_95pct: int
    visual_dim: int
    audio_dim: int
    n_samples: int
    source_path: str | None = None

    @property
    def dim(self) -> int:
        return int(self.mean.numel())


def _as_float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def load_pca_artifact(path: str | Path, device: torch.device | str = "cpu") -> PCAArtifact:
    """Load a PCA artifact saved by ``scripts/20_fit_pca_noise.py``."""

    device = torch.device(device)
    raw = torch.load(Path(path), map_location=device, weights_only=False)
    return PCAArtifact(
        feature_space=str(raw.get("feature_space", "joint")),
        mean=_as_float_tensor(raw["mean"], device),
        components=_as_float_tensor(raw["components"], device),
        eigenvalues=_as_float_tensor(raw["eigenvalues"], device),
        explained_variance_ratio=_as_float_tensor(raw["explained_variance_ratio"], device),
        dim_at_80pct=int(raw.get("dim_at_80pct", raw.get("top_k", 0))),
        dim_at_90pct=int(raw.get("dim_at_90pct", raw.get("top_k", 0))),
        dim_at_95pct=int(raw.get("dim_at_95pct", raw.get("top_k", 0))),
        visual_dim=int(raw.get("visual_dim", 0)),
        audio_dim=int(raw.get("audio_dim", 0)),
        n_samples=int(raw.get("n_samples", 0)),
        source_path=str(path),
    )


class PCANoise:
    """Sampler and certificate geometry for PCA-aligned Gaussian smoothing.

    The PCA basis is fitted on pooled training features. For joint PCA, one
    joint noise vector is sampled per clip and split into visual/audio parts;
    the same vector is broadcast over the temporal tokens. This keeps training
    and certification matched to the pooled-feature certificate described in
    the Phase 2 guide.
    """

    def __init__(
        self,
        artifact: PCAArtifact,
        sigma: float,
        strategy: str,
        *,
        top_k: int | None = None,
        off_sigma: float = 1e-3,
        equalize_budget: bool = True,
        eps: float = 1e-8,
    ) -> None:
        if strategy in ANISOTROPIC_NOISE_MODES:
            strategy = ANISOTROPIC_NOISE_MODES[strategy]
        valid = set(ANISOTROPIC_NOISE_MODES.values())
        if strategy not in valid:
            raise ValueError(f"Unknown PCA noise strategy: {strategy}. Expected one of {sorted(valid)}")
        if artifact.feature_space not in {"joint", "visual", "audio"}:
            raise ValueError("feature_space must be 'joint', 'visual', or 'audio'")

        self.artifact = artifact
        self.sigma = float(sigma)
        self.strategy = strategy
        self.top_k = int(top_k or artifact.dim_at_90pct or artifact.dim)
        self.top_k = max(1, min(self.top_k, artifact.dim))
        self.off_sigma = float(off_sigma)
        self.equalize_budget = bool(equalize_budget)
        self.eps = float(eps)
        self.device = artifact.mean.device

        if artifact.components.shape[0] < artifact.dim or artifact.components.shape[1] != artifact.dim:
            raise ValueError(
                "PCA artifact must contain a full square component matrix. "
                f"Got components={tuple(artifact.components.shape)}, dim={artifact.dim}."
            )
        self._variances = self._build_variances()
        self._std = torch.sqrt(torch.clamp(self._variances, min=self.eps))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        sigma: float,
        strategy: str,
        *,
        device: torch.device | str = "cpu",
        top_k: int | None = None,
        off_sigma: float = 1e-3,
        equalize_budget: bool = True,
    ) -> "PCANoise":
        artifact = load_pca_artifact(path, device=device)
        return cls(
            artifact,
            sigma=sigma,
            strategy=strategy,
            top_k=top_k,
            off_sigma=off_sigma,
            equalize_budget=equalize_budget,
        )

    @property
    def dim(self) -> int:
        return self.artifact.dim

    @property
    def variances(self) -> torch.Tensor:
        return self._variances

    def to(self, device: torch.device | str) -> "PCANoise":
        return PCANoise(
            load_pca_artifact(self.artifact.source_path, device=device)
            if self.artifact.source_path
            else PCAArtifact(
                feature_space=self.artifact.feature_space,
                mean=self.artifact.mean.to(device),
                components=self.artifact.components.to(device),
                eigenvalues=self.artifact.eigenvalues.to(device),
                explained_variance_ratio=self.artifact.explained_variance_ratio.to(device),
                dim_at_80pct=self.artifact.dim_at_80pct,
                dim_at_90pct=self.artifact.dim_at_90pct,
                dim_at_95pct=self.artifact.dim_at_95pct,
                visual_dim=self.artifact.visual_dim,
                audio_dim=self.artifact.audio_dim,
                n_samples=self.artifact.n_samples,
                source_path=None,
            ),
            sigma=self.sigma,
            strategy=self.strategy,
            top_k=self.top_k,
            off_sigma=self.off_sigma,
            equalize_budget=self.equalize_budget,
            eps=self.eps,
        )

    def _build_variances(self) -> torch.Tensor:
        eig = torch.clamp(self.artifact.eigenvalues.float(), min=self.eps)
        dim = int(eig.numel())
        total_budget = dim * (self.sigma ** 2)

        if self.strategy == "eigenvalue_proportional":
            raw = eig
        elif self.strategy == "subspace_projection":
            raw = torch.full_like(eig, self.off_sigma ** 2)
            if self.equalize_budget:
                off_budget = float(raw[self.top_k :].sum().detach().cpu())
                on_var = max(self.eps, (total_budget - off_budget) / self.top_k)
                raw[: self.top_k] = on_var
                return raw
            raw[: self.top_k] = self.sigma ** 2
            return raw
        elif self.strategy == "inverse_eigenvalue":
            raw = torch.full_like(eig, self.off_sigma ** 2)
            raw[: self.top_k] = 1.0 / eig[: self.top_k]
        else:  # pragma: no cover - guarded in __init__.
            raise ValueError(self.strategy)

        raw_sum = torch.clamp(raw.sum(), min=self.eps)
        return raw * (total_budget / raw_sum)

    def _sample_vectors(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        z = torch.randn(batch_size, self.dim, device=self.device, dtype=torch.float32)
        coeffs = z * self._std.unsqueeze(0)
        noise = coeffs @ self.artifact.components.float()
        return noise.to(dtype=dtype)

    def add_noise(self, visual: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add PCA noise to a visual/audio feature batch."""

        if visual.device != self.device:
            moved = self.to(visual.device)
            return moved.add_noise(visual, audio)

        batch_size = int(visual.shape[0])
        noise = self._sample_vectors(batch_size, dtype=visual.dtype)
        space = self.artifact.feature_space

        if space == "joint":
            visual_dim = self.artifact.visual_dim
            audio_dim = self.artifact.audio_dim
            if visual_dim <= 0 or audio_dim <= 0:
                raise ValueError("Joint PCA artifact must store visual_dim and audio_dim.")
            if visual.shape[-1] != visual_dim or audio.shape[-1] != audio_dim:
                raise ValueError(
                    "Feature dimensions do not match PCA artifact: "
                    f"visual {visual.shape[-1]} vs {visual_dim}, audio {audio.shape[-1]} vs {audio_dim}."
                )
            visual_noise = noise[:, :visual_dim].unsqueeze(1)
            audio_noise = noise[:, visual_dim : visual_dim + audio_dim].unsqueeze(1)
            return visual + visual_noise, audio + audio_noise

        if space == "visual":
            if visual.shape[-1] != self.dim:
                raise ValueError(f"Visual dimension {visual.shape[-1]} does not match PCA dim {self.dim}.")
            return visual + noise.unsqueeze(1), audio

        if audio.shape[-1] != self.dim:
            raise ValueError(f"Audio dimension {audio.shape[-1]} does not match PCA dim {self.dim}.")
        return visual, audio + noise.unsqueeze(1)

    def certificate_metrics(self, pA_lower: float) -> dict[str, Any]:
        """Return ellipsoid-derived certificate diagnostics for a probability bound."""

        if pA_lower <= 0.5:
            return {
                "certified_radius_l2": 0.0,
                "certified_radius_onmanifold": 0.0,
                "certified_ellipsoid_log_volume": None,
                "certified_ellipsoid_volume": None,
                "pca_top_k": self.top_k,
                "anisotropic_strategy": self.strategy,
            }

        z = float(norm.ppf(pA_lower))
        variances_np = self._variances.detach().cpu().numpy().astype(float)
        min_var = float(np.min(variances_np))
        top_var = float(np.mean(variances_np[: self.top_k]))
        radius_l2 = z * math.sqrt(max(min_var, self.eps))
        radius_on = z * math.sqrt(max(top_var, self.eps))

        dim = self.dim
        log_unit_ball = (dim / 2.0) * math.log(math.pi) - float(gammaln(dim / 2.0 + 1.0))
        log_det_sqrt = 0.5 * float(np.sum(np.log(np.clip(variances_np, self.eps, None))))
        log_volume = log_unit_ball + dim * math.log(max(z, self.eps)) + log_det_sqrt
        volume = math.exp(log_volume) if log_volume < 700 else None

        return {
            "certified_radius_l2": float(radius_l2),
            "certified_radius_onmanifold": float(radius_on),
            "certified_ellipsoid_log_volume": float(log_volume),
            "certified_ellipsoid_volume": volume,
            "pca_top_k": self.top_k,
            "anisotropic_strategy": self.strategy,
            "noise_budget_trace": float(np.sum(variances_np)),
            "min_axis_sigma": float(math.sqrt(max(min_var, self.eps))),
            "mean_onmanifold_sigma": float(math.sqrt(max(top_var, self.eps))),
        }

    def metadata(self) -> dict[str, Any]:
        variances_np = self._variances.detach().cpu().numpy().astype(float)
        return {
            "feature_space": self.artifact.feature_space,
            "strategy": self.strategy,
            "sigma": self.sigma,
            "top_k": self.top_k,
            "off_sigma": self.off_sigma,
            "equalize_budget": self.equalize_budget,
            "dim": self.dim,
            "visual_dim": self.artifact.visual_dim,
            "audio_dim": self.artifact.audio_dim,
            "dim_at_80pct": self.artifact.dim_at_80pct,
            "dim_at_90pct": self.artifact.dim_at_90pct,
            "dim_at_95pct": self.artifact.dim_at_95pct,
            "noise_budget_trace": float(np.sum(variances_np)),
            "variance_min": float(np.min(variances_np)),
            "variance_max": float(np.max(variances_np)),
            "variance_mean_top_k": float(np.mean(variances_np[: self.top_k])),
            "source_path": self.artifact.source_path,
        }
