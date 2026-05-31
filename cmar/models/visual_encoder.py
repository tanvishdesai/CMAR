from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class DINOv2FeatureExtractor(nn.Module):
    """DINOv2-Small wrapper for cache generation and optional raw experiments."""

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        tune_layernorm: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        import timm

        self.model_name = model_name
        self.image_size = image_size
        self.model = self._create_model(
            timm=timm,
            model_name=model_name,
            pretrained=pretrained,
            image_size=image_size,
        )
        self.freeze_except_layernorm(tune_layernorm=tune_layernorm)

    @staticmethod
    def _create_model(timm, model_name: str, pretrained: bool, image_size: int) -> nn.Module:
        """Create DINOv2 with the project's 224px input contract.

        Recent `timm` DINOv2 pretrained configs default to 518px inputs. If we
        do not override this, `PatchEmbed` asserts on the 224px frames used by
        FakeAVCeleb and by the original CMAR plan. `img_size=224` makes timm
        resize the positional embedding while loading the pretrained weights;
        `dynamic_img_size=True` relaxes strict shape checks for compatible
        timm versions.
        """

        kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "img_size": image_size,
            "dynamic_img_size": True,
        }
        try:
            return timm.create_model(model_name, **kwargs)
        except TypeError:
            kwargs.pop("dynamic_img_size")
            return timm.create_model(model_name, **kwargs)

    def freeze_except_layernorm(self, tune_layernorm: bool = False) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        if tune_layernorm:
            for module in self.model.modules():
                if isinstance(module, nn.LayerNorm):
                    for param in module.parameters():
                        param.requires_grad = True

    def layernorm_parameters(self) -> Iterable[nn.Parameter]:
        for module in self.model.modules():
            if isinstance(module, nn.LayerNorm):
                yield from module.parameters()

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        output = self.model.forward_features(frames)
        if isinstance(output, dict):
            if "x_norm_clstoken" in output:
                return output["x_norm_clstoken"]
            if "pooled" in output:
                return output["pooled"]
            if "x" in output:
                output = output["x"]
        if output.ndim == 3:
            return output[:, 0]
        if output.ndim == 2:
            return output
        raise RuntimeError(f"Unexpected DINOv2 output shape: {tuple(output.shape)}")
