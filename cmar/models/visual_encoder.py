from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from torch import nn
from PIL import Image


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


class HFVisionFeatureExtractor(nn.Module):
    """Generic Hugging Face vision encoder wrapper for encoder-family studies."""

    def __init__(self, model_name: str, *, projection: bool = False) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        if projection and "clip" in model_name.lower():
            from transformers import CLIPVisionModelWithProjection

            self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def extract_frames(
        self,
        frames: List[np.ndarray],
        device: torch.device,
        image_size: int = 224,
    ) -> torch.Tensor:
        images = [
            Image.fromarray(frame.astype(np.uint8), mode="RGB").resize((image_size, image_size))
            for frame in frames
        ]
        inputs = self.processor(images=images, return_tensors="pt")
        # Move to device; use float16 for floating-point tensors to cut
        # memory and compute in half on CUDA.
        inputs = {
            k: v.to(device=device, dtype=torch.float16 if v.is_floating_point() else None)
            for k, v in inputs.items()
        }
        use_amp = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = self.model(**inputs)
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds.float().detach().cpu()
        hidden = outputs.last_hidden_state
        if hidden.shape[1] > 1:
            return hidden[:, 0].float().detach().cpu()
        return hidden.mean(dim=1).float().detach().cpu()


def build_visual_feature_extractor(
    model_name: str = "facebook/dinov2-small",
    *,
    image_size: int = 224,
    pretrained: bool = True,
) -> nn.Module:
    """Build a visual encoder by public model name.

    The default path preserves the original timm DINOv2-Small cache contract.
    Other models are used for the Phase 2 encoder-family scaling-law study.
    """

    # Models with known timm equivalents use the fast DINOv2FeatureExtractor
    # path (timm forward_features + CLS-token extraction).  This avoids the
    # HuggingFace AutoImageProcessor overhead which is the main bottleneck.
    aliases = {
        "facebook/dinov2-small": "vit_small_patch14_dinov2.lvd142m",
        "dinov2-small": "vit_small_patch14_dinov2.lvd142m",
        "facebook/dinov2-base": "vit_base_patch14_dinov2.lvd142m",
        "dinov2-base": "vit_base_patch14_dinov2.lvd142m",
        # CLIP ViT-B/16 routed through timm for ~5-10× faster extraction
        # compared to the HF CLIPVisionModelWithProjection path.
        "openai/clip-vit-base-patch16": "vit_base_patch16_clip_224.openai",
        "clip-vit-base-patch16": "vit_base_patch16_clip_224.openai",
    }
    if model_name in aliases or model_name.startswith("vit_"):
        return DINOv2FeatureExtractor(
            model_name=aliases.get(model_name, model_name),
            pretrained=pretrained,
            tune_layernorm=False,
            image_size=image_size,
        )
    if "clip" in model_name.lower():
        return HFVisionFeatureExtractor(model_name, projection=True)
    return HFVisionFeatureExtractor(model_name, projection=False)
