from __future__ import annotations

import torch

from av_robustbench.models.adapters import TorchFeatureDetector


class ToyLinearDetector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(2, 1)

    def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        del audio
        return self.head(visual.reshape(visual.shape[0], -1))


def test_torch_feature_detector_normalizes_plain_tensor_output() -> None:
    detector = TorchFeatureDetector(ToyLinearDetector(), name="toy", feature_dims={"visual": (2,)})
    visual = torch.ones(3, 2)
    audio = torch.zeros(3, 1)
    output = detector.predict(visual, audio)
    assert detector.name == "toy"
    assert detector.input_type == "features"
    assert output["logits"].shape == (3, 1)
    assert output["probs"].shape == (3,)


def test_torch_feature_detector_loads_raw_state_dict(tmp_path) -> None:
    source = ToyLinearDetector()
    with torch.no_grad():
        source.head.weight.fill_(0.5)
        source.head.bias.fill_(0.25)
    checkpoint = tmp_path / "toy.pt"
    torch.save(source.state_dict(), checkpoint)

    loaded = TorchFeatureDetector.from_checkpoint(
        checkpoint,
        module=ToyLinearDetector(),
        name="toy_loaded",
    )
    visual = torch.ones(1, 2)
    audio = torch.zeros(1, 1)
    assert loaded.name == "toy_loaded"
    assert torch.allclose(loaded.predict(visual, audio)["logits"], torch.tensor([[1.25]]))
