# Publishing av-robustbench

## Local Verification

```bash
cd av-robustbench
python -m pytest
python -m ruff check src tests
python -m pip wheel . --no-deps -w %TEMP%\avrobustbench-wheel-check
av-robustbench list-models
```

## Register Real Checkpoints

The package ships real model-loading code and built-in adapter names for the CertAV/CMAR entries. It does not fabricate checkpoint URLs. Before publishing a leaderboard entry, upload the trained weights to HuggingFace Hub or pass local checkpoint paths:

```python
from av_robustbench.models import register_model
from av_robustbench.models.adapters import CertAVAdapter

register_model(
    "certav_sigma100",
    CertAVAdapter,
    {
        "dataset": "fakeavceleb",
        "training_sigma": 1.0,
        "hf_repo_id": "your-org/certav-sigma100",
        "hf_filename": "best.pt",
    },
    overwrite=True,
)
```

## Build and Upload

```bash
python -m pip install build twine
python -m build
twine check dist/*
twine upload dist/*
```

## Expected Artifact Contract

Each public evaluation should include:

- `robustness_card.json`
- `robustness_card.md`
- Attack settings and sigma values
- Dataset split and cache metadata
- Checkpoint URL or reproducible checkpoint hash

