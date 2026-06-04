# Preprocessing and Cache Contract

The preprocessing script is the first real compute step. It prevents every later training or evaluation run from re-decoding all videos and re-running foundation encoders.

## Current Completed Cache

Preprocessing has been completed and uploaded to Kaggle. The Kaggle dataset shown
as **CMAR Clean Features V1** contains the final `cmar_cache/` folder, including
clean train/val/test features and all 12 degraded test conditions.

Use this cache directly for training and evaluation:

```text
/kaggle/input/cmar-features-clean-v1/cmar_cache
```

The current cache is complete for FakeAVCeleb clean train/val/test and
FakeAVCeleb degraded test. It does not include LAV-DF feature tensors, so pass
`--skip-lavdf` to `scripts/03_evaluate_clean_degraded.py` unless you later build
a separate LAV-DF feature cache.

If Kaggle mounts the dataset under a different slug, locate it with:

```bash
find /kaggle/input -maxdepth 5 -type d -name cmar_cache
```

The commands below are retained only for rebuilding or repairing the cache.

## Rebuild Command

```bash
python scripts/01_preprocess_features.py \
  --config configs/preprocess_fakeavceleb.json \
  --dataset-root /kaggle/input/fakeavceleb/FakeAVCeleb_v1.2 \
  --lavdf-root /kaggle/input/lav-df/LAV-DF \
  --output-dir /kaggle/working/cmar_cache
```

## Output

```text
cmar_cache/
  features/
    visual/
      train/{clip_id}.pt
      val/{clip_id}.pt
      test/{clip_id}.pt
    audio/
      train/{clip_id}.pt
      val/{clip_id}.pt
      test/{clip_id}.pt
    degraded_test/
      d1_jpeg75/visual/{clip_id}.pt
      d2_jpeg50/visual/{clip_id}.pt
      d3_resize075/visual/{clip_id}.pt
      d4_resize050/visual/{clip_id}.pt
      d5_vnoise001/visual/{clip_id}.pt
      d6_vnoise002/visual/{clip_id}.pt
      d7_mp3_128k/audio/{clip_id}.pt
      d8_mp3_64k/audio/{clip_id}.pt
      d9_anoise_30db/audio/{clip_id}.pt
      d10_anoise_20db/audio/{clip_id}.pt
      d11_h264_crf28/visual/{clip_id}.pt
      d11_h264_crf28/audio/{clip_id}.pt
      d12_social/visual/{clip_id}.pt
      d12_social/audio/{clip_id}.pt
  manifests/
    train.csv
    val.csv
    test.csv
    lavdf_test.csv
  metadata.json
```

## Feature Shapes

- Visual: `(16, 384)` DINOv2-Small CLS features, one vector per sampled frame.
- Audio: `(<=64, 384)` pooled Whisper-Tiny encoder temporal features by default. Raw Whisper output is much longer, so pooling during preprocessing keeps the cache and later training batches small.
- Stored dtype: `float16` by default to fit Kaggle dataset limits.

The visual extractor explicitly builds DINOv2 with `img_size=224`. This matters
because some `timm` DINOv2 pretrained configs default to 518px and will assert
when fed the 224px FakeAVCeleb frames used by CMAR.

## Training Behavior

Training uses `CachedAVDataset`, which loads these `.pt` files directly. The consistency loss uses cheap feature-space augmentations during training, while the degraded test conditions are generated from raw media before feature extraction.

That distinction is intentional:

- Raw degradations are used for evaluation fidelity.
- Feature-space augmentation is used for fast repeated training.
- Backbones are not re-run during normal training.

## Audio Decode Notes

The preprocessing loader decodes audio from `.mp4` with `ffmpeg` first. This is
intentional: `soundfile` often cannot read audio streams embedded in MP4 files,
and `librosa` then falls back to its slower deprecated `audioread` path. If
`ffmpeg` is unavailable or a clip has a bad audio stream, the code falls back to
`librosa` and pads with zeros only as a last resort.

## Crash-Safe Kaggle Mode

Kaggle backend error 137 usually means the container was killed by the OS. Use
small resumable slices instead of one huge preprocessing run:

```bash
python scripts/01_preprocess_features.py \
  --config configs/preprocess_fakeavceleb.json \
  --dataset-root /kaggle/input/datasets/shreyaty08/fakeavceleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2 \
  --lavdf-root /kaggle/input/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF \
  --output-dir /kaggle/working/cmar_cache \
  --no-degraded \
  --max-new-rows 200 \
  --max-runtime-seconds 900 \
  --chunk-size 50
```

Rerun the same command until clean train/val/test features are complete. The
script skips existing `.pt` files, so each run resumes.

Before training, verify coverage:

```bash
python scripts/02_train_cmar.py \
  --config configs/train_cmar.json \
  --cache-dir /kaggle/working/cmar_cache \
  --cache-report-only
```

Then generate degraded test features condition by condition:

```bash
python scripts/01_preprocess_features.py \
  --config configs/preprocess_fakeavceleb.json \
  --dataset-root /kaggle/input/datasets/shreyaty08/fakeavceleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2 \
  --lavdf-root /kaggle/input/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF \
  --output-dir /kaggle/working/cmar_cache \
  --degraded-only \
  --conditions d1_jpeg75 \
  --max-new-rows 200 \
  --max-runtime-seconds 900 \
  --chunk-size 50
```

Repeat with the remaining conditions. H.264 is implemented as sampled-frame
roundtrip compression rather than full-video re-encoding to avoid the previous
ffmpeg crash pattern.

Clean features are sufficient for model training. Degraded features are only
needed for robustness evaluation, degraded-condition metrics, and TTDA-style
test-time ensembles.

Historical Colab helpers were moved out of `docs/` to keep the paper-facing
documentation readable. If the cache must be rebuilt through Colab, see
`notebooks/COLAB_UPLOAD_CLEAN_CACHE.py` and
`notebooks/colab_degrade_preprocess_worker.py`.
