# Dataset Layout Notes

These notes are based on the screenshots in `dataset info/`.

## FakeAVCeleb

Observed Kaggle structure:

```text
FakeAVCeleb_v1.2/
  FakeAVCeleb_v1.2/
    FakeVideo-FakeAudio/
    FakeVideo-RealAudio/
    RealVideo-FakeAudio/
    RealVideo-RealAudio/
    README.txt
    meta_data.csv
  frames/
    FakeVideo-FakeAudio/
    FakeVideo-RealAudio/
      African/
      Asian (East)/
        men/
        women/
      Asian (South)/
        men/
          id00032/
          id00033/
          id00078/
            *.mp4 or frame files
      Caucasian (...)
    RealVideo-FakeAudio/
    RealVideo-RealAudio/
```

The implementation scans for the deepest directory containing all four AV category folders:

- `RealVideo-RealAudio` -> label `0`, AV category `RR`
- `FakeVideo-RealAudio` -> label `1`, AV category `FR`
- `RealVideo-FakeAudio` -> label `1`, AV category `RF`
- `FakeVideo-FakeAudio` -> label `1`, AV category `FF`

It ignores paths containing `frames` or `moved` when scanning the raw video category folders.

## LAV-DF

Observed Kaggle structure:

```text
LAV-DF/
  dev/
  test/
  train/
  README.md
  metadata.json
  metadata.min.json
```

`metadata.min.json` preview shows rows like:

```json
{
  "file": "test/000000.mp4",
  "n_fakes": 0,
  "fake_periods": [],
  "duration": 4.672,
  "original": null,
  "modify_video": false,
  "modify_audio": false,
  "split": "test",
  "video_frames": 115,
  "audio_channels": 1,
  "audio_frames": 72704
}
```

The implementation builds `manifests/lavdf_test.csv` from `metadata.min.json` or `metadata.json` when the LAV-DF root is available.

## Completed CMAR Feature Cache

The completed Kaggle cache dataset is shown as **CMAR Clean Features V1**. The
name is historical: this dataset now contains both clean features and all
degraded-test condition features.

Expected mounted structure:

```text
cmar_cache/
  features/
    audio/
      train/
      val/
      test/
    visual/
      train/
      val/
      test/
    degraded_test/
      d1_jpeg75/
      d2_jpeg50/
      d3_resize075/
      d4_resize050/
      d5_vnoise001/
      d6_vnoise002/
      d7_mp3_128k/
      d8_mp3_64k/
      d9_anoise_30db/
      d10_anoise_20db/
      d11_h264_crf28/
      d12_social/
  manifests/
    train.csv
    val.csv
    test.csv
  colab_cache_report.json
  degraded_cache_report.json
  metadata.json
  preprocess_auto_status.json
```

Use `/kaggle/input/cmar-features-clean-v1/cmar_cache` when Kaggle mounts the
dataset by its current slug. If the folder name differs, run
`find /kaggle/input -maxdepth 5 -type d -name cmar_cache` and use the returned
path as `--cache-dir`.
