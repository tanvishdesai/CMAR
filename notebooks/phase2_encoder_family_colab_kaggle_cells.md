# Phase 2 Encoder-Family Colab + Kaggle Cells

This notebook is split out from `phase2_final_kaggle_cells.md` because the
new encoder-family sweep calls `scripts/01_preprocess_features.py`, which can
crash inside Kaggle for heavier visual/audio encoder pairs.

Dependency answer:

- Cells after the old encoder-family preprocessing block do not all require the
  new encoder cache outputs.
- Direction C conformal calibration/evaluation is independent and only uses
  the existing clean cache/checkpoints.
- The final summarizer/display/packaging cells can run without the new encoder
  outputs, but `scripts/23_phase2_summarize.py` will include the encoder-family
  rows only when each pair has `pca_joint.summary.json` and
  `baseline_no_noise_cert.json` under `phase2/encoder_study/<pair>/`.

Use this file for the split encoder-family workflow:

1. Run Part A in Google Colab once per encoder pair. It preprocesses the cache,
   stores progress in Google Drive, zips the completed cache, and uploads it as
   a Kaggle dataset.
2. Attach the uploaded pair-cache datasets to Kaggle and run Part B there. It
   unzips the caches, runs PCA, trains the no-noise baseline, certifies it, and
   writes the encoder scaling summary.

---

## Part A: Colab Encoder Cache Worker

Copy this entire cell into one Google Colab code cell. Edit the config block,
especially `KAGGLE_USERNAME`, `KAGGLE_KEY`, and `PAIR_NAME`.

Recommended parallel pattern:

- Account 1: `PAIR_NAME = "clip_whispertiny"`
- Account 2: `PAIR_NAME = "dinov2small_whisperbase"`

```python
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


# =========================
# CONFIG: EDIT THESE VALUES
# =========================

KAGGLE_USERNAME = "vasuaashadesai"
KAGGLE_KEY = "fab2e63fd07eb970e71e37a33ba8ce83"

FAKEAVCELEB_DATASET = "shreyaty08/fakeavceleb"

# Choose exactly one encoder pair per Colab notebook/account.
PAIR_NAME = "clip_whispertiny"
WORKER_NAME = "account_02"

# Share these folders with every Colab account you use.
DRIVE_CMAR_DIR = "/content/drive/MyDrive/CMAR"
DRIVE_CMAR_ZIP = "/content/drive/MyDrive/CMAR.zip"
DRIVE_ENCODER_ROOT = "/content/drive/MyDrive/cmar_encoder_family"

# Local runtime folders.
CMAR_DIR = "/content/CMAR"
DATA_ROOT = "/content/data"
LOCAL_ENCODER_ROOT = "/content/cmar_encoder_family"

# Tune these down if a pair still runs out of memory.
# These are the DEFAULTS; per-pair overrides below can adjust them.
SLICE_ROWS = 500
SLICE_SECONDS = 1800
CHUNK_SIZE = 25
MAX_TOTAL_SECONDS = 13_000

# Keep False unless you know a stale lock belongs to you.
ALLOW_IF_LOCK_EXISTS = True


ENCODER_PAIRS = {
    "clip_whispertiny": {
        "visual": "openai/clip-vit-base-patch16",
        "audio": "openai/whisper-tiny",
        "dataset_suffix": "cmar-encoder-cache-clip-whispertiny",
        "title": "CMAR Encoder Cache CLIP Whisper Tiny",
        # CLIP now routes through timm (fast path); no special tuning needed.
    },
    "dinov2small_whisperbase": {
        "visual": "facebook/dinov2-small",
        "audio": "openai/whisper-base",
        "dataset_suffix": "cmar-encoder-cache-dinov2small-whisperbase",
        "title": "CMAR Encoder Cache DINOv2 Small Whisper Base",
        # Same fast DINOv2 timm path + slightly larger Whisper.
    },
    "dinov2small_hubert": {
        "visual": "facebook/dinov2-small",
        "audio": "facebook/hubert-base-ls960",
        "dataset_suffix": "cmar-encoder-cache-dinov2small-hubert",
        "title": "CMAR Encoder Cache DINOv2 Small HuBERT",
    },
}


def run(command, check=True, cwd=None, capture=False):
    print("\n[run]", " ".join(map(str, command)), flush=True)
    completed = subprocess.run(
        list(map(str, command)),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture and completed.stdout:
        print(completed.stdout)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(map(str, command))}")
    return completed.returncode


def mount_drive():
    from google.colab import drive
    drive.mount("/content/drive")


def setup_kaggle_auth():
    if KAGGLE_USERNAME == "YOUR_KAGGLE_USERNAME" or KAGGLE_KEY == "YOUR_KAGGLE_KEY":
        raise ValueError("Fill KAGGLE_USERNAME and KAGGLE_KEY first.")
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_path = kaggle_dir / "kaggle.json"
    token_path.write_text(json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}), encoding="utf-8")
    token_path.chmod(0o600)
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "kaggle",
        "timm>=0.9.12",
        "transformers>=4.37",
        "librosa>=0.10",
        "soundfile>=0.12",
        "opencv-python",
        "pandas",
        "tqdm",
        "scikit-learn",
    ])
    run(["kaggle", "--version"], capture=True)


def prepare_cmar_source():
    dst = Path(CMAR_DIR)
    if (dst / "scripts" / "01_preprocess_features.py").exists():
        print("[setup] CMAR source already present:", dst)
        return

    drive_src = Path(DRIVE_CMAR_DIR)
    if (drive_src / "scripts" / "01_preprocess_features.py").exists():
        print("[setup] copying CMAR folder from Drive:", drive_src)
        shutil.copytree(drive_src, dst, dirs_exist_ok=True)
        return

    zip_path = Path(DRIVE_CMAR_ZIP)
    if not zip_path.exists():
        from google.colab import files
        print("[upload] Upload CMAR.zip now.")
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("No CMAR.zip uploaded.")
        zip_path = Path("/content") / next(iter(uploaded.keys()))

    print("[setup] extracting:", zip_path)
    shutil.unpack_archive(str(zip_path), "/content")
    if (dst / "scripts" / "01_preprocess_features.py").exists():
        return

    candidates = list(Path("/content").glob("**/scripts/01_preprocess_features.py"))
    if not candidates:
        raise RuntimeError("Could not find CMAR scripts after extraction.")
    source_root = candidates[0].parents[1]
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(source_root, dst)


def download_dataset(slug, out_dir):
    out_dir = Path(out_dir)
    done = out_dir / ".download_complete"
    if done.exists():
        print("[data] already downloaded:", slug)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", slug, "-p", out_dir, "--unzip"])
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def find_fakeav_root():
    root = Path(DATA_ROOT) / "fakeavceleb"
    categories = {"RealVideo-RealAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"}
    candidates = []
    for candidate in root.rglob("RealVideo-RealAudio"):
        parent = candidate.parent
        if "frames" in {part.lower() for part in parent.parts}:
            continue
        if all((parent / name).exists() for name in categories):
            mp4_count = sum(1 for path in parent.rglob("*.mp4"))
            candidates.append((mp4_count, parent))
    if not candidates:
        raise RuntimeError(f"Could not find raw FakeAVCeleb root under {root}.")
    candidates.sort(reverse=True, key=lambda item: item[0])
    print("[paths] FakeAVCeleb candidates:", [(count, str(path)) for count, path in candidates[:3]])
    return candidates[0][1]


def pair_cfg():
    if PAIR_NAME not in ENCODER_PAIRS:
        raise ValueError(f"Unknown PAIR_NAME={PAIR_NAME}. Choices: {sorted(ENCODER_PAIRS)}")
    return ENCODER_PAIRS[PAIR_NAME]


def local_pair_dir():
    return Path(LOCAL_ENCODER_ROOT) / PAIR_NAME


def local_pair_cache():
    return local_pair_dir() / "cmar_cache"


def drive_pair_dir():
    return Path(DRIVE_ENCODER_ROOT) / PAIR_NAME


def drive_pair_cache():
    return drive_pair_dir() / "cmar_cache"


def sync_pair_from_drive():
    src = drive_pair_cache()
    dst = local_pair_cache()
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        run(["rsync", "-a", f"{src}/", f"{dst}/"])
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def sync_pair_to_drive():
    src = local_pair_cache()
    dst = drive_pair_cache()
    if not src.exists():
        print("[sync] no local pair cache yet:", src)
        return
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        run(["rsync", "-a", f"{src}/", f"{dst}/"])
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def setup_pair_lock():
    lock_dir = Path(DRIVE_ENCODER_ROOT) / "parallel_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / f"{PAIR_NAME}.lock"
    if lock.exists() and not ALLOW_IF_LOCK_EXISTS:
        print(lock.read_text(encoding="utf-8"))
        raise RuntimeError(f"Lock exists for {PAIR_NAME}. Delete it only if you are sure it is stale.")
    lock.write_text(
        json.dumps(
            {"pair": PAIR_NAME, "worker": WORKER_NAME, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            indent=2,
        ),
        encoding="utf-8",
    )
    return lock


def release_pair_lock(lock):
    try:
        Path(lock).unlink(missing_ok=True)
    except Exception as exc:
        print("[warn] could not remove lock:", exc)


def cache_counts():
    import pandas as pd

    cache = local_pair_cache()
    report = {}
    complete = True
    for split in ["train", "val", "test"]:
        manifest_path = cache / "manifests" / f"{split}.csv"
        if not manifest_path.exists():
            report[split] = {"manifest_rows": 0, "visual_files": 0, "audio_files": 0, "complete": False}
            complete = False
            continue
        manifest = pd.read_csv(manifest_path)
        visual = len(list((cache / "features" / "visual" / split).glob("*.pt")))
        audio = len(list((cache / "features" / "audio" / split).glob("*.pt")))
        split_complete = len(manifest) > 0 and len(manifest) == visual == audio
        report[split] = {
            "manifest_rows": int(len(manifest)),
            "visual_files": int(visual),
            "audio_files": int(audio),
            "complete": bool(split_complete),
        }
        complete = complete and split_complete
    report["complete"] = bool(complete)
    return report


def print_counts(prefix):
    counts = cache_counts()
    print(prefix, json.dumps(counts, indent=2), flush=True)
    return counts


def write_worker_status(status):
    status_dir = Path(DRIVE_ENCODER_ROOT) / "parallel_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"{PAIR_NAME}_{WORKER_NAME}.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_one_slice(fakeav_root):
    cfg = pair_cfg()
    cmd = [
        sys.executable,
        str(Path(CMAR_DIR) / "scripts" / "01_preprocess_features.py"),
        "--config",
        str(Path(CMAR_DIR) / "configs" / "preprocess_fakeavceleb.json"),
        "--dataset-root",
        str(fakeav_root),
        "--output-dir",
        str(local_pair_cache()),
        "--visual-model-name",
        cfg["visual"],
        "--audio-model-name",
        cfg["audio"],
        "--no-degraded",
        "--splits",
        "train",
        "val",
        "test",
        "--max-new-rows",
        str(SLICE_ROWS),
        "--max-runtime-seconds",
        str(SLICE_SECONDS),
        "--chunk-size",
        str(CHUNK_SIZE),
    ]
    # Use visual micro-batching if configured for this pair (e.g. ViT-Base).
    micro = cfg.get("visual_micro_batch", 0)
    if micro > 0:
        cmd += ["--visual-micro-batch", str(micro)]
    return run(cmd, check=False)


def validate_complete_cache():
    counts = cache_counts()
    print("[validate]", json.dumps(counts, indent=2))
    if not counts["complete"]:
        raise RuntimeError("Encoder cache is incomplete. Resume preprocessing before upload.")
    return counts


def prepare_upload_dir():
    cfg = pair_cfg()
    upload = Path("/content") / f"kaggle_upload_{PAIR_NAME}"
    if upload.exists():
        shutil.rmtree(upload)
    upload.mkdir(parents=True)

    pair_dir = local_pair_dir()
    (pair_dir / "pair_config.json").write_text(
        json.dumps({"name": PAIR_NAME, "visual": cfg["visual"], "audio": cfg["audio"]}, indent=2),
        encoding="utf-8",
    )

    zip_name = f"encoder_{PAIR_NAME}_cmar_cache.zip"
    run(["zip", "-r", "-q", str(upload / zip_name), PAIR_NAME], cwd=str(Path(LOCAL_ENCODER_ROOT)))

    dataset_id = f"{KAGGLE_USERNAME.lower()}/{cfg['dataset_suffix']}"
    metadata = {
        "title": cfg["title"],
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            f"CMAR encoder-family feature cache for {PAIR_NAME}: "
            f"visual={cfg['visual']}, audio={cfg['audio']}. "
            "Contains a zipped pair directory with cmar_cache/manifests and cmar_cache/features."
        ),
    }
    (upload / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print((upload / "dataset-metadata.json").read_text())
    print("Zip size GB:", round((upload / zip_name).stat().st_size / 1024**3, 3))
    return upload, dataset_id


def upload_dataset(upload_dir, dataset_id):
    message = f"CMAR encoder-family cache for {PAIR_NAME}"
    rc = run(["kaggle", "datasets", "version", "-p", str(upload_dir), "-m", message, "--dir-mode", "zip"], check=False)
    if rc != 0:
        print("[upload] version failed; trying create")
        run(["kaggle", "datasets", "create", "-p", str(upload_dir), "--dir-mode", "zip"], check=True)
    print("[done] Upload command finished. Dataset:", dataset_id)


def main():
    mount_drive()
    setup_kaggle_auth()
    prepare_cmar_source()
    download_dataset(FAKEAVCELEB_DATASET, Path(DATA_ROOT) / "fakeavceleb")
    fakeav_root = find_fakeav_root()
    print("[pair]", PAIR_NAME, pair_cfg())
    print("[paths] FakeAVCeleb:", fakeav_root)
    print("[cache] Drive pair cache:", drive_pair_cache())
    print("[cache] Local pair cache:", local_pair_cache())

    sync_pair_from_drive()
    lock = setup_pair_lock()
    started = time.time()
    try:
        while (time.time() - started) < MAX_TOTAL_SECONDS:
            before = print_counts("[before]")
            write_worker_status(
                {
                    "worker": WORKER_NAME,
                    "pair": PAIR_NAME,
                    "phase": "before_slice",
                    "counts": before,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            if before["complete"]:
                print("[done] pair cache already complete")
                break

            rc = run_one_slice(fakeav_root)
            sync_pair_to_drive()
            after = print_counts("[after]")
            write_worker_status(
                {
                    "worker": WORKER_NAME,
                    "pair": PAIR_NAME,
                    "phase": "after_slice",
                    "return_code": rc,
                    "counts": after,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            if rc != 0:
                print("[warn] slice failed; reduce SLICE_ROWS or CHUNK_SIZE if this repeats")
                break
            if after["complete"]:
                print("[done] pair cache complete and synced to Drive")
                break
        else:
            print("[stop] MAX_TOTAL_SECONDS reached; rerun this worker to resume")
    finally:
        sync_pair_to_drive()
        final_counts = print_counts("[final]")
        if final_counts["complete"]:
            release_pair_lock(lock)
        else:
            print("[lock] keeping lock because pair cache is incomplete:", lock)

    validate_complete_cache()
    upload_dir, dataset_id = prepare_upload_dir()
    upload_dataset(upload_dir, dataset_id)


main()
```

---

## Part B: Kaggle Downstream Cells

Attach these Kaggle datasets:

- `cmar-code`, mounted as `/kaggle/input/datasets/vasuaashadesai/cmar-code/CMAR`
- the three Colab-uploaded pair-cache datasets from Part A

Run these cells after all required pair caches have been uploaded.

## Cell 1: Configure Paths, Install Requirements, Define Helpers

```python
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
from typing import Iterable

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

CODE_ROOT = Path("/kaggle/input/datasets/vasuaashadesai/cmar-code/CMAR")
PROJECT_DIR = CODE_ROOT
os.chdir(PROJECT_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

PHASE2 = KAGGLE_WORKING / "phase2_encoder_family"
ENCODER_STUDY = PHASE2 / "encoder_study"
ENCODER_STUDY.mkdir(parents=True, exist_ok=True)

def run(args: Iterable[object], cwd: Path = PROJECT_DIR) -> None:
    cmd = [str(x) for x in args]
    print("\n[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)

def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

print("PROJECT_DIR =", PROJECT_DIR)
print("PHASE2 =", PHASE2)
```

---

## Cell 2: Configure Run Size And Encoder Pairs

```python
# Set True only for quick wiring checks.
SMOKE = False

if SMOKE:
    N0 = 20
    N = 100
    MAX_SAMPLES = 30
    EPOCHS = 2
else:
    N0 = 100
    N = 1000
    MAX_SAMPLES = None
    EPOCHS = 30

PRIMARY_SEED = 2026

ENCODER_PAIRS = [
    "clip_whispertiny",
    "dinov2small_whisperbase",
    "dinov2small_hubert",
]
```

---

## Cell 3: Unpack Uploaded Pair Caches

This cell searches attached Kaggle inputs for
`encoder_<pair>_cmar_cache.zip` and unpacks each pair to
`PHASE2/encoder_study/<pair>/cmar_cache`.

```python
def find_pair_zip(pair: str) -> Path:
    zip_name = f"encoder_{pair}_cmar_cache.zip"
    hits = list(KAGGLE_INPUT.glob(f"**/{zip_name}"))
    if not hits:
        raise FileNotFoundError(f"Could not find {zip_name}. Attach the Kaggle dataset uploaded from Colab.")
    return hits[0]

for pair in ENCODER_PAIRS:
    pair_dir = ENCODER_STUDY / pair
    pair_cache = pair_dir / "cmar_cache"
    if (pair_cache / "features").exists() and (pair_cache / "manifests").exists():
        print("skip existing unpacked cache:", pair_cache)
        continue

    pair_dir.mkdir(parents=True, exist_ok=True)
    zip_path = find_pair_zip(pair)
    print("unpacking", zip_path, "->", ENCODER_STUDY)
    shutil.unpack_archive(str(zip_path), str(ENCODER_STUDY))

    if not (pair_cache / "features").exists() or not (pair_cache / "manifests").exists():
        raise RuntimeError(f"Unpacked cache is incomplete for {pair}: {pair_cache}")
```

---

## Cell 4: Fit PCA, Train No-Noise Baselines, Certify

```python
for pair in ENCODER_PAIRS:
    pair_dir = ENCODER_STUDY / pair
    pair_cache = pair_dir / "cmar_cache"
    pair_dir.mkdir(parents=True, exist_ok=True)

    pca_path = pair_dir / "pca_joint.pt"
    pca_summary = pair_dir / "pca_joint.summary.json"
    if not pca_path.exists() or not pca_summary.exists():
        run([
            sys.executable, "scripts/20_fit_pca_noise.py",
            "--cache-dir", pair_cache,
            "--feature-space", "joint",
            "--output", pca_path,
            "--summary-output", pca_summary,
        ])
    else:
        print("skip existing PCA:", pca_path)

    no_noise_dir = pair_dir / "baseline_no_noise"
    no_noise_ckpt = no_noise_dir / "best.pt"
    if not no_noise_ckpt.exists():
        run([
            sys.executable, "scripts/14_train_baseline_no_noise.py",
            "--cache-dir", pair_cache,
            "--output-dir", no_noise_dir,
            "--epochs", EPOCHS,
            "--batch-size", "8",
            "--grad-accum", "4",
            "--patience", "7",
            "--seed", PRIMARY_SEED,
        ])
    else:
        print("skip existing checkpoint:", no_noise_ckpt)

    cert_out = pair_dir / "baseline_no_noise_cert.json"
    if not cert_out.exists():
        args = [
            sys.executable, "scripts/11_certify.py",
            "--checkpoint", no_noise_ckpt,
            "--sigma", "1.00",
            "--noise-mode", "joint",
            "--cache-dir", pair_cache,
            "--output", cert_out,
            "--n0", N0,
            "--n", N,
            "--alpha", "0.001",
            "--seed", PRIMARY_SEED,
        ]
        if MAX_SAMPLES is not None:
            args += ["--max-samples", MAX_SAMPLES]
        run(args)
    else:
        print("skip existing certification:", cert_out)
```

---

## Cell 5: Summarize Encoder Scaling Rows

```python
SUMMARY = PHASE2 / "summary"
run([
    sys.executable, "scripts/23_phase2_summarize.py",
    "--phase2-dir", PHASE2,
    "--output-dir", SUMMARY,
])

print(read_json(SUMMARY / "phase2_summary.json"))

import pandas as pd
display(pd.read_csv(SUMMARY / "phase2_encoder_scaling.csv"))
```

---

## Cell 6: Package Downstream Encoder Outputs For Kaggle Upload

```python
ARCHIVE_BASE = KAGGLE_WORKING / "certav_phase2_encoder_family_results"
if ARCHIVE_BASE.exists():
    shutil.rmtree(ARCHIVE_BASE)
shutil.copytree(PHASE2, ARCHIVE_BASE)
print("Upload this folder as a Kaggle dataset:")
print(ARCHIVE_BASE)
```
