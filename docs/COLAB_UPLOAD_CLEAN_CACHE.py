# Copy this entire file into one Google Colab code cell.
# It uploads the completed clean CMAR feature cache from Google Drive to Kaggle.

import json
import os
import shutil
import subprocess
from pathlib import Path


# =========================
# CONFIG: EDIT THESE VALUES
# =========================

KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME"
KAGGLE_KEY = "YOUR_KAGGLE_KEY"

OUTPUT_DATASET_ID = f"{KAGGLE_USERNAME.lower()}/cmar-features-clean-v1"
OUTPUT_DATASET_TITLE = "CMAR Clean Features V1"

DRIVE_CACHE_DIR = "/content/drive/MyDrive/cmar_cache"
WORK_CACHE_DIR = "/content/cmar_cache"
UPLOAD_DIR = "/content/cmar_kaggle_upload_clean"


def run(command, check=True, cwd=None):
    print("\n[run]", " ".join(map(str, command)), flush=True)
    completed = subprocess.run(
        list(map(str, command)),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
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
    run(["pip", "install", "-q", "kaggle"])
    run(["kaggle", "--version"])


def sync_drive_cache_to_local():
    src = Path(DRIVE_CACHE_DIR)
    dst = Path(WORK_CACHE_DIR)
    if not src.exists():
        raise FileNotFoundError(f"Drive cache not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        run(["rsync", "-a", f"{src}/", f"{dst}/"])
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def validate_clean_cache():
    root = Path(WORK_CACHE_DIR)
    required = [
        root / "manifests" / "train.csv",
        root / "manifests" / "val.csv",
        root / "manifests" / "test.csv",
        root / "features" / "visual" / "train",
        root / "features" / "audio" / "train",
        root / "features" / "visual" / "val",
        root / "features" / "audio" / "val",
        root / "features" / "visual" / "test",
        root / "features" / "audio" / "test",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Clean cache is missing required paths:\n" + "\n".join(missing))

    import pandas as pd

    expected = {"train": 3850, "val": 825, "test": 825}
    report = {}
    ok = True
    for split, expected_rows in expected.items():
        manifest = pd.read_csv(root / "manifests" / f"{split}.csv")
        visual = len(list((root / "features" / "visual" / split).glob("*.pt")))
        audio = len(list((root / "features" / "audio" / split).glob("*.pt")))
        report[split] = {
            "manifest_rows": len(manifest),
            "visual_files": visual,
            "audio_files": audio,
            "complete": len(manifest) == visual == audio == expected_rows,
        }
        ok = ok and report[split]["complete"]
    print(json.dumps(report, indent=2))
    if not ok:
        raise RuntimeError("Clean cache is not complete. Resume preprocessing before upload.")
    return report


def prepare_upload_dir():
    upload = Path(UPLOAD_DIR)
    if upload.exists():
        shutil.rmtree(upload)
    upload.mkdir(parents=True)

    run(["zip", "-r", "-q", str(upload / "cmar_cache_clean.zip"), "cmar_cache"], cwd="/content")

    metadata = {
        "title": OUTPUT_DATASET_TITLE,
        "id": OUTPUT_DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            "Clean CMAR feature cache: DINOv2-Small visual features and "
            "Whisper-Tiny audio features for FakeAVCeleb train/val/test."
        ),
    }
    (upload / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print((upload / "dataset-metadata.json").read_text())
    print("Zip size GB:", round((upload / "cmar_cache_clean.zip").stat().st_size / 1024**3, 3))
    return upload


def upload_dataset(upload_dir):
    message = "CMAR clean feature cache"
    rc = run(["kaggle", "datasets", "version", "-p", upload_dir, "-m", message, "--dir-mode", "zip"], check=False)
    if rc != 0:
        print("[upload] version failed; trying create")
        rc = run(["kaggle", "datasets", "create", "-p", upload_dir, "--dir-mode", "zip"], check=True)
    print("[done] Upload command finished. Dataset:", OUTPUT_DATASET_ID)


mount_drive()
setup_kaggle_auth()
sync_drive_cache_to_local()
validate_clean_cache()
upload_dir = prepare_upload_dir()
upload_dataset(upload_dir)

print("\nYou can train CMAR from this clean cache. Degraded cache is not required for training.")
