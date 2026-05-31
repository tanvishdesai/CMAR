# Copy this entire file into one Google Colab code cell.
# Fill in the CONFIG block, then run the cell. Keep the notebook private if you
# hard-code a Kaggle key.

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path


# =========================
# CONFIG: EDIT THESE VALUES
# =========================

KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME"
KAGGLE_KEY = "YOUR_KAGGLE_KEY_FROM_KAGGLE_JSON"

# Input datasets on Kaggle.
FAKEAVCELEB_DATASET = "shreyaty08/fakeavceleb"
LAVDF_DATASET = "elin75/localized-audio-visual-deepfake-dataset-lav-df"

# Output Kaggle dataset to create/update.
OUTPUT_DATASET_ID = f"{KAGGLE_USERNAME}/cmar-features-v1"
OUTPUT_DATASET_TITLE = "CMAR Features V1"

# Put CMAR.zip in Google Drive at this path, or upload it manually to /content.
# The zip should contain the CMAR folder with cmar/, scripts/, configs/, docs/.
CMAR_ZIP_DRIVE_PATH = "/content/drive/MyDrive/CMAR.zip"
CMAR_ZIP_UPLOAD_PATH = "/content/CMAR.zip"

# Persistent feature-cache mirror in Google Drive.
DRIVE_CACHE_DIR = "/content/drive/MyDrive/cmar_cache"

# Fast local working paths. These vanish if Colab disconnects, but are synced to Drive.
CMAR_DIR = "/content/CMAR"
WORK_CACHE_DIR = "/content/cmar_cache"
DATA_ROOT = "/content/data"

# Preprocessing mode:
#   "clean"    -> train/val/test clean features only. Do this first.
#   "degraded" -> degraded test features only, after clean is done.
#   "all"      -> clean and degraded. More fragile; use only if the runtime is stable.
PREPROCESS_MODE = "clean"

# For degraded mode, use a small subset at a time if needed, e.g. ["d1_jpeg75"].
DEGRADED_CONDITIONS = [
    "d1_jpeg75",
    "d2_jpeg50",
    "d3_resize075",
    "d4_resize050",
    "d5_vnoise001",
    "d6_vnoise002",
    "d7_mp3_128k",
    "d8_mp3_64k",
    "d9_anoise_30db",
    "d10_anoise_20db",
    "d11_h264_crf28",
    "d12_social",
]

# Runtime safety. Lower slice rows if Colab still crashes.
SLICE_ROWS = 150
SLICE_SECONDS = 780
CHUNK_SIZE = 50
MAX_TOTAL_SECONDS = 10_000

# Upload behavior.
UPLOAD_TO_KAGGLE_WHEN_DONE = True
ALLOW_PARTIAL_UPLOAD = False
PACKAGE_CACHE_AS_SINGLE_ZIP = True
KAGGLE_UPLOAD_DIR = "/content/cmar_kaggle_upload"


# =========================
# UTILITIES
# =========================

def run(command, check=True, cwd=None):
    print("\n[run]", " ".join(map(str, command)), flush=True)
    completed = subprocess.run(list(map(str, command)), cwd=cwd, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(map(str, command))}")
    return completed.returncode


def run_capture(command, check=True, cwd=None):
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
    return completed.returncode, completed.stdout


def install_dependencies():
    run([sys.executable, "-m", "pip", "install", "-q", "kaggle", "timm>=0.9.12", "transformers>=4.37",
         "librosa>=0.10", "soundfile>=0.12", "opencv-python", "pandas", "tqdm", "scikit-learn", "seaborn"])


def setup_kaggle_auth():
    if not KAGGLE_USERNAME or KAGGLE_USERNAME == "YOUR_KAGGLE_USERNAME":
        raise ValueError("Fill KAGGLE_USERNAME and KAGGLE_KEY at the top of the cell.")
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_path = kaggle_dir / "kaggle.json"
    token_path.write_text(json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}), encoding="utf-8")
    token_path.chmod(0o600)
    run(["kaggle", "--version"])


def mount_drive():
    from google.colab import drive
    drive.mount("/content/drive")


def prepare_cmar_source():
    cmar_dir = Path(CMAR_DIR)
    if (cmar_dir / "scripts" / "01_preprocess_auto.py").exists():
        print(f"[setup] CMAR source already exists: {cmar_dir}")
        return

    zip_path = Path(CMAR_ZIP_DRIVE_PATH)
    if not zip_path.exists():
        zip_path = Path(CMAR_ZIP_UPLOAD_PATH)
    if not zip_path.exists():
        from google.colab import files
        print("[upload] Upload CMAR.zip now.")
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError("No CMAR.zip uploaded.")
        first_name = next(iter(uploaded.keys()))
        zip_path = Path("/content") / first_name

    print(f"[setup] extracting {zip_path}")
    shutil.unpack_archive(str(zip_path), "/content")

    # Handle zips that contain either CMAR/... or files directly.
    if (Path("/content") / "CMAR" / "scripts" / "01_preprocess_auto.py").exists():
        return
    candidates = list(Path("/content").glob("**/scripts/01_preprocess_auto.py"))
    if not candidates:
        raise RuntimeError("Could not find scripts/01_preprocess_auto.py after extracting CMAR zip.")
    source_root = candidates[0].parents[1]
    if cmar_dir.exists():
        shutil.rmtree(cmar_dir)
    shutil.copytree(source_root, cmar_dir)
    print(f"[setup] copied source root {source_root} -> {cmar_dir}")


def download_dataset(slug, out_dir):
    out_dir = Path(out_dir)
    done = out_dir / ".download_complete"
    if done.exists():
        print(f"[data] already downloaded: {slug} -> {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", slug, "-p", out_dir, "--unzip"])
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def find_fakeav_root():
    root = Path(DATA_ROOT) / "fakeavceleb"
    for candidate in root.rglob("RealVideo-RealAudio"):
        parent = candidate.parent
        required = ["RealVideo-RealAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"]
        if all((parent / name).exists() for name in required):
            return str(parent)
    raise RuntimeError(f"Could not find FakeAVCeleb category root under {root}")


def find_lavdf_root():
    root = Path(DATA_ROOT) / "lavdf"
    for candidate in root.rglob("metadata.min.json"):
        return str(candidate.parent)
    for candidate in root.rglob("metadata.json"):
        return str(candidate.parent)
    print("[warn] LAV-DF metadata not found; continuing without lavdf root.")
    return None


def sync_drive_to_local():
    drive_cache = Path(DRIVE_CACHE_DIR)
    work_cache = Path(WORK_CACHE_DIR)
    if not drive_cache.exists():
        return
    work_cache.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        run(["rsync", "-a", f"{drive_cache}/", f"{work_cache}/"])
    else:
        shutil.copytree(drive_cache, work_cache, dirs_exist_ok=True)


def sync_local_to_drive():
    drive_cache = Path(DRIVE_CACHE_DIR)
    work_cache = Path(WORK_CACHE_DIR)
    if not work_cache.exists():
        return
    drive_cache.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        run(["rsync", "-a", f"{work_cache}/", f"{drive_cache}/"])
    else:
        shutil.copytree(work_cache, drive_cache, dirs_exist_ok=True)


def run_preprocessing(fakeav_root, lavdf_root):
    cmd = [
        sys.executable,
        str(Path(CMAR_DIR) / "scripts" / "01_preprocess_auto.py"),
        "--config", str(Path(CMAR_DIR) / "configs" / "preprocess_fakeavceleb.json"),
        "--dataset-root", fakeav_root,
        "--output-dir", WORK_CACHE_DIR,
        "--mode", PREPROCESS_MODE,
        "--slice-rows", str(SLICE_ROWS),
        "--slice-seconds", str(SLICE_SECONDS),
        "--chunk-size", str(CHUNK_SIZE),
        "--max-total-seconds", str(MAX_TOTAL_SECONDS),
        "--mirror-dir", DRIVE_CACHE_DIR,
        "--restore-from-mirror",
    ]
    if lavdf_root:
        cmd.extend(["--lavdf-root", lavdf_root])
    if PREPROCESS_MODE in {"degraded", "all"}:
        cmd.append("--conditions")
        cmd.extend(DEGRADED_CONDITIONS)
    run(cmd, check=True)
    sync_local_to_drive()


def cache_report():
    sys.path.insert(0, CMAR_DIR)
    from cmar.training.dataset import cache_coverage_report
    from cmar.config import DEGRADED_CONDITIONS as ALL_DEG
    from cmar.evaluation.degradations import DEGRADATION_SPECS
    from cmar.utils.cache import feature_path
    import pandas as pd

    cache = Path(WORK_CACHE_DIR)
    manifest_dir = cache / "manifests"
    reports = {"clean": {}, "degraded": {}}
    clean_complete = True
    for split in ["train", "val", "test"]:
        csv = manifest_dir / f"{split}.csv"
        if not csv.exists():
            clean_complete = False
            reports["clean"][split] = {"complete": False, "reason": "missing manifest"}
            continue
        rep = cache_coverage_report(cache, csv, split=split)
        reports["clean"][split] = rep
        clean_complete = clean_complete and bool(rep["complete"])

    degraded_complete = True
    test_csv = manifest_dir / "test.csv"
    if test_csv.exists():
        test = pd.read_csv(test_csv)
        for condition in ALL_DEG:
            spec = DEGRADATION_SPECS[condition]
            total = len(test)
            visual = total if not spec.visual else sum(
                feature_path(cache, "visual", "test", str(row["clip_id"]), condition=condition).exists()
                for _, row in test.iterrows()
            )
            audio = total if not spec.audio else sum(
                feature_path(cache, "audio", "test", str(row["clip_id"]), condition=condition).exists()
                for _, row in test.iterrows()
            )
            complete = visual >= total and audio >= total
            reports["degraded"][condition] = {"total": total, "visual": int(visual), "audio": int(audio), "complete": complete}
            degraded_complete = degraded_complete and complete
    else:
        degraded_complete = False

    reports["clean_complete"] = clean_complete
    reports["degraded_complete"] = degraded_complete
    reports["ready_for_training"] = clean_complete
    Path(WORK_CACHE_DIR, "colab_cache_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2)[:5000])
    return reports


def write_dataset_metadata(upload_dir):
    metadata = {
        "title": OUTPUT_DATASET_TITLE,
        "id": OUTPUT_DATASET_ID.lower(),
        "licenses": [{"name": "CC0-1.0"}],
        "description": "CMAR preprocessed DINOv2/Whisper feature cache generated from FakeAVCeleb/LAV-DF workflow."
    }
    Path(upload_dir, "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def prepare_kaggle_upload_dir():
    source_cache = Path(WORK_CACHE_DIR)
    upload_dir = Path(KAGGLE_UPLOAD_DIR)
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    if PACKAGE_CACHE_AS_SINGLE_ZIP:
        zip_path = upload_dir / "cmar_cache.zip"
        if zip_path.exists():
            zip_path.unlink()
        run(["zip", "-r", "-q", str(zip_path), "cmar_cache"], cwd="/content")
        print(f"[upload] packaged cache zip: {zip_path} ({zip_path.stat().st_size / 1024**3:.2f} GB)")
    else:
        shutil.copytree(source_cache, upload_dir / "cmar_cache", dirs_exist_ok=True)

    write_dataset_metadata(upload_dir)
    print("[upload] dataset metadata:")
    print((upload_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    return upload_dir


def upload_to_kaggle_if_ready(reports):
    if not UPLOAD_TO_KAGGLE_WHEN_DONE:
        print("[upload] disabled")
        return
    if not ALLOW_PARTIAL_UPLOAD:
        if PREPROCESS_MODE == "clean" and not reports["clean_complete"]:
            print("[upload] clean cache incomplete; not uploading yet.")
            return
        if PREPROCESS_MODE in {"degraded", "all"} and not (reports["clean_complete"] and reports["degraded_complete"]):
            print("[upload] full cache incomplete; not uploading yet.")
            return

    upload_dir = prepare_kaggle_upload_dir()

    # Try version first. If the dataset does not exist yet, create it.
    message = f"CMAR feature cache update: mode={PREPROCESS_MODE}, time={time.strftime('%Y-%m-%d %H:%M:%S')}"
    rc, _ = run_capture(["kaggle", "datasets", "version", "-p", upload_dir, "-m", message, "--dir-mode", "zip"], check=False)
    if rc != 0:
        print("[upload] version failed; trying dataset create")
        run_capture(["kaggle", "datasets", "create", "-p", upload_dir, "--dir-mode", "zip"], check=True)


# =========================
# MAIN
# =========================

mount_drive()
install_dependencies()
setup_kaggle_auth()
prepare_cmar_source()

download_dataset(FAKEAVCELEB_DATASET, Path(DATA_ROOT) / "fakeavceleb")
download_dataset(LAVDF_DATASET, Path(DATA_ROOT) / "lavdf")

sync_drive_to_local()
fakeav_root = find_fakeav_root()
lavdf_root = find_lavdf_root()
print("[paths] FakeAVCeleb:", fakeav_root)
print("[paths] LAV-DF:", lavdf_root)

run_preprocessing(fakeav_root, lavdf_root)
reports = cache_report()
sync_local_to_drive()
upload_to_kaggle_if_ready(reports)

print("\n[DONE] Colab CMAR preprocessing driver finished.")
