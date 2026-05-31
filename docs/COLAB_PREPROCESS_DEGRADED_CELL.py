# Copy this entire file into one Google Colab code cell.
# It resumes from the clean cache in Google Drive and generates degraded test
# features condition by condition.

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# =========================
# CONFIG: EDIT THESE VALUES
# =========================

KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME"
KAGGLE_KEY = "YOUR_KAGGLE_KEY"

FAKEAVCELEB_DATASET = "shreyaty08/fakeavceleb"
LAVDF_DATASET = "elin75/localized-audio-visual-deepfake-dataset-lav-df"

CMAR_ZIP_DRIVE_PATH = "/content/drive/MyDrive/CMAR.zip"
CMAR_ZIP_UPLOAD_PATH = "/content/CMAR.zip"

DRIVE_CACHE_DIR = "/content/drive/MyDrive/cmar_cache"
WORK_CACHE_DIR = "/content/cmar_cache"
CMAR_DIR = "/content/CMAR"
DATA_ROOT = "/content/data"

# This script reuses the SAME cmar_cache made by clean preprocessing:
#   /content/drive/MyDrive/cmar_cache
#   /content/cmar_cache
#
# Degraded files are added under:
#   cmar_cache/features/degraded_test/<condition>/visual/*.pt
#   cmar_cache/features/degraded_test/<condition>/audio/*.pt
#
# Nothing is written to a separate cache folder.

# Process a few at a time if Colab is unstable. Completed conditions are skipped
# automatically on rerun.
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

# Recommended for Colab stability: 1 condition per run. Increase to 2-3 if your
# runtime is stable. Rerun the cell later; completed conditions are skipped.
MAX_CONDITIONS_THIS_RUN = 1

SLICE_ROWS = 100
SLICE_SECONDS = 700
CHUNK_SIZE = 25
MAX_TOTAL_SECONDS = 10_000

# Optional: upload degraded cache as a separate Kaggle dataset when complete.
UPLOAD_TO_KAGGLE_WHEN_DONE = False
OUTPUT_DATASET_ID = f"{KAGGLE_USERNAME.lower()}/cmar-features-degraded-v1"
OUTPUT_DATASET_TITLE = "CMAR Degraded Test Features V1"
UPLOAD_DIR = "/content/cmar_kaggle_upload_degraded"


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


def run_live(command, check=True, cwd=None):
    print("\n[run-live]", " ".join(map(str, command)), flush=True)
    completed = subprocess.run(list(map(str, command)), cwd=cwd, check=False)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(map(str, command))}")
    return completed.returncode


def mount_drive():
    from google.colab import drive
    drive.mount("/content/drive")


def setup_kaggle_auth():
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_path = kaggle_dir / "kaggle.json"
    token_path.write_text(json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}), encoding="utf-8")
    token_path.chmod(0o600)
    run(["pip", "install", "-q", "kaggle", "timm>=0.9.12", "transformers>=4.37",
         "librosa>=0.10", "soundfile>=0.12", "opencv-python", "pandas", "tqdm", "scikit-learn"])
    run(["kaggle", "--version"])


def prepare_cmar_source():
    cmar_dir = Path(CMAR_DIR)
    if (cmar_dir / "scripts" / "01_preprocess_auto.py").exists():
        return
    zip_path = Path(CMAR_ZIP_DRIVE_PATH)
    if not zip_path.exists():
        zip_path = Path(CMAR_ZIP_UPLOAD_PATH)
    if not zip_path.exists():
        from google.colab import files
        print("[upload] Upload CMAR.zip now.")
        uploaded = files.upload()
        first_name = next(iter(uploaded.keys()))
        zip_path = Path("/content") / first_name
    shutil.unpack_archive(str(zip_path), "/content")
    if not (Path(CMAR_DIR) / "scripts" / "01_preprocess_auto.py").exists():
        candidates = list(Path("/content").glob("**/scripts/01_preprocess_auto.py"))
        if not candidates:
            raise RuntimeError("Could not find CMAR source after unzip.")
        source_root = candidates[0].parents[1]
        if cmar_dir.exists():
            shutil.rmtree(cmar_dir)
        shutil.copytree(source_root, cmar_dir)


def download_dataset(slug, out_dir):
    out_dir = Path(out_dir)
    done = out_dir / ".download_complete"
    if done.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", slug, "-p", out_dir, "--unzip"])
    done.write_text("done", encoding="utf-8")


def find_fakeav_root_from_manifest():
    import pandas as pd

    manifest = Path(WORK_CACHE_DIR) / "manifests" / "test.csv"
    if not manifest.exists():
        return None
    df = pd.read_csv(manifest)
    if "video_path" not in df.columns or df.empty:
        return None
    first_path = Path(str(df.iloc[0]["video_path"]))
    parts = first_path.parts
    categories = {"RealVideo-RealAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"}
    for idx, part in enumerate(parts):
        if part in categories:
            return str(Path(*parts[:idx]))
    return None


def find_fakeav_root():
    manifest_root = find_fakeav_root_from_manifest()
    if manifest_root and Path(manifest_root).exists():
        return manifest_root

    root = Path(DATA_ROOT) / "fakeavceleb"
    categories = {"RealVideo-RealAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"}
    candidates = []
    for candidate in root.rglob("RealVideo-RealAudio"):
        parent = candidate.parent
        if "frames" in {part.lower() for part in parent.parts}:
            continue
        required = list(categories)
        if all((parent / name).exists() for name in required):
            mp4_count = sum(1 for path in parent.rglob("*.mp4"))
            candidates.append((mp4_count, parent))
    if candidates:
        candidates.sort(reverse=True, key=lambda item: item[0])
        return str(candidates[0][1])
    raise RuntimeError("Could not find FakeAVCeleb root.")


def find_lavdf_root():
    root = Path(DATA_ROOT) / "lavdf"
    for candidate in root.rglob("metadata.min.json"):
        return str(candidate.parent)
    for candidate in root.rglob("metadata.json"):
        return str(candidate.parent)
    return None


def sync_drive_to_local():
    src = Path(DRIVE_CACHE_DIR)
    dst = Path(WORK_CACHE_DIR)
    if not src.exists():
        raise FileNotFoundError("Clean cache not found in Drive. Run clean preprocessing first.")
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        run(["rsync", "-a", f"{src}/", f"{dst}/"])
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def sync_local_to_drive():
    src = Path(WORK_CACHE_DIR)
    dst = Path(DRIVE_CACHE_DIR)
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        run(["rsync", "-a", f"{src}/", f"{dst}/"])
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def condition_counts(condition):
    sys.path.insert(0, CMAR_DIR)
    from cmar.evaluation.degradations import DEGRADATION_SPECS
    from cmar.utils.cache import feature_path
    import pandas as pd

    cache = Path(WORK_CACHE_DIR)
    test_csv = cache / "manifests" / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Missing test manifest: {test_csv}")
    test = pd.read_csv(test_csv)
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
    done = min(visual, audio) if spec.visual and spec.audio else (visual if spec.visual else audio)
    complete = visual >= total and audio >= total
    return {
        "total": total,
        "visual": int(visual),
        "audio": int(audio),
        "done": int(done),
        "remaining": int(total - done),
        "complete": bool(complete),
    }


def print_condition_status(condition, index=None, total_conditions=None, prefix="[status]"):
    counts = condition_counts(condition)
    label = f"{condition}"
    if index is not None and total_conditions is not None:
        label = f"{condition} ({index}/{total_conditions})"
    print(
        f"{prefix} {label}: done={counts['done']}/{counts['total']} "
        f"remaining={counts['remaining']} visual={counts['visual']} audio={counts['audio']} "
        f"complete={counts['complete']}",
        flush=True,
    )
    return counts


def run_one_degradation(condition, fakeav_root, lavdf_root):
    cmd = [
        sys.executable,
        str(Path(CMAR_DIR) / "scripts" / "01_preprocess_auto.py"),
        "--config", str(Path(CMAR_DIR) / "configs" / "preprocess_fakeavceleb.json"),
        "--output-dir", WORK_CACHE_DIR,
        "--mode", "degraded",
        "--conditions", condition,
        "--slice-rows", str(SLICE_ROWS),
        "--slice-seconds", str(SLICE_SECONDS),
        "--chunk-size", str(CHUNK_SIZE),
        "--max-total-seconds", str(MAX_TOTAL_SECONDS),
        "--mirror-dir", DRIVE_CACHE_DIR,
        "--restore-from-mirror",
    ]
    # Degraded mode intentionally uses existing manifests from cmar_cache.
    # Passing dataset-root is unnecessary and can be harmful if the downloader
    # exposes both raw-video and frames folders.
    if fakeav_root:
        cmd.extend(["--dataset-root", fakeav_root])
    if lavdf_root:
        cmd.extend(["--lavdf-root", lavdf_root])
    run_live(cmd)
    sync_local_to_drive()


def degraded_report():
    sys.path.insert(0, CMAR_DIR)
    from cmar.config import DEGRADED_CONDITIONS as ALL_DEG
    from cmar.evaluation.degradations import DEGRADATION_SPECS
    from cmar.utils.cache import feature_path
    import pandas as pd

    cache = Path(WORK_CACHE_DIR)
    test = pd.read_csv(cache / "manifests" / "test.csv")
    report = {}
    all_done = True
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
        report[condition] = {"total": total, "visual": int(visual), "audio": int(audio), "complete": complete}
        all_done = all_done and complete
    report["degraded_complete"] = all_done
    print(json.dumps(report, indent=2))
    (Path(WORK_CACHE_DIR) / "degraded_cache_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def upload_degraded_if_complete(report):
    if not UPLOAD_TO_KAGGLE_WHEN_DONE:
        return
    if not report["degraded_complete"]:
        print("[upload] degraded cache incomplete; not uploading.")
        return
    upload = Path(UPLOAD_DIR)
    if upload.exists():
        shutil.rmtree(upload)
    upload.mkdir(parents=True)
    run(["zip", "-r", "-q", str(upload / "cmar_cache_degraded.zip"), "cmar_cache"], cwd="/content")
    metadata = {
        "title": OUTPUT_DATASET_TITLE,
        "id": OUTPUT_DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
        "description": "CMAR degraded test feature cache."
    }
    (upload / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rc = run(["kaggle", "datasets", "version", "-p", upload, "-m", "CMAR degraded cache", "--dir-mode", "zip"], check=False)
    if rc != 0:
        run(["kaggle", "datasets", "create", "-p", upload, "--dir-mode", "zip"])


mount_drive()
setup_kaggle_auth()
prepare_cmar_source()
download_dataset(FAKEAVCELEB_DATASET, Path(DATA_ROOT) / "fakeavceleb")
download_dataset(LAVDF_DATASET, Path(DATA_ROOT) / "lavdf")
sync_drive_to_local()

fakeav_root = find_fakeav_root()
lavdf_root = find_lavdf_root()
print("[paths] FakeAVCeleb:", fakeav_root)
print("[paths] LAV-DF:", lavdf_root)
print("[cache] Reusing clean/degraded cache folder:", WORK_CACHE_DIR)
print("[drive] Persistent cache mirror:", DRIVE_CACHE_DIR)

processed_this_run = 0
total_conditions = len(DEGRADED_CONDITIONS)
for idx, condition in enumerate(DEGRADED_CONDITIONS, start=1):
    before = print_condition_status(condition, idx, total_conditions, prefix="[before]")
    if before["complete"]:
        print(f"[skip] {condition} already complete; moving to next condition.", flush=True)
        continue
    if processed_this_run >= MAX_CONDITIONS_THIS_RUN:
        print(
            f"[stop] Reached MAX_CONDITIONS_THIS_RUN={MAX_CONDITIONS_THIS_RUN}. "
            "Rerun this cell to continue with the next incomplete degradation.",
            flush=True,
        )
        break
    print(
        f"\n[start] Processing degradation {idx}/{total_conditions}: {condition}. "
        f"{before['remaining']} samples left.",
        flush=True,
    )
    run_one_degradation(condition, fakeav_root, lavdf_root)
    after = print_condition_status(condition, idx, total_conditions, prefix="[after]")
    sync_local_to_drive()
    processed_this_run += 1
    if not after["complete"]:
        print(
            f"[note] {condition} still incomplete after this run. "
            "Rerun the cell; it will resume this same condition.",
            flush=True,
        )
        break

report = degraded_report()
sync_local_to_drive()
upload_degraded_if_complete(report)

print("\n[DONE] Degraded preprocessing driver finished.")
