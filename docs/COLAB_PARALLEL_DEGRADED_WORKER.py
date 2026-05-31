# Copy this entire file into one Google Colab code cell.
#
# Purpose:
#   Run ONE degradation condition in ONE Colab account/notebook.
#
# Why this exists:
#   Copying the full clean cache from Google Drive can take 20+ minutes because
#   it contains thousands of small .pt files. Degraded preprocessing only needs
#   the manifests and raw videos, not the clean feature tensors. This worker
#   copies only manifests, processes one condition locally, then syncs only that
#   condition folder back into the shared Drive cmar_cache.
#
# Safe parallel pattern:
#   Account 1: CONDITION_TO_PROCESS = "d4_resize050"
#   Account 2: CONDITION_TO_PROCESS = "d5_vnoise001"
#   Account 3: CONDITION_TO_PROCESS = "d6_vnoise002"
#   ...
#
# All workers write to different subfolders:
#   /content/drive/MyDrive/cmar_cache/features/degraded_test/<condition>/

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

KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME"
KAGGLE_KEY = "YOUR_KAGGLE_KEY"

FAKEAVCELEB_DATASET = "shreyaty08/fakeavceleb"
LAVDF_DATASET = "elin75/localized-audio-visual-deepfake-dataset-lav-df"

# Choose exactly one condition per notebook/account.
CONDITION_TO_PROCESS = "d4_resize050"

# Give each browser/account a label so lock/status files are readable.
WORKER_NAME = "account_01"

# Shared Drive folders. Share these folders with every Google account you use.
DRIVE_CMAR_DIR = "/content/drive/MyDrive/CMAR"
DRIVE_CMAR_ZIP = "/content/drive/MyDrive/CMAR.zip"
DRIVE_CACHE_DIR = "/content/drive/MyDrive/cmar_cache"

# Local runtime folders.
CMAR_DIR = "/content/CMAR"
WORK_CACHE_DIR = "/content/cmar_cache"
DATA_ROOT = "/content/data"

# Slice size. If a condition still crashes, reduce SLICE_ROWS to 25-50.
SLICE_ROWS = 100
SLICE_SECONDS = 700
CHUNK_SIZE = 25
MAX_TOTAL_SECONDS = 13_000

# Lock behavior. Keep this False unless you know the stale lock is yours.
ALLOW_IF_LOCK_EXISTS = False


ALL_CONDITIONS = [
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


def run(command, check=True, cwd=None, capture=False):
    print("\n[run]", " ".join(map(str, command)), flush=True)
    if capture:
        completed = subprocess.run(
            list(map(str, command)),
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout)
    else:
        completed = subprocess.run(list(map(str, command)), cwd=cwd, check=False)
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


def find_lavdf_root():
    root = Path(DATA_ROOT) / "lavdf"
    for candidate in root.rglob("metadata.min.json"):
        return candidate.parent
    for candidate in root.rglob("metadata.json"):
        return candidate.parent
    return None


def copy_manifests_only(fakeav_root):
    import pandas as pd

    src = Path(DRIVE_CACHE_DIR) / "manifests"
    dst = Path(WORK_CACHE_DIR) / "manifests"
    if not src.exists():
        raise FileNotFoundError(f"Drive manifests not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["train.csv", "val.csv", "test.csv", "lavdf_test.csv"]:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    categories = {"RealVideo-RealAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"}
    for name in ["train.csv", "val.csv", "test.csv"]:
        path = dst / name
        df = pd.read_csv(path)
        new_video_paths = []
        new_audio_paths = []
        for _, row in df.iterrows():
            old = Path(str(row["video_path"]))
            parts = old.parts
            rel = None
            for idx, part in enumerate(parts):
                if part in categories:
                    rel = Path(*parts[idx:])
                    break
            if rel is None and "source_category" in row:
                rel = Path(str(row["source_category"])) / old.name
            new_path = Path(fakeav_root) / rel if rel is not None else old
            new_video_paths.append(str(new_path))
            new_audio_paths.append(str(new_path))
        df["video_path"] = new_video_paths
        df["audio_path"] = new_audio_paths
        df.to_csv(path, index=False)
    print("[manifest] copied and rewrote manifests into", dst)


def drive_condition_dir(condition):
    return Path(DRIVE_CACHE_DIR) / "features" / "degraded_test" / condition


def local_condition_dir(condition):
    return Path(WORK_CACHE_DIR) / "features" / "degraded_test" / condition


def sync_condition_from_drive(condition):
    src = drive_condition_dir(condition)
    dst = local_condition_dir(condition)
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        run(["rsync", "-a", f"{src}/", f"{dst}/"])
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def sync_condition_to_drive(condition):
    src = local_condition_dir(condition)
    dst = drive_condition_dir(condition)
    if not src.exists():
        print("[sync] no local condition output yet:", src)
        return
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        run(["rsync", "-a", f"{src}/", f"{dst}/"])
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def setup_condition_lock(condition):
    lock_dir = Path(DRIVE_CACHE_DIR) / "parallel_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / f"{condition}.lock"
    if lock.exists() and not ALLOW_IF_LOCK_EXISTS:
        print(lock.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"Lock exists for {condition}. Another worker may be running it. "
            "Delete the lock manually only if you are sure it is stale."
        )
    lock.write_text(
        json.dumps(
            {
                "condition": condition,
                "worker": WORKER_NAME,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return lock


def release_condition_lock(lock):
    try:
        Path(lock).unlink(missing_ok=True)
    except Exception as exc:
        print("[warn] could not remove lock:", exc)


def condition_counts(condition):
    sys.path.insert(0, CMAR_DIR)
    from cmar.evaluation.degradations import DEGRADATION_SPECS
    from cmar.utils.cache import feature_path
    import pandas as pd

    cache = Path(WORK_CACHE_DIR)
    test = pd.read_csv(cache / "manifests" / "test.csv")
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
    return {
        "total": int(total),
        "visual": int(visual),
        "audio": int(audio),
        "done": int(done),
        "remaining": int(total - done),
        "complete": bool(visual >= total and audio >= total),
    }


def print_counts(condition, prefix):
    counts = condition_counts(condition)
    print(
        f"{prefix} {condition}: done={counts['done']}/{counts['total']} "
        f"remaining={counts['remaining']} visual={counts['visual']} audio={counts['audio']} "
        f"complete={counts['complete']}",
        flush=True,
    )
    return counts


def run_one_slice(condition):
    cmd = [
        sys.executable,
        str(Path(CMAR_DIR) / "scripts" / "01_preprocess_features.py"),
        "--config",
        str(Path(CMAR_DIR) / "configs" / "preprocess_fakeavceleb.json"),
        "--output-dir",
        WORK_CACHE_DIR,
        "--degraded-only",
        "--conditions",
        condition,
        "--max-new-rows",
        str(SLICE_ROWS),
        "--max-runtime-seconds",
        str(SLICE_SECONDS),
        "--chunk-size",
        str(CHUNK_SIZE),
    ]
    return run(cmd, check=False)


def write_worker_status(condition, status):
    status_dir = Path(DRIVE_CACHE_DIR) / "parallel_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / f"{condition}_{WORKER_NAME}.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def main():
    if CONDITION_TO_PROCESS not in ALL_CONDITIONS:
        raise ValueError(f"Unknown condition: {CONDITION_TO_PROCESS}")

    mount_drive()
    setup_kaggle_auth()
    prepare_cmar_source()
    download_dataset(FAKEAVCELEB_DATASET, Path(DATA_ROOT) / "fakeavceleb")
    download_dataset(LAVDF_DATASET, Path(DATA_ROOT) / "lavdf")

    fakeav_root = find_fakeav_root()
    lavdf_root = find_lavdf_root()
    print("[paths] FakeAVCeleb:", fakeav_root)
    print("[paths] LAV-DF:", lavdf_root)
    print("[cache] shared Drive cache:", DRIVE_CACHE_DIR)
    print("[cache] local minimal cache:", WORK_CACHE_DIR)
    print("[condition]", CONDITION_TO_PROCESS)

    copy_manifests_only(fakeav_root)
    sync_condition_from_drive(CONDITION_TO_PROCESS)

    lock = setup_condition_lock(CONDITION_TO_PROCESS)
    started = time.time()
    try:
        while (time.time() - started) < MAX_TOTAL_SECONDS:
            before = print_counts(CONDITION_TO_PROCESS, "[before]")
            write_worker_status(
                CONDITION_TO_PROCESS,
                {
                    "worker": WORKER_NAME,
                    "condition": CONDITION_TO_PROCESS,
                    "phase": "before_slice",
                    "counts": before,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            if before["complete"]:
                print("[done] condition already complete")
                break

            rc = run_one_slice(CONDITION_TO_PROCESS)
            sync_condition_to_drive(CONDITION_TO_PROCESS)
            after = print_counts(CONDITION_TO_PROCESS, "[after]")
            write_worker_status(
                CONDITION_TO_PROCESS,
                {
                    "worker": WORKER_NAME,
                    "condition": CONDITION_TO_PROCESS,
                    "phase": "after_slice",
                    "return_code": rc,
                    "counts": after,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            if rc != 0:
                print("[warn] slice failed; reduce SLICE_ROWS if this repeats")
                break
            if after["complete"]:
                print("[done] condition complete and synced to Drive")
                break
        else:
            print("[stop] MAX_TOTAL_SECONDS reached; rerun this worker to resume")
    finally:
        sync_condition_to_drive(CONDITION_TO_PROCESS)
        final_counts = print_counts(CONDITION_TO_PROCESS, "[final]")
        if final_counts["complete"]:
            release_condition_lock(lock)
        else:
            print("[lock] keeping lock because condition is incomplete:", lock)


main()
