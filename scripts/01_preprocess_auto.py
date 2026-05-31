from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cmar.config import DEGRADED_CONDITIONS
from cmar.evaluation.degradations import DEGRADATION_SPECS
from cmar.training.dataset import cache_coverage_report
from cmar.utils.cache import feature_path


def count_degraded(cache_dir: Path, manifest_csv: Path, condition: str) -> Dict[str, int]:
    manifest = pd.read_csv(manifest_csv)
    spec = DEGRADATION_SPECS[condition]
    out = {"total": len(manifest), "visual": 0, "audio": 0}
    if spec.visual:
        out["visual"] = sum(
            feature_path(cache_dir, "visual", "test", str(row["clip_id"]), condition=condition).exists()
            for _, row in manifest.iterrows()
        )
    else:
        out["visual"] = out["total"]
    if spec.audio:
        out["audio"] = sum(
            feature_path(cache_dir, "audio", "test", str(row["clip_id"]), condition=condition).exists()
            for _, row in manifest.iterrows()
        )
    else:
        out["audio"] = out["total"]
    out["complete"] = int(out["visual"] >= out["total"] and out["audio"] >= out["total"])
    return out


def run_command(command: List[str]) -> int:
    print("\n[run]", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def write_status(path: Path, status: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def sync_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        command = ["rsync", "-a", f"{src}/", f"{dst}/"]
        print("[sync]", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        return
    print(f"[sync] copying {src} -> {dst}", flush=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automatically resume CMAR preprocessing in safe subprocess slices."
    )
    parser.add_argument("--config", default="configs/preprocess_fakeavceleb.json")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--lavdf-root", default=None)
    parser.add_argument("--output-dir", default="/kaggle/working/cmar_cache")
    parser.add_argument("--mode", choices=["clean", "degraded", "all"], default="clean")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--conditions", nargs="*", default=DEGRADED_CONDITIONS)
    parser.add_argument("--slice-rows", type=int, default=150)
    parser.add_argument("--slice-seconds", type=float, default=780.0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-total-seconds", type=float, default=10800.0)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--status-json", default=None)
    parser.add_argument("--mirror-dir", default=None, help="Optional persistent mirror, e.g. Google Drive.")
    parser.add_argument(
        "--restore-from-mirror",
        action="store_true",
        help="Copy mirror-dir into output-dir before starting. Useful after Colab reconnects.",
    )
    args = parser.parse_args()

    root = Path(args.output_dir)
    mirror_dir = Path(args.mirror_dir) if args.mirror_dir else None
    if mirror_dir is not None and args.restore_from_mirror and mirror_dir.exists():
        sync_tree(mirror_dir, root)

    manifest_dir = root / "manifests"
    preprocess_script = Path(__file__).resolve().parent / "01_preprocess_features.py"
    started = time.time()
    failures = 0
    status_path = Path(args.status_json) if args.status_json else root / "preprocess_auto_status.json"

    def common_args() -> List[str]:
        cmd = [
            sys.executable,
            str(preprocess_script),
            "--config",
            args.config,
            "--output-dir",
            args.output_dir,
            "--max-new-rows",
            str(args.slice_rows),
            "--max-runtime-seconds",
            str(args.slice_seconds),
            "--chunk-size",
            str(args.chunk_size),
        ]
        if args.dataset_root:
            cmd.extend(["--dataset-root", args.dataset_root])
        if args.lavdf_root:
            cmd.extend(["--lavdf-root", args.lavdf_root])
        return cmd

    def time_left() -> bool:
        return (time.time() - started) < args.max_total_seconds

    # Ensure manifests exist before coverage reports. This first command may do
    # some feature extraction too, but it is still bounded by the slice budget.
    if not (manifest_dir / "train.csv").exists():
        cmd = common_args() + ["--no-degraded", "--splits", args.splits[0]]
        rc = run_command(cmd)
        if rc != 0:
            raise SystemExit(rc)

    if args.mode in {"clean", "all"}:
        for split in args.splits:
            while time_left():
                report = cache_coverage_report(
                    root,
                    manifest_dir / f"{split}.csv",
                    split=split,
                    visual_only=False,
                )
                status = {"phase": "clean", "split": split, "report": report}
                write_status(status_path, status)
                print(
                    f"[status] clean {split}: "
                    f"{report['available_rows']}/{report['total_rows']} available",
                    flush=True,
                )
                if report["complete"]:
                    break
                cmd = common_args() + ["--no-degraded", "--splits", split]
                rc = run_command(cmd)
                if mirror_dir is not None:
                    sync_tree(root, mirror_dir)
                if rc != 0:
                    failures += 1
                    print(f"[warn] slice failed with rc={rc}; failures={failures}", flush=True)
                    if failures >= args.max_failures:
                        raise SystemExit(rc)
                    args.slice_rows = max(25, args.slice_rows // 2)
                    args.slice_seconds = max(300.0, args.slice_seconds * 0.75)
                else:
                    failures = 0
            if not time_left():
                print("[stop] total runtime budget reached during clean preprocessing", flush=True)
                return

    if args.mode in {"degraded", "all"}:
        test_manifest = manifest_dir / "test.csv"
        for condition in args.conditions:
            while time_left():
                counts = count_degraded(root, test_manifest, condition)
                write_status(
                    status_path,
                    {"phase": "degraded", "condition": condition, "counts": counts},
                )
                print(
                    f"[status] {condition}: visual={counts['visual']}/{counts['total']} "
                    f"audio={counts['audio']}/{counts['total']}",
                    flush=True,
                )
                if counts["complete"]:
                    break
                cmd = common_args() + ["--degraded-only", "--conditions", condition]
                rc = run_command(cmd)
                if mirror_dir is not None:
                    sync_tree(root, mirror_dir)
                if rc != 0:
                    failures += 1
                    print(f"[warn] slice failed with rc={rc}; failures={failures}", flush=True)
                    if failures >= args.max_failures:
                        raise SystemExit(rc)
                    args.slice_rows = max(25, args.slice_rows // 2)
                    args.slice_seconds = max(300.0, args.slice_seconds * 0.75)
                else:
                    failures = 0
            if not time_left():
                print("[stop] total runtime budget reached during degraded preprocessing", flush=True)
                return

    write_status(status_path, {"phase": "done", "elapsed_seconds": time.time() - started})
    if mirror_dir is not None:
        sync_tree(root, mirror_dir)
    print("[done] Requested preprocessing is complete.", flush=True)


if __name__ == "__main__":
    main()
