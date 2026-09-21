"""Submission file generation and validation (single source of truth)."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

SUBMISSION_ARCNAME = "pred_results.csv"


def save_predictions(
    output_csv: Path,
    filenames: list[str],
    pred_indices: list[int],
    idx_to_class: list[str],
) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for name, pred in zip(filenames, pred_indices):
            writer.writerow([name, idx_to_class[pred]])
    return output_csv


def zip_submission(csv_path: Path) -> Path:
    """Zip the csv under the competition-required arcname pred_results.csv."""
    zip_path = csv_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=SUBMISSION_ARCNAME)
    return zip_path


def validate_submission(csv_path: Path, zip_path: Path | None, test_dir: Path, num_classes: int) -> bool:
    """Print competition-rule checks; returns True if everything passes."""
    ok = True

    if zip_path is not None:
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                if names != [SUBMISSION_ARCNAME]:
                    print(f"FAIL zip contents: {names} (expected exactly ['{SUBMISSION_ARCNAME}'])")
                    ok = False
                else:
                    print(f"OK   zip contains exactly {SUBMISSION_ARCNAME}")
        else:
            print(f"WARN zip not found: {zip_path}")

    rows: list[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fp:
        for lineno, row in enumerate(csv.reader(fp), 1):
            if len(row) != 2:
                print(f"FAIL line {lineno}: expected 2 fields, got {len(row)}: {row}")
                ok = False
                continue
            rows.append((row[0], row[1].strip()))

    valid_ids = {f"{i:04d}" for i in range(num_classes)}
    bad_ids = [(n, c) for n, c in rows if c not in valid_ids]
    if bad_ids:
        print(f"FAIL {len(bad_ids)} rows with invalid class id, e.g. {bad_ids[:3]}")
        ok = False
    else:
        print(f"OK   all {len(rows)} class ids are 4-digit and within 0000..{num_classes - 1:04d}")

    test_files = {f.name for f in Path(test_dir).iterdir() if f.is_file()}
    pred_files = [n for n, _ in rows]
    pred_set = set(pred_files)
    if len(pred_files) != len(pred_set):
        print(f"FAIL duplicate filenames in csv: {len(pred_files) - len(pred_set)}")
        ok = False
    missing = test_files - pred_set
    extra = pred_set - test_files
    if missing:
        print(f"FAIL {len(missing)} test images missing from csv, e.g. {sorted(missing)[:3]}")
        ok = False
    if extra:
        print(f"FAIL {len(extra)} csv rows not in test dir, e.g. {sorted(extra)[:3]}")
        ok = False
    if not missing and not extra:
        print(f"OK   filenames exactly match test dir ({len(test_files)} files)")

    print("RESULT:", "PASS" if ok else "FAIL")
    return ok
