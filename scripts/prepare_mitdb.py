#!/usr/bin/env python3
"""Prepare MIT-BIH Arrhythmia Database for medtsllm2 beat classification.

Creates:
    data/mitdb/processed/train.npz
    data/mitdb/processed/val.npz
    data/mitdb/processed/test.npz
    data/mitdb/processed/metadata.json

The test set uses the common AAMI-style DS2 record set. The DS1 records are
split at record level into training and validation subsets, avoiding beat-level
leakage. Paced records 102, 104, 107, and 217 are excluded, as is common in
this protocol.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import wfdb

CLASS_ORDER = ["N", "S", "V", "F", "Q"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_ORDER)}

# AAMI EC57-style five-group mapping. Symbols not present here are rhythm,
# quality, or other non-beat annotations and are ignored.
SYMBOL_TO_AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

# Common inter-patient division used in AAMI-style MIT-BIH studies.
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118",
    "119", "122", "124", "201", "203", "205", "207", "208", "209",
    "215", "220", "223", "230",
]
DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200",
    "202", "210", "212", "213", "214", "219", "221", "222", "228",
    "231", "232", "233", "234",
]
EXCLUDED_PACED_RECORDS = ["102", "104", "107", "217"]

# x, y, record_ids, descriptions, selected_lead
RecordData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("data/mitdb"),
        help="Dataset root. Raw WFDB files must be directly in ROOT/raw.",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download mitdb 1.0.0 from PhysioNet using wfdb.dl_database.",
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument(
        "--lead", default="MLII",
        help="Preferred signal name; falls back to channel 0 when unavailable.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing processed files.",
    )
    return parser.parse_args()


def select_channel(signal_names: Sequence[str], preferred: str) -> Tuple[int, str]:
    if not signal_names:
        raise ValueError("WFDB record contains no signal names.")
    names_upper = [str(name).strip().upper() for name in signal_names]
    preferred_upper = preferred.strip().upper()
    index = names_upper.index(preferred_upper) if preferred_upper in names_upper else 0
    return index, str(signal_names[index])


def parse_demographics(comments: Sequence[str]) -> Tuple[int | None, str | None]:
    """Extract age and sex from MIT-BIH header comments when available."""
    text = " ".join(str(comment) for comment in (comments or []))
    age = None
    sex = None

    # Typical MIT-BIH comment prefix: "69 M ...".
    match = re.search(r"(?:^|\s)(\d{1,3})\s+([MF])(?:\s|$)", text, re.I)
    if match:
        candidate = int(match.group(1))
        if 0 < candidate < 120:
            age = candidate
        sex = "male" if match.group(2).upper() == "M" else "female"
    else:
        # Defensive support for reversed forms such as "M 69".
        match = re.search(r"(?:^|\s)([MF])\s+(\d{1,3})(?:\s|$)", text, re.I)
        if match:
            candidate = int(match.group(2))
            if 0 < candidate < 120:
                age = candidate
            sex = "male" if match.group(1).upper() == "M" else "female"

    return age, sex


def make_description(age: int | None, sex: str | None) -> str:
    parts = []
    if age is not None:
        parts.append(f"age {age}")
    if sex is not None:
        parts.append(sex)
    info = ", ".join(parts) if parts else "not available"
    return f"Patient information: {info}."


def extract_record(
    raw_dir: Path,
    record_id: str,
    window_size: int,
    preferred_lead: str,
) -> RecordData:
    record_path = raw_dir / record_id
    record = wfdb.rdrecord(str(record_path), physical=True)
    annotation = wfdb.rdann(str(record_path), "atr")

    if record.p_signal is None:
        raise ValueError(f"Record {record_id} has no physical signal.")
    if int(round(float(record.fs))) != 360:
        raise ValueError(
            f"Record {record_id} has unexpected sampling rate {record.fs}; "
            "MIT-BIH v1.0.0 is expected to be 360 Hz."
        )

    channel_idx, lead_used = select_channel(record.sig_name, preferred_lead)
    signal = np.asarray(record.p_signal[:, channel_idx], dtype=np.float32)
    age, sex = parse_demographics(getattr(record, "comments", []))
    description = make_description(age, sex)

    left = window_size // 2
    right = window_size - left
    beats: List[np.ndarray] = []
    labels: List[int] = []

    for sample, symbol in zip(annotation.sample, annotation.symbol):
        aami_class = SYMBOL_TO_AAMI.get(symbol)
        if aami_class is None:
            continue

        start = int(sample) - left
        end = int(sample) + right
        # Skip incomplete boundary windows instead of padding artificial ECG.
        if start < 0 or end > signal.shape[0]:
            continue

        beat = signal[start:end]
        if beat.shape[0] != window_size or not np.isfinite(beat).all():
            continue

        beats.append(beat[:, None])
        labels.append(CLASS_TO_INDEX[aami_class])

    if not beats:
        raise RuntimeError(f"No usable beat windows extracted from record {record_id}.")

    x = np.stack(beats).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    record_ids = np.full(len(y), record_id, dtype="<U3")
    descriptions = np.full(len(y), description, dtype=f"<U{max(64, len(description))}")
    return x, y, record_ids, descriptions, lead_used


def class_counts(y: np.ndarray) -> np.ndarray:
    return np.bincount(y, minlength=len(CLASS_ORDER)).astype(np.int64)


def choose_validation_records(
    records: Dict[str, RecordData], val_fraction: float
) -> List[str]:
    """Choose a record-disjoint validation subset approximating DS1 balance."""
    record_ids = sorted(records)
    n_val = max(1, min(len(record_ids) - 1, int(round(len(record_ids) * val_fraction))))
    total_counts = sum(
        (class_counts(records[r][1]) for r in record_ids),
        np.zeros(len(CLASS_ORDER), dtype=np.int64),
    )
    total_samples = sum(len(records[r][1]) for r in record_ids)
    target_counts = total_counts.astype(np.float64) * val_fraction
    target_samples = total_samples * val_fraction

    best_score = float("inf")
    best_combo: Tuple[str, ...] | None = None
    for combo in itertools.combinations(record_ids, n_val):
        counts = sum(
            (class_counts(records[r][1]) for r in combo),
            np.zeros(len(CLASS_ORDER), dtype=np.int64),
        )
        n_samples = sum(len(records[r][1]) for r in combo)

        relative_error = (counts - target_counts) / (target_counts + 5.0)
        score = float(np.mean(relative_error ** 2))
        score += 0.25 * float(
            ((n_samples - target_samples) / (target_samples + 1.0)) ** 2
        )

        # Discourage omitting a class that has enough DS1 samples to appear.
        missing = (counts == 0) & (total_counts >= 5)
        score += 4.0 * float(missing.sum())

        if score < best_score:
            best_score = score
            best_combo = combo

    if best_combo is None:
        raise RuntimeError("Unable to choose validation records.")
    return list(best_combo)


def combine(
    records: Dict[str, RecordData], record_ids: Iterable[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [records[r] for r in record_ids]
    x = np.concatenate([item[0] for item in selected], axis=0)
    y = np.concatenate([item[1] for item in selected], axis=0)
    groups = np.concatenate([item[2] for item in selected], axis=0)
    descriptions = np.concatenate([item[3] for item in selected], axis=0)
    return x, y, groups, descriptions


def save_split(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    record_ids: np.ndarray,
    descriptions: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        x=x,
        y=y,
        record_ids=record_ids,
        descriptions=descriptions,
    )
    counts = Counter(CLASS_ORDER[int(i)] for i in y)
    print(
        f"{path.stem:>5}: samples={len(y):6d}, shape={x.shape}, "
        f"records={len(set(record_ids.tolist())):2d}, counts={dict(counts)}"
    )


def ensure_raw_data(raw_dir: Path, download: bool) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if download:
        print(f"Downloading PhysioNet mitdb into {raw_dir} ...")
        wfdb.dl_database("mitdb", dl_dir=str(raw_dir))

    required = DS1_RECORDS + DS2_RECORDS
    missing = [record for record in required if not (raw_dir / f"{record}.hea").exists()]
    if missing:
        raise FileNotFoundError(
            "Missing MIT-BIH records in " + str(raw_dir) + ": " + ", ".join(missing) + "\n"
            "Run with --download, or extract the PhysioNet archive so files such "
            "as 100.hea, 100.dat, and 100.atr are directly inside ROOT/raw/."
        )


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive.")
    if not 0.05 <= args.val_fraction <= 0.40:
        raise ValueError("--val-fraction must be between 0.05 and 0.40.")

    root = args.root.resolve()
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    outputs = [processed_dir / f"{split}.npz" for split in ("train", "val", "test")]
    if not args.overwrite and any(path.exists() for path in outputs):
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(
            "Processed files already exist: " + ", ".join(existing) +
            ". Use --overwrite to replace them."
        )

    ensure_raw_data(raw_dir, args.download)

    all_records: Dict[str, RecordData] = {}
    lead_by_record: Dict[str, str] = {}
    description_by_record: Dict[str, str] = {}

    for record_id in DS1_RECORDS + DS2_RECORDS:
        x, y, groups, descriptions, lead_used = extract_record(
            raw_dir, record_id, args.window_size, args.lead
        )
        all_records[record_id] = (x, y, groups, descriptions, lead_used)
        lead_by_record[record_id] = lead_used
        description_by_record[record_id] = str(descriptions[0])
        counts = Counter(CLASS_ORDER[int(i)] for i in y)
        print(
            f"record {record_id}: beats={len(y):4d}, lead={lead_used}, "
            f"{descriptions[0]}, counts={dict(counts)}"
        )

    ds1 = {record: all_records[record] for record in DS1_RECORDS}
    val_records = choose_validation_records(ds1, args.val_fraction)
    val_set = set(val_records)
    train_records = [record for record in DS1_RECORDS if record not in val_set]
    test_records = list(DS2_RECORDS)

    x_train, y_train, g_train, d_train = combine(all_records, train_records)
    x_val, y_val, g_val, d_val = combine(all_records, val_records)
    x_test, y_test, g_test, d_test = combine(all_records, test_records)

    save_split(processed_dir / "train.npz", x_train, y_train, g_train, d_train)
    save_split(processed_dir / "val.npz", x_val, y_val, g_val, d_val)
    save_split(processed_dir / "test.npz", x_test, y_test, g_test, d_test)

    metadata = {
        "database": "MIT-BIH Arrhythmia Database",
        "version": "1.0.0",
        "sampling_rate_hz": 360,
        "window_size": args.window_size,
        "preferred_lead": args.lead,
        "class_order": CLASS_ORDER,
        "symbol_to_aami": SYMBOL_TO_AAMI,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "excluded_paced_records": EXCLUDED_PACED_RECORDS,
        "lead_by_record": lead_by_record,
        "description_by_record": description_by_record,
        "split_note": (
            "Common record-level DS1/DS2 protocol. PhysioNet documents that "
            "records 201 and 202 came from the same subject, so the common "
            "record split is not perfectly subject-disjoint."
        ),
    }
    metadata_path = processed_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nRecord-disjoint split:")
    print("train:", " ".join(train_records))
    print("val:  ", " ".join(val_records))
    print("test: ", " ".join(test_records))
    print("Saved metadata to", metadata_path)


if __name__ == "__main__":
    main()
