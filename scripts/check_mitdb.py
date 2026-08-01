#!/usr/bin/env python3
"""Validate prepared MIT-BIH arrays and record-level split isolation."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

CLASS_ORDER = ["N", "S", "V", "F", "Q"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/mitdb"))
    parser.add_argument("--window-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = args.root / "processed"
    record_sets = {}

    for split in ("train", "val", "test"):
        path = processed / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as arrays:
            required = {"x", "y", "record_ids", "descriptions"}
            missing = required.difference(arrays.files)
            if missing:
                raise KeyError(f"{path} is missing keys: {sorted(missing)}")
            x = arrays["x"]
            y = arrays["y"]
            record_ids = arrays["record_ids"].astype(str)
            descriptions = arrays["descriptions"].astype(str)

        assert x.ndim == 3 and x.shape[1:] == (args.window_size, 1), x.shape
        assert y.shape == (len(x),), (x.shape, y.shape)
        assert record_ids.shape == (len(x),), (x.shape, record_ids.shape)
        assert descriptions.shape == (len(x),), (x.shape, descriptions.shape)
        assert np.isfinite(x).all(), f"Non-finite signal value in {split}"
        assert len(y) == 0 or (int(y.min()) >= 0 and int(y.max()) < len(CLASS_ORDER))
        assert all(text.startswith("Patient information:") for text in descriptions)

        counts = Counter(CLASS_ORDER[int(label)] for label in y)
        record_sets[split] = set(record_ids.tolist())
        print(
            f"{split:>5}: samples={len(y):6d}, shape={x.shape}, "
            f"records={len(record_sets[split]):2d}, counts={dict(counts)}"
        )

    assert record_sets["train"].isdisjoint(record_sets["val"])
    assert record_sets["train"].isdisjoint(record_sets["test"])
    assert record_sets["val"].isdisjoint(record_sets["test"])

    metadata_path = processed / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata["window_size"] == args.window_size
        assert metadata["class_order"] == CLASS_ORDER

    print("OK: train, validation, and test records are disjoint.")
    print("OK: patient-information descriptions are present.")


if __name__ == "__main__":
    main()
