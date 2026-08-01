"""MIT-BIH beat classification dataset for the original medtsllm2 method.

The raw WFDB records are converted by ``scripts/prepare_mitdb.py`` into
beat-centred windows with shape ``[N, history_len, 1]``. Reference beat
annotations are mapped to the five AAMI groups in the fixed order N, S, V, F,
Q. Per-record age/sex text is returned as ``descriptions`` so the original
MedTsLLM patient-information prompt (``prompting.clip = true``) remains active.
"""
from pathlib import Path

import numpy as np

from .ptbxl import ClassificationDataset

AAMI_CLASS_ORDER = ["N", "S", "V", "F", "Q"]
AAMI_CLASS_NAMES = [
    "Normal / bundle-branch / escape beat",
    "Supraventricular ectopic beat",
    "Ventricular ectopic beat",
    "Fusion beat",
    "Unknown / paced beat",
]


class MITDBClassificationDataset(ClassificationDataset):
    """Five-class, beat-level MIT-BIH Arrhythmia classification."""

    supported_tasks = ["classification"]
    n_classes = len(AAMI_CLASS_ORDER)
    class_names = AAMI_CLASS_ORDER

    description = (
        "The MIT-BIH Arrhythmia Database contains 48 half-hour, two-channel "
        "ambulatory ECG recordings sampled at 360 Hz. Each example is a "
        "single beat-centred ECG window whose reference annotation is mapped "
        "to one of five AAMI groups: N, S, V, F, or Q."
    )
    task_description = (
        "Classify the ECG beat into one of five AAMI beat categories: "
        "N (normal-like), S (supraventricular ectopic), V (ventricular "
        "ectopic), F (fusion), or Q (unknown or paced)."
    )

    def _processed_path(self, split: str) -> Path:
        root = Path(self.dataset_config.get("root", "data/mitdb"))
        if not root.is_absolute():
            root = Path(__file__).resolve().parent.parent / root
        processed_dir = self.dataset_config.get("processed_dir", "processed")
        return root / processed_dir / f"{split}.npz"

    def get_data(self, split=None):
        split = split or self.split
        path = self._processed_path(split)
        if not path.exists():
            raise FileNotFoundError(
                f"Processed MIT-BIH split not found: {path}\n"
                "Run: python3 scripts/prepare_mitdb.py --root data/mitdb --download"
            )

        with np.load(path, allow_pickle=False) as arrays:
            x = arrays["x"].astype(np.float32, copy=False)
            y = arrays["y"].astype(np.int64, copy=False)
            if "descriptions" in arrays.files:
                descriptions = arrays["descriptions"].astype(str).tolist()
            else:
                descriptions = ["Patient information: not available."] * len(y)

        expected_len = int(self.history_len)
        if x.ndim != 3 or x.shape[1:] != (expected_len, 1):
            raise ValueError(
                f"Expected MIT-BIH shape [N, {expected_len}, 1], but {path} "
                f"contains {x.shape}. Re-run preprocessing with "
                f"--window-size {expected_len}."
            )
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError(f"Invalid labels in {path}: x={x.shape}, y={y.shape}")
        if len(descriptions) != len(y):
            raise ValueError(
                f"Invalid descriptions in {path}: labels={len(y)}, "
                f"descriptions={len(descriptions)}"
            )
        if len(y) and (int(y.min()) < 0 or int(y.max()) >= self.n_classes):
            raise ValueError(
                f"Labels in {path} must be in [0, {self.n_classes - 1}]."
            )

        return {"data": x, "labels": y, "descriptions": descriptions}


mitdb_datasets = {
    "classification": MITDBClassificationDataset,
}
