"""MIT-BIH beat classification dataset for the MedTsLLM classification method.

The raw WFDB records are converted by ``scripts/prepare_mitdb.py`` into
beat-centred windows with shape [N, history_len, 1]. Beat annotations are
mapped to the five AAMI groups in the fixed order N, S, V, F, Q.

To match the patient-information prompt used by the PTB-XL MedTsLLM
configuration, this loader reads age and sex from each MIT-BIH WFDB header
and returns one patient description for every beat.
"""

from pathlib import Path
import re

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
    supported_tasks = ["classification"]
    n_classes = len(AAMI_CLASS_ORDER)
    class_names = AAMI_CLASS_ORDER

    description = (
        "The MIT-BIH Arrhythmia Database contains 48 half-hour, two-channel "
        "ambulatory ECG recordings sampled at 360 Hz. For this task, each "
        "example is a single beat-centred ECG window and the beat annotation "
        "is mapped to one of five AAMI groups: N, S, V, F, or Q."
    )
    task_description = (
        "Classify the ECG beat into one of five AAMI beat categories: "
        "N (normal-like), S (supraventricular ectopic), V (ventricular "
        "ectopic), F (fusion), or Q (unknown/paced)."
    )

    def _root(self):
        root = Path(self.dataset_config.get("root", "data/mit-bih"))
        if not root.is_absolute():
            root = Path(__file__).resolve().parent.parent / root
        return root

    def _processed_path(self, split):
        processed_dir = self.dataset_config.get("processed_dir", "processed")
        return self._root() / processed_dir / f"{split}.npz"

    @staticmethod
    def _age_sex_from_comments(comments):
        """Extract age and sex without using labels or diagnosis information."""
        age = None
        sex = None

        for comment in comments or []:
            text = str(comment).strip().lstrip("#").strip()
            if not text:
                continue

            # Typical MIT-BIH comment: "69 M 1085 1629 x1"
            match = re.search(
                r"(?<!\d)(\d{1,3}|\?+)\s+([MFmf]|male|female)(?!\w)",
                text,
            )
            if match:
                if match.group(1).isdigit():
                    candidate_age = int(match.group(1))
                    if 0 < candidate_age < 120:
                        age = candidate_age
                sex_token = match.group(2).lower()
                sex = "male" if sex_token in {"m", "male"} else "female"
                break

            age_match = re.search(r"\bage\s*[:=]?\s*(\d{1,3})\b", text, re.I)
            if age_match:
                candidate_age = int(age_match.group(1))
                if 0 < candidate_age < 120:
                    age = candidate_age

            sex_match = re.search(
                r"\b(?:sex|gender)\s*[:=]?\s*(male|female|m|f)\b",
                text,
                re.I,
            )
            if sex_match:
                sex_token = sex_match.group(1).lower()
                sex = "male" if sex_token in {"m", "male"} else "female"

        return age, sex

    def _record_description(self, record_id):
        """Create the same kind of patient-information text used for PTB-XL."""
        try:
            import wfdb

            header = wfdb.rdheader(str(self._root() / "raw" / str(record_id)))
            age, sex = self._age_sex_from_comments(
                getattr(header, "comments", None)
            )
        except Exception:
            age, sex = None, None

        parts = []
        if age is not None:
            parts.append(f"age {age}")
        if sex is not None:
            parts.append(sex)

        info = ", ".join(parts) if parts else "not available"
        return f"Patient information: {info}."

    def get_data(self, split=None):
        split = split or self.split
        path = self._processed_path(split)
        if not path.exists():
            raise FileNotFoundError(
                f"Processed MIT-BIH split not found: {path}\n"
                "Run: python3 scripts/prepare_mitdb.py "
                f"--root {self._root()}"
            )

        with np.load(path, allow_pickle=False) as data:
            x = data["x"].astype(np.float32, copy=False)
            y = data["y"].astype(np.int64, copy=False)
            record_ids = (
                data["record_ids"].astype(str, copy=False)
                if "record_ids" in data.files
                else None
            )

        expected_len = int(self.history_len)
        if x.ndim != 3 or x.shape[1] != expected_len or x.shape[2] != 1:
            raise ValueError(
                f"Expected MIT-BIH data shape [N, {expected_len}, 1], "
                f"but {path} contains {x.shape}. Re-run preprocessing with "
                f"--window-size {expected_len}."
            )
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError(
                f"Invalid labels in {path}: x={x.shape}, y={y.shape}"
            )
        if len(y) and (y.min() < 0 or y.max() >= self.n_classes):
            raise ValueError(
                f"Labels in {path} must be in [0, {self.n_classes - 1}]."
            )

        result = {"data": x, "labels": y}

        # The MedTsLLM PTB-XL decoder uses prompting.clip=true. Supply an
        # equivalent patient-information field for MIT-BIH without label leakage.
        if record_ids is not None:
            unique_ids = sorted(set(record_ids.tolist()))
            description_by_record = {
                rid: self._record_description(rid) for rid in unique_ids
            }
            result["descriptions"] = [
                description_by_record[rid] for rid in record_ids.tolist()
            ]

        return result


mitdb_datasets = {
    "classification": MITDBClassificationDataset,
}
