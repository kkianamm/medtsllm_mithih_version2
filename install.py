#!/usr/bin/env python3
"""Install the MIT-BIH dataset-only adaptation into a medtsllm2 checkout.

Usage:
    python3 /path/to/MedTsLLM2_MITBIH/install.py /path/to/medtsllm2
"""
from __future__ import annotations

import argparse
import compileall
import shutil
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(
            f"Could not patch {label}: expected source text was not found. "
            "Use a clean checkout of kkianamm/medtsllm2 main."
        )
    return text.replace(old, new, 1)


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".before_mitdb.bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def patch_model(path: Path) -> None:
    backup(path)
    text = path.read_text(encoding="utf-8")

    old_supported = (
        'supported_tasks = ["forecasting", "reconstruction", '
        '"anomaly_detection", "semantic_segmentation", "segmentation", '
        '"pretraining"]'
    )
    new_supported = (
        'supported_tasks = ["forecasting", "reconstruction", '
        '"anomaly_detection", "semantic_segmentation", "segmentation", '
        '"classification", "pretraining"]'
    )
    if '"classification"' not in text[text.find("supported_tasks"):text.find("supported_modes")]:
        text = replace_once(
            text, old_supported, new_supported,
            "models/medtsllm.py supported_tasks",
        )

    if not (
        'elif self.task == "classification":' in text
        and 'self.n_outputs = self.n_classes' in text
    ):
        old = (
            '        elif self.task == "segmentation":\n'
            '            self.n_outputs_per_step = 1\n'
            '            assert self.config.tasks.segmentation.mode in '
            '["boundary-prediction", "steps-to-boundary"]\n'
            '        else:\n'
            '            raise ValueError(f"Task {self.task} is not supported.")\n'
            '        self.n_outputs = self.n_outputs_per_step * self.pred_len\n'
        )
        new = (
            '        elif self.task == "segmentation":\n'
            '            self.n_outputs_per_step = 1\n'
            '            assert self.config.tasks.segmentation.mode in '
            '["boundary-prediction", "steps-to-boundary"]\n'
            '        elif self.task == "classification":\n'
            '            # One K-class prediction for the whole input sequence.\n'
            '            self.n_outputs_per_step = self.n_classes\n'
            '        else:\n'
            '            raise ValueError(f"Task {self.task} is not supported.")\n'
            '        self.n_outputs = self.n_outputs_per_step * self.pred_len\n'
            '        if self.task == "classification":\n'
            '            self.n_outputs = self.n_classes\n'
        )
        text = replace_once(
            text, old, new,
            "models/medtsllm.py classification output dimensions",
        )

    forward_start = text.find("def forward(self, inputs):")
    forward_end = text.find("def encode_ts", forward_start)
    forward_region = text[forward_start:forward_end]
    if 'elif self.task == "classification":' not in forward_region:
        old = (
            '            elif self.task == "segmentation":\n'
            '                if self.config.tasks.segmentation.mode == '
            '"boundary-prediction":\n'
            '                    pred = F.sigmoid(pred)\n\n'
            '        return pred\n'
        )
        new = (
            '            elif self.task == "segmentation":\n'
            '                if self.config.tasks.segmentation.mode == '
            '"boundary-prediction":\n'
            '                    pred = F.sigmoid(pred)\n'
            '            elif self.task == "classification":\n'
            '                pred = F.softmax(pred, dim=-1)\n\n'
            '        return pred\n'
        )
        text = replace_once(
            text, old, new,
            "models/medtsllm.py classification evaluation activation",
        )

    predict_start = text.find("def predict(self, inputs):")
    predict_end = text.find("def build_prompt", predict_start)
    predict_region = text[predict_start:predict_end]
    if "[bs, n_classes]" not in predict_region:
        old = (
            '        dec_out = self.output_projection(dec_out)       '
            '# [bs, pred_len * n_features]\n'
            '        if self.covariate_mode == "independent":\n'
        )
        new = (
            '        dec_out = self.output_projection(dec_out)       '
            '# [bs, pred_len * n_features]\n\n'
            '        if self.task == "classification":\n'
            '            if self.covariate_mode in ["independent", "merge-end"]:\n'
            '                raise NotImplementedError(\n'
            '                    "Classification supports covariate modes that preserve the batch dimension."\n'
            '                )\n'
            '            return dec_out  # [bs, n_classes]\n\n'
            '        if self.covariate_mode == "independent":\n'
        )
        text = replace_once(
            text, old, new,
            "models/medtsllm.py classification prediction head",
        )

    task_start = text.find("def get_task_description")
    task_end = text.find("def load_pretrained", task_start)
    task_region = text[task_start:task_end]
    if 'elif self.task == "classification":' not in task_region:
        old = (
            '        elif self.task == "segmentation":\n'
            '            self.task_description = f"Identify the change points '
            'in the past {self.seq_len} steps of data to segment the sequence."\n'
            '        else:\n'
            '            raise ValueError(f"Task {self.task} is not supported.")\n'
        )
        new = (
            '        elif self.task == "segmentation":\n'
            '            self.task_description = f"Identify the change points '
            'in the past {self.seq_len} steps of data to segment the sequence."\n'
            '        elif self.task == "classification":\n'
            '            self.task_description = f"Classify the entire sequence '
            'of {self.seq_len} steps into one of the diagnostic classes using '
            'the following information."\n'
            '        else:\n'
            '            raise ValueError(f"Task {self.task} is not supported.")\n'
        )
        text = replace_once(
            text, old, new,
            "models/medtsllm.py classification task prompt",
        )

    path.write_text(text, encoding="utf-8")


def patch_tasks_init(path: Path) -> None:
    backup(path)
    text = path.read_text(encoding="utf-8")
    if "from .classification import ClassificationTask" not in text:
        marker = "from .semantic_segmentation import SemanticSegmentationTask\n"
        if marker not in text:
            raise RuntimeError("Could not patch tasks/__init__.py import.")
        text = text.replace(
            marker,
            marker + "from .classification import ClassificationTask\n",
            1,
        )
    if '"classification": ClassificationTask,' not in text:
        marker = '    "semantic_segmentation": SemanticSegmentationTask,\n'
        if marker not in text:
            raise RuntimeError("Could not patch tasks/__init__.py lookup.")
        text = text.replace(
            marker,
            marker + '    "classification": ClassificationTask,\n',
            1,
        )
    path.write_text(text, encoding="utf-8")


def patch_datasets_init(path: Path) -> None:
    backup(path)
    text = path.read_text(encoding="utf-8")
    if "from .mitdb import mitdb_datasets" not in text:
        marker = "from .dreams import dreams_datasets\n"
        if marker not in text:
            raise RuntimeError("Could not patch datasets/__init__.py import.")
        text = text.replace(
            marker,
            marker + "from .mitdb import mitdb_datasets\n",
            1,
        )
    if '"MIT-BIH": mitdb_datasets,' not in text:
        marker = '    "dreams": dreams_datasets,\n'
        if marker not in text:
            raise RuntimeError("Could not patch datasets/__init__.py lookup.")
        text = text.replace(
            marker,
            marker + '    "MIT-BIH": mitdb_datasets,\n',
            1,
        )
    path.write_text(text, encoding="utf-8")


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"installed {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", nargs="?", default=".", type=Path)
    args = parser.parse_args()

    bundle = Path(__file__).resolve().parent
    repo = args.repo.resolve()

    required = [
        repo / "models/medtsllm.py",
        repo / "tasks/__init__.py",
        repo / "tasks/classification.py",
        repo / "datasets/__init__.py",
        repo / "datasets/ptbxl.py",
        repo / "requirements.txt",
        repo / "train.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "This does not look like kkianamm/medtsllm2. Missing:\n"
            + "\n".join(missing)
        )

    copy_file(bundle / "datasets/mitdb.py", repo / "datasets/mitdb.py")
    copy_file(bundle / "scripts/prepare_mitdb.py", repo / "scripts/prepare_mitdb.py")
    copy_file(bundle / "scripts/check_mitdb.py", repo / "scripts/check_mitdb.py")
    copy_file(
        bundle / "configs/datasets/mitdb_decoder.toml",
        repo / "configs/datasets/mitdb_decoder.toml",
    )

    patch_model(repo / "models/medtsllm.py")
    patch_tasks_init(repo / "tasks/__init__.py")
    patch_datasets_init(repo / "datasets/__init__.py")

    requirements_path = repo / "requirements.txt"
    requirements = requirements_path.read_text(encoding="utf-8")
    if "wfdb==4.1.2" not in requirements.replace(" ", ""):
        with requirements_path.open("a", encoding="utf-8") as handle:
            if requirements and not requirements.endswith("\n"):
                handle.write("\n")
            handle.write(
                "\n# MIT-BIH WFDB reader (compatible with numpy < 1.25)\n"
                "wfdb==4.1.2\n"
            )

    ok = compileall.compile_file(str(repo / "models/medtsllm.py"), quiet=1)
    ok = compileall.compile_file(str(repo / "tasks/__init__.py"), quiet=1) and ok
    ok = compileall.compile_file(str(repo / "datasets/__init__.py"), quiet=1) and ok
    ok = compileall.compile_file(str(repo / "datasets/mitdb.py"), quiet=1) and ok
    ok = compileall.compile_file(str(repo / "scripts/prepare_mitdb.py"), quiet=1) and ok
    ok = compileall.compile_file(str(repo / "scripts/check_mitdb.py"), quiet=1) and ok
    if not ok:
        raise RuntimeError("Python syntax validation failed.")

    print("\nMIT-BIH adaptation installed successfully.")
    print("Next:")
    print("  python3 -m pip install -r requirements.txt")
    print(
        "  python3 scripts/prepare_mitdb.py --root data/mitdb "
        "--download --window-size 512"
    )
    print("  python3 scripts/check_mitdb.py --root data/mitdb --window-size 512")
    print("  python3 train.py configs/datasets/mitdb_decoder.toml")


if __name__ == "__main__":
    main()
