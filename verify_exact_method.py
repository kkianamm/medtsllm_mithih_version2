#!/usr/bin/env python3
"""Static checks for the exact medtsllm2-method / MIT-BIH conversion."""

from pathlib import Path
import py_compile
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

    model_path = repo / "models" / "medtsllm.py"
    task_path = repo / "tasks" / "classification.py"
    task_init_path = repo / "tasks" / "__init__.py"
    dataset_init_path = repo / "datasets" / "__init__.py"
    mitdb_path = repo / "datasets" / "mitdb.py"
    config_path = repo / "configs" / "datasets" / "mitdb_decoder.toml"

    for path in (
        model_path,
        task_path,
        task_init_path,
        dataset_init_path,
        mitdb_path,
        config_path,
    ):
        require(path.exists(), f"Missing required file: {path}")

    model = model_path.read_text()
    task = task_path.read_text()
    task_init = task_init_path.read_text()
    dataset_init = dataset_init_path.read_text()
    mitdb = mitdb_path.read_text()

    require('"classification"' in model, "Model does not support classification.")
    require(
        "return dec_out  # [bs, n_classes]" in model,
        "Reference sequence-classification output path is missing.",
    )
    require(
        "biomedcoop" not in model.lower(),
        "BiomedCoOp is still imported or active in models/medtsllm.py.",
    )
    require(
        "self.aux_loss" not in task,
        "Non-reference auxiliary-loss training remains active.",
    )
    require(
        "ClassificationTask" in task_init
        and '"classification": ClassificationTask' in task_init,
        "ClassificationTask is not registered.",
    )
    require(
        "from .mitdb import mitdb_datasets" in dataset_init
        and '"MIT-BIH": mitdb_datasets' in dataset_init,
        "MIT-BIH is not registered.",
    )
    require(
        'supported_tasks = ["classification"]' in mitdb,
        "MIT-BIH loader does not explicitly support classification.",
    )

    with config_path.open("rb") as f:
        config = tomllib.load(f)

    require(config["task"] == "classification", "Wrong task in config.")
    require(config["data"]["dataset"] == "MIT-BIH", "Wrong dataset in config.")
    require(config["training"]["batch_size"] == 16, "Batch size differs from reference.")
    require(
        config["training"]["eval_metric"] == "accuracy",
        "Checkpoint metric differs from reference.",
    )
    require(
        config["models"]["medtsllm"]["prompting"]["clip"] is True,
        "Patient-information prompting is not enabled.",
    )
    require(
        config["setup"]["dtype"] == "mixed",
        "Dtype differs from reference configuration.",
    )

    for path in (model_path, task_path, task_init_path, dataset_init_path, mitdb_path):
        py_compile.compile(str(path), doraise=True)

    processed = repo / config["datasets"]["MIT-BIH"]["root"] /         config["datasets"]["MIT-BIH"]["processed_dir"]
    if processed.exists():
        import numpy as np

        for split in ("train", "val", "test"):
            path = processed / f"{split}.npz"
            require(path.exists(), f"Missing processed split: {path}")
            with np.load(path, allow_pickle=False) as data:
                require("x" in data.files and "y" in data.files, f"Bad NPZ: {path}")
                require(
                    "record_ids" in data.files,
                    f"{path} lacks record_ids needed for patient prompts.",
                )
                x = data["x"]
                y = data["y"]
                require(x.ndim == 3 and x.shape[1:] == (512, 1), f"Bad x shape: {x.shape}")
                require(y.ndim == 1 and len(y) == len(x), f"Bad y shape: {y.shape}")

    print("PASS: exact medtsllm2 classification method with MIT-BIH-only adaptation.")


if __name__ == "__main__":
    main()
