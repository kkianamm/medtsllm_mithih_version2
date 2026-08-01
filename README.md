# Exact medtsllm2 method on MIT-BIH

This package converts `kkianamm/Medtsllm-mitbih` into the classification
method used by `kkianamm/medtsllm2`, while changing only the dataset-specific
parts to MIT-BIH.

## Files changed

1. `models/medtsllm.py`
   - Restored directly from `medtsllm2`.
   - The reference `medtsllm_classification.patch` model changes are applied.
   - The BiomedCoOp import and prototype classifier are removed from the active
     model implementation.

2. `tasks/classification.py`
   - Replaced by the `medtsllm2` cross-entropy classification trainer.

3. `tasks/__init__.py`
   - Registers `ClassificationTask`.

4. `datasets/__init__.py`
   - Keeps PTB-XL registration and adds MIT-BIH registration.

5. `datasets/mitdb.py`
   - Loads the prepared MIT-BIH beat windows.
   - Explicitly declares classification support.
   - Returns the five AAMI classes in fixed order.
   - Extracts age and sex from MIT-BIH WFDB headers so `clip=true` uses the same
     type of patient-information prompt as the PTB-XL configuration.
   - It does not put labels or diagnosis information into the prompt.

6. `configs/datasets/mitdb_decoder.toml`
   - Uses the same model and training hyperparameters as the PTB-XL decoder
     configuration, except for the dataset-specific fields.

The existing `train.py` is intentionally left unchanged. Its JSON saving code
does not alter the model, loss, optimizer, gradients, or predictions.

## Install

Extract this package, then run from the package folder:

```bash
bash apply_exact_method.sh /path/to/Medtsllm-mitbih
```

Or place the package inside the repository and run:

```bash
bash medtsllm2_exact_mitbih/apply_exact_method.sh .
```

## Validate the data

```bash
cd /path/to/Medtsllm-mitbih
pip install -r requirements-mitdb.txt
python3 scripts/check_mitdb.py --root data/mit-bih
```

Your processed NPZ files must contain `record_ids`. The repository's existing
MIT-BIH preparation script already writes them.

## Train

```bash
python3 train.py configs/datasets/mitdb_decoder.toml
```

## Restore the old files

The installer creates a timestamped directory such as:

```text
backup_before_exact_method_20260801_165900/
```

Copy files from that directory back into the repository to undo the conversion.
