#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-$PWD}"

if [[ ! -f "$REPO_DIR/train.py" || ! -d "$REPO_DIR/models" ]]; then
  echo "ERROR: $REPO_DIR does not look like the Medtsllm-mitbih repository."
  echo "Run this from the repository root, or pass the repository path:"
  echo "  bash apply_exact_method.sh /path/to/Medtsllm-mitbih"
  exit 1
fi

cd "$REPO_DIR"

BACKUP_DIR="backup_before_exact_method_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR/models" "$BACKUP_DIR/tasks" \
         "$BACKUP_DIR/datasets" "$BACKUP_DIR/configs/datasets"

for f in \
  models/medtsllm.py \
  tasks/classification.py \
  tasks/__init__.py \
  datasets/__init__.py \
  datasets/mitdb.py \
  configs/datasets/mitdb_decoder.toml
do
  if [[ -f "$f" ]]; then
    cp -a "$f" "$BACKUP_DIR/$f"
  fi
done

download_file() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -LfsS "$url" -o "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$output" "$url"
  else
    echo "ERROR: curl or wget is required."
    exit 1
  fi
}

echo "[1/6] Restoring the exact medtsllm2 base model..."
download_file \
  "https://raw.githubusercontent.com/kkianamm/medtsllm2/main/models/medtsllm.py" \
  "models/medtsllm.py"

echo "[2/6] Applying medtsllm2's classification changes to the model..."
patch --batch --forward -p0 < "$PACKAGE_DIR/patches/models_medtsllm_classification.patch"

echo "[3/6] Installing exact classification task and registrations..."
cp "$PACKAGE_DIR/tasks/classification.py" tasks/classification.py
cp "$PACKAGE_DIR/tasks/__init__.py" tasks/__init__.py
cp "$PACKAGE_DIR/datasets/__init__.py" datasets/__init__.py

echo "[4/6] Installing the MIT-BIH-only loader and configuration..."
cp "$PACKAGE_DIR/datasets/mitdb.py" datasets/mitdb.py
cp "$PACKAGE_DIR/configs/datasets/mitdb_decoder.toml" \
   configs/datasets/mitdb_decoder.toml

echo "[5/6] Ensuring the MIT-BIH WFDB dependency is installed..."
if [[ -f requirements-mitdb.txt ]]; then
  if ! grep -Eq '^[[:space:]]*wfdb([=<>!~]|$)' requirements-mitdb.txt; then
    printf '\nwfdb==4.1.2\n' >> requirements-mitdb.txt
  fi
else
  cat > requirements-mitdb.txt <<'EOF'
# Compatible with this repository's numpy < 1.25 constraint.
wfdb==4.1.2
EOF
fi

echo "[6/6] Running static verification..."
python3 "$PACKAGE_DIR/verify_exact_method.py" "$REPO_DIR"

echo
echo "Conversion complete."
echo "Backup: $REPO_DIR/$BACKUP_DIR"
echo
echo "Next commands:"
echo "  pip install -r requirements-mitdb.txt"
echo "  python3 scripts/check_mitdb.py --root data/mit-bih"
echo "  python3 train.py configs/datasets/mitdb_decoder.toml"
