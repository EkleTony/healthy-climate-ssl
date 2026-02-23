#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DATA_DIR="data/v1_seed7"
SEED=7
EPOCHS=100

MASK_RATIO=0.3
NOISE_STD=0.0
LAM_SMOOTH=1e-3

mkdir -p "$DATA_DIR"

echo "======================================================"
echo "   GeoModRank Self-Supervised Spatial Learning"
echo "   Anthony O. Ekle"
echo "======================================================"
echo "Repo root: $REPO_ROOT"
echo "DATA_DIR : $DATA_DIR"
echo "Starting full pipeline..."
echo ""

echo "======================================"
echo " Step 0: Generate Deterministic Synthetic Dataset"
echo "======================================"
python -m src.data.generate --data_dir "$DATA_DIR" --seed "$SEED"
echo ""

echo "======================================"
echo " Step 1: Build Spatial kNN Graph"
echo "======================================"
python -m src.data.graph "$DATA_DIR" configs/dataset.yaml "$SEED"
echo ""

echo "======================================"
echo " Step 2: Train GeoModRank Graph-Based Framework (SSL)"
echo "======================================"
python -m src.training.train_ssl \
  --data_dir "$DATA_DIR" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --mask_ratio "$MASK_RATIO" \
  --noise_std "$NOISE_STD" \
  --use_coords \
  --lam_smooth "$LAM_SMOOTH"
echo ""

echo "======================================"
echo " Verify: required artifacts exist"
echo "======================================"
ls -lh "$DATA_DIR"
test -f "$DATA_DIR/embeddings.pt" || (echo "[ERROR] embeddings.pt not created in $DATA_DIR" && exit 1)
test -f "$DATA_DIR/geomodrank.pt" || (echo "[ERROR] geomodrank.pt not created in $DATA_DIR" && exit 1)
echo "[OK] embeddings.pt and geomodrank.pt exist."
echo ""

echo "======================================"
echo " Step 3: Create Stratified Split"
echo "======================================"
python -m src.eval.split --data_dir "$DATA_DIR" --seed "$SEED"
echo ""

echo "======================================"
echo " Step 4: Downstream Interpolation (VAL)"
echo "======================================"
python -m src.training.eval_interpolation --data_dir "$DATA_DIR" --seed "$SEED" --mode val --use_mlp
echo ""

echo "======================================"
echo " Step 5: Downstream Interpolation (TEST — final)"
echo "======================================"
python -m src.training.eval_interpolation --data_dir "$DATA_DIR" --seed "$SEED" --mode test --use_mlp
echo ""

echo "======================================"
echo " Step 6: Running Programmatic PyTest Checks"
echo "======================================"
pytest -v
echo ""

echo "======================================"
echo " Pipeline Completed Successfully"
echo "======================================"