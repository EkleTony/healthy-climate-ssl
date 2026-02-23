#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data/v1_seed7"
SEED=7
EPOCHS=100

# GeoModRank SSL knobs
MASK_RATIO=0.3
NOISE_STD=0.0
LAM_SMOOTH=1e-3

echo "======================================================"
echo "   GeoModRank Self-Supervised Spatial Learning"
echo "   Anthony O. Ekle"
echo "======================================================"
echo "Starting full pipeline..."
echo ""

echo "======================================"
echo " Step 0: Generate Deterministic Synthetic Dataset"
echo "======================================"
# This writes regions.pt, features.pt, masks.pt, targets.pt, meta.json into $DATA_DIR
python -m src.data.generate
echo ""

echo "======================================"
echo " Step 1: Build Spatial kNN Graph"
echo "======================================"
# This writes graph.pt into $DATA_DIR
python -m src.data.graph "$DATA_DIR" configs/dataset.yaml "$SEED"
echo ""

echo "======================================"
echo " Step 2: Running Programmatic PyTest Checks"
echo "======================================"
pytest -v
echo ""

echo "======================================"
echo " Step 3: Train GeoModRank Graph-Based Framework (SSL)"
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
echo " Step 4: Create Stratified Split"
echo "======================================"
python -m src.eval.split \
  --data_dir "$DATA_DIR" \
  --seed "$SEED"
echo ""

echo "======================================"
echo " Step 5: Downstream Interpolation (VAL)"
echo "======================================"
python -m src.training.eval_interpolation \
  --data_dir "$DATA_DIR" \
  --seed "$SEED" \
  --mode val \
  --use_mlp
echo ""

echo "======================================"
echo " Step 6: Downstream Interpolation (TEST — final)"
echo "======================================"
python -m src.training.eval_interpolation \
  --data_dir "$DATA_DIR" \
  --seed "$SEED" \
  --mode test \
  --use_mlp
echo ""

echo "======================================"
echo " Pipeline Completed Successfully"
echo "======================================"