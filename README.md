# AI for Healthy Climate Adaptation — Synthetic Geospatial SSL (Harvard PostDoc)

This repository contains my submission for the homework assignment associated with the Postdoctoral Research Position in AI for Healthy Climate Adaptation.

It provides a fully reproducible, end-to-end pipeline that:

1. Generates a deterministic synthetic geospatial dataset  
2. Trains a spatially-aware self-supervised model (GeoModRank)  
3. Extracts region embeddings  
4. Evaluates embeddings on a downstream interpolation task  

All reported results in `report.pdf` are reproducible from this repository using the commands below.

---

# Quick Reproduction (End-to-End)

From a clean clone:

```
git clone <repo>
cd healthy-climate-ssl

conda create -n havard_postdoc python=3.10
conda activate havard_postdoc
pip install -r requirements.txt

./run_all.sh
```

This executes the full pipeline and reproduces all results reported in `report.pdf`.

All outputs are written to:

```
data/v1_seed7/
```

---

# 1. Environment Setup

```
conda create -n havard_postdoc python=3.10
conda activate havard_postdoc
pip install -r requirements.txt
```

Python version used: 3.10  
Frameworks: PyTorch, PyTorch Geometric  

---

# 2. Generate Synthetic Dataset

Dataset configuration is stored in:

```
configs/dataset.yaml
```

Reported results use:

- Seed: 7
- Version directory: `data/v1_seed7/`

To generate the dataset:

```
python -m src.data.generate
```

This produces:

- regions.pt
- features.pt
- masks.pt
- targets.pt
- meta.json

All generation is deterministic given the seed in the config.

---

# 3. Build Spatial kNN Graph

```
python -m src.data.graph data/v1_seed7 configs/dataset.yaml 7
```

This constructs:

- graph.pt (edge_index, edge_weight, k, seed)

Graph construction is deterministic.

---

# 4. Train Self-Supervised Model (GeoModRank)

```
python -m src.training.train_ssl data/v1_seed7
```

This:

- Trains GeoModRank using masked multimodal reconstruction  
- Applies Laplacian smoothness regularization  
- Saves:
  - geomodrank.pt (checkpoint)
  - embeddings.pt
  - train_ssl_meta.json

Embeddings are 192-dimensional region representations.

---

# 5. Extract Region Embeddings

Embeddings are automatically exported during training and saved as:

```
data/v1_seed7/embeddings.pt
```

Each row corresponds to one region.

---

# 6. Downstream Interpolation Evaluation

```
python -m src.downstream.evaluate data/v1_seed7
```

This:

- Creates deterministic 70/10/20 train/val/test split (stratified by state_id)
- Trains downstream predictors:
  - Ridge regression
  - Small MLP
- Compares against coordinate-only baselines:
  - IDW
  - kNN
  - NWKR
- Saves:
  - splits_seed7.pt
  - downstream_results_seed7_val.json
  - downstream_results_seed7_test.json

Metrics:
- MSE
- RMSE
- R²

---

# Programmatic Validation

Before training, automated tests run via:

```
pytest
```

Tests verify:

- Schema and tensor shapes
- SSL training stability (no NaNs)
- Embedding export shape and ordering

All tests must pass before training proceeds.

---

# Dataset Design Summary

Regions are defined by planar coordinates `(x, y) ∈ ℝ²`.  
A kNN graph is constructed using Euclidean distance.

Each region contains three modalities:

- Climate-like features  
- Pollution-like features  
- Socio-demographic-like features  

Structured missingness masks are generated per modality.

Two continuous targets (`y1`, `y2`) depend on:

- Smooth spatial latent fields  
- Modality-level aggregates  
- State-level shifts  
- Additive noise  

All components are deterministic given seed 7.

---

# Reproducibility Details

Seed used for reported results: **7**

Key hyperparameters are stored in:

- `configs/dataset.yaml`
- `train_ssl_meta.json`

All randomness is controlled via explicit seeding.

---

# Deliverables Included

This repository contains:

- Full implementation (well-documented code)
- Programmatic validation tests
- End-to-end runnable pipeline
- `report.pdf` (2–3 page research note)
- Reproducible results under fixed seed

---

# Computational Resources

Experiments were conducted on macOS (Apple Silicon, CPU-only PyTorch).  
Dataset size (N=2000) and model size were selected to ensure full reproducibility.

---

# AI Assistance Disclosure

AI-based tools were used for structural suggestions and documentation drafting.  
All modeling decisions, implementation logic, debugging, and evaluation were developed and verified by the author.  
The author assumes full responsibility for the final implementation.