# AI for Healthy Climate Adaptation — Synthetic Geospatial SSL (Harvard PostDoc)

This repository contains my submission for the homework assignment associated with the Postdoctoral Research Position in AI for Healthy Climate Adaptation.

It provides a fully reproducible, end-to-end pipeline that:

1. Generates a deterministic synthetic geospatial dataset  
2. Trains a spatially-aware self-supervised model (GeoModRank)  
3. Extracts region embeddings  
4. Evaluates embeddings on a downstream interpolation task  

All results reported in `report.pdf` are reproducible from this repository using the commands below.

---

# Quick Reproduction (End-to-End)

From a clean clone:

```bash
git clone <repo>
cd healthy-climate-ssl

conda create -n harvard_postdoc python=3.10
conda activate harvard_postdoc
pip install -r requirements.txt

chmod +x run_all.sh
./run_all.sh
```

This runs:

- Programmatic checks (pytest)  
- Dataset generation  
- kNN graph construction  
- Self-supervised training (GeoModRank)  
- Embedding export  
- Deterministic train/val/test split  
- Downstream interpolation (validation + final test)  

All outputs are written to:

```
data/v1_seed7/
```

---

## 1. Environment Setup

```bash
conda create -n harvard_postdoc python=3.10
conda activate harvard_postdoc
pip install -r requirements.txt
```

Python version used: **3.10**  
Frameworks: **PyTorch**, **PyTorch Geometric**

---

## 2. Generate Synthetic Dataset

Dataset configuration:

```
configs/dataset.yaml
```

Reported results use:

- Seed: **7**
- Output directory: `data/v1_seed7/`

Generate the dataset:

```bash
python -m src.data.generate
```

Artifacts produced:

- `regions.pt`
- `features.pt`
- `masks.pt`
- `targets.pt`
- `meta.json`

Dataset generation is deterministic given the seed.

---

## 3. Build Spatial kNN Graph

```bash
python -m src.data.graph data/v1_seed7 configs/dataset.yaml 7
```

Produces:

- `graph.pt` (edge_index, edge_weight, k, seed)

Graph construction is deterministic.

---

## 4. Train Self-Supervised Model (GeoModRank)

```bash
python -m src.training.train_ssl data/v1_seed7
```

This:

- Trains GeoModRank via masked multimodal reconstruction  
- Applies Laplacian smoothness regularization  
- Saves:
  - `geomodrank.pt`
  - `embeddings.pt`
  - `train_ssl_meta.json`

Embeddings are 192-dimensional region representations.

---

## 5. Extract Region Embeddings

Embeddings are automatically exported during training:

```
data/v1_seed7/embeddings.pt
```

Each row corresponds to one region in the dataset.

---

## 6. Downstream Interpolation Evaluation

```bash
python -m src.downstream.evaluate data/v1_seed7
```

This:

- Creates deterministic 70/10/20 train/val/test split (stratified by `state_id`)
- Trains downstream predictors:
  - Ridge regression
  - Small MLP
- Compares against coordinate-only baselines:
  - Mean
  - IDW
  - kNN
  - NWKR
- Saves:
  - `splits_seed7.pt`
  - `downstream_results_seed7_val.json`
  - `downstream_results_seed7_test.json`

Metrics reported:
- MSE
- RMSE
- R²

---

### Programmatic Validation

Automated tests run before training:

```bash
pytest
```

Tests verify:

- Schema and tensor shapes  
- SSL training stability (no NaNs)  
- Embedding export shape and ordering  

All tests must pass before training proceeds.

---

### Dataset Design Summary

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

All components are deterministic under seed 7.

---

### Reproducibility Details

Seed used for reported results: **7**  
Python version: **3.10**

Key hyperparameters and configuration stored in:

- `configs/dataset.yaml`
- `train_ssl_meta.json`

All randomness is controlled via explicit seeding.

---

### Deliverables Included

This repository contains:

- Full implementation (well-commented and organized)
- Programmatic validation tests
- End-to-end runnable pipeline
- `report.pdf` (2–3 page research note)
- Deterministic reproduction of reported results

---

### Computational Resources

Experiments were conducted on macOS with a 3.2 GHz 8-core CPU and 8 GB RAM.  
No GPU acceleration was used.  
Dataset size (N=2000) and embedding dimension (192) were selected to ensure reproducibility.

---

### AI Assistance Disclosure

AI-based tools (large language models) were used for structural suggestions, documentation drafting, and minor code refactoring.

No core modeling logic, mathematical formulation, data generation process, or evaluation procedure was automatically generated without manual verification. All results were independently executed, validated, and tested for determinism.

The author assumes full responsibility for correctness and reproducibility.