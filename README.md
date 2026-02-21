# AI for Healthy Climate Adaptation — Synthetic Geospatial SSL (Havard PostDoc)


This repository contains my submission for the homework assignment associated with the Postdoctoral Research Position in AI for Healthy Climate Adaptation.

The objective is to construct an end-to-end pipeline that:

1. Generates a deterministic synthetic geospatial dataset with multimodal structure  
2. Learns spatially-aware representations via self-supervised training  
3. Evaluates learned embeddings on a region-level interpolation task  

The design is motivated by geospatial foundation modeling (e.g., PDFM), but implemented in a controlled synthetic setting.

---

## Repository Structure

```
.
├── configs/
│   └── dataset.yaml
├── src/
│   ├── data/
│   │   ├── generate.py
│   │   ├── graph.py
│   │   └── plot_graph.py
│   └── utils/
│       └── seed.py
├── data/
│   └── v1_seed7/
├── README.md
└── report.pdf  (to be added)
```

---

## Part A — Synthetic Dataset

### Spatial Structure

Regions are defined by planar 2D coordinates `(x, y) ∈ ℝ²`.  
Distances are Euclidean. A kNN graph is constructed from these coordinates.

To introduce realistic heterogeneity, regions are clustered into `K_states` via KMeans, producing a grouping variable `state_id`.

### Modalities

Each region contains three semantically distinct feature blocks:

- **Climate-like features**
- **Pollution-like features**
- **Socio-demographic-like features**

Each modality has its own dimensionality and noise structure.

### Missingness

Modality-specific missingness masks are explicitly generated and stored.  
Missing values are masked and handled explicitly (never treated as ordinary numeric values).

### Targets

Two continuous region-level targets (`y1`, `y2`) are generated as functions of:

- Smooth latent spatial fields  
- Modality-level aggregates  
- State-level shifts  
- Idiosyncratic noise  

This ensures that interpolation is meaningful but non-trivial.

### Determinism

Given a fixed seed and configuration, dataset generation is fully deterministic.

---

## Data Format

All artifacts are stored as PyTorch `.pt` files.

```
data/v1_seed7/
```

**regions.pt**
- `coords` — FloatTensor `[N, 2]`
- `state_id` — LongTensor `[N]`

**features.pt**
- `climate` — `[N, d1]`
- `pollution` — `[N, d2]`
- `socio` — `[N, d3]`

**masks.pt**
- Boolean masks aligned with each modality

**targets.pt**
- `y1`, `y2` — FloatTensor `[N]`

**graph.pt**
- `edge_index` — LongTensor `[2, E]`
- `edge_weight`
- `k`, `seed`

**meta.json**
- Seed and full configuration

---

## Running the Pipeline

### 1. Generate Dataset

```
python -m src.data.generate
```

### 2. Build kNN Graph

```
python -m src.data.graph data/v1_seed7 configs/dataset.yaml 7
```

### 3. Visualize Graph

```
python src/data/plot_graph.py
```

---

## Part B — Self-Supervised Representation Learning (to be completed)

A spatially-aware self-supervised model will be trained to reconstruct masked multimodal inputs.  
The model will produce modality-aware region embeddings suitable for downstream evaluation.

---

## Part C — Downstream Interpolation (to be completed)

Evaluation protocol:

- 70 / 10 / 20 region split (train / validation / test)
- Stratified by `state_id` when feasible
- Downstream predictor: ridge regression or small MLP
- Baseline: coordinate-only spatial regression (kNN or IDW)

Metrics reported per target:
- MSE
- RMSE
- R²

---

## Reproducibility

All components are seed-controlled and deterministic.  
Seeds are stored in `meta.json`, and graph parameters are stored in `graph.pt`.

---

## Computational Resources

Experiments were conducted on macOS (Apple Silicon, CPU-only PyTorch).  
Model and dataset sizes were intentionally kept modest to ensure reproducibility.

---

## AI Assistance Disclosure

AI-based tools were used for code refactoring, structural suggestions, and documentation drafting.  
All final code was manually reviewed, verified for correctness, and tested for determinism.  
The author remains fully responsible for the final implementation.
