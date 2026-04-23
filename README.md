# AI for Healthy Climate Adaptation — Synthetic Geospatial SSL (Harvard PostDoc)


This repository implements a spatially-aware self-supervised learning framework for synthetic geospatial data, integrating spatial graph structure with multimodal masked reconstruction to learn region-level representations under strict deterministic controls for full reproducibility.

This repository was developed as part of a research project on spatially-aware self-supervised learning for geospatial data, originally prepared in the context of a postdoctoral research evaluation in AI for climate adaptation.



## GeoModRank Framework Overview

![GeoModRank Framework](geoModRank_framework.jpg)

*Figure 1: End-to-end pipeline for deterministic synthetic geospatial data generation, spatial kNN graph construction, self-supervised training (GeoModRank), embedding extraction, and downstream interpolation evaluation.*

The framework implements a fully reproducible, end-to-end pipeline that:

1. Generates a deterministic synthetic geospatial dataset  
2. Constructs a spatial kNN graph over regions  
3. Trains a spatially-aware self-supervised model (GeoModRank)  
4. Extracts region-level embeddings  
5. Evaluates embeddings on a downstream spatial interpolation task  

All results reported in the accompanying [report.pdf](report.pdf) are reproducible using the commands below.

## Quick Reproduction 

From a clean clone:

```bash
git clone <repo>
cd healthy-climate-ssl

conda create -n harvard_postdoc python=3.10
conda activate harvard_postdoc
pip install -r requirements.txt

./run_all.sh
```

This script runs the full pipeline: validation tests, dataset generation, spatial kNN graph construction, self-supervised training (GeoModRank), embedding export, and downstream interpolation.

All outputs are written to:

```
data/v1_seed7/
```
All experiments are fully deterministic under seed **7**.


---

### Environment Setup

```bash
conda create -n harvard_postdoc python=3.10
conda activate harvard_postdoc
pip install -r requirements.txt
```

- `Python`: **3.10**  
- `Frameworks`: **PyTorch**, **PyTorch Geometric**
- `Hardware`: CPU-only (8-core, 8GB RAM); no GPU used.

---


### Implementation Summary

The pipeline includes:

1. Deterministic synthetic geospatial data with multimodal features  
2. A spatially-aware self-supervised model (GeoModRank)  
3. 192-dimensional region embeddings with a 70/10/20 interpolation split  
4. Downstream evaluation (Ridge / MLP) against coordinate-based baselines  

All experiments are seeded for reproducibility, and validation tests (pytest) verify data integrity and training.

---

### Deliverables

This repository includes:

- Complete implementation  
- Validation tests  
- End-to-end pipeline  
- `report.pdf` (3-page research note)  

---
### AI Assistance Disclosure

AI-based tools (including large language models such as ChatGPT) were used for structural suggestions, documentation refinement, and minor refactoring.

All modeling logic, mathematical formulation, dataset generation, and evaluation procedures were independently implemented, executed, and verified. The author assumes full responsibility for correctness and reproducibility.
