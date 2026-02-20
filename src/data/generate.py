import os
import json
import yaml
import torch
import numpy as np
from sklearn.cluster import KMeans

from src.utils.seed import seed_all


def smooth_field(coords: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Create a smooth spatial signal over 2D coordinates.
    This mimics large-scale geographic variation (e.g., climate gradients).
    """
    x = coords[:, 0]
    y = coords[:, 1]
    return scale * (np.sin(0.5 * x) + np.cos(0.5 * y))


def generate(seed: int, config_path: str, out_dir: str) -> str:
    """
    Generate a deterministic synthetic geospatial dataset.

    The dataset includes:
      - Region coordinates
      - Multimodal feature blocks (climate, pollution, socio)
      - Explicit missingness masks
      - Two continuous region-level targets

    The function writes all artifacts to disk and returns the dataset path.
    """
    seed_all(seed)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    name = config["name"]
    N = int(config["N"])
    K = int(config["K_states"])

    dims = config["dims"]
    d_clim = int(dims["climate"])
    d_poll = int(dims["pollution"])
    d_soc = int(dims["socio"])

    miss_cfg = config["missingness"]
    miss_clim = float(miss_cfg["climate_rate"])
    miss_poll = float(miss_cfg["pollution_rate"])
    miss_soc = float(miss_cfg["socio_rate"])

    # ------------------------------------------------------------------
    # 1) Generate spatial coordinates (clustered geography)
    # ------------------------------------------------------------------
    centers = np.random.uniform(-5, 5, size=(K, 2))
    cluster_ids = np.random.choice(K, size=N)
    coords = centers[cluster_ids] + np.random.normal(0, 0.8, size=(N, 2))

    # Derive a grouping variable to introduce regional heterogeneity
    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
    state_id = kmeans.fit_predict(coords)

    # ------------------------------------------------------------------
    # 2) Construct latent spatial fields
    # ------------------------------------------------------------------
    latent_global = smooth_field(coords, scale=1.0)
    latent_secondary = smooth_field(coords, scale=0.6)

    # ------------------------------------------------------------------
    # 3) Build modality feature blocks
    # ------------------------------------------------------------------

    # Climate: smooth large-scale spatial signal + mild noise
    climate = np.stack(
        [latent_global + np.random.normal(0, 0.30, N) for _ in range(d_clim)],
        axis=1,
    )

    # Pollution: influenced by multiple spatial components + stronger noise
    pollution = np.stack(
        [
            latent_secondary + 0.5 * latent_global
            + np.random.normal(0, 0.40, N)
            for _ in range(d_poll)
        ],
        axis=1,
    )

    # Socio-demographic: state-level shift + local variation
    state_shift = np.random.normal(0, 0.6, size=K)
    socio = np.stack(
        [
            state_shift[state_id] + np.random.normal(0, 0.50, N)
            for _ in range(d_soc)
        ],
        axis=1,
    )

    # ------------------------------------------------------------------
    # 4) Introduce modality-specific missingness
    # ------------------------------------------------------------------
    M_clim = np.random.rand(N, d_clim) > miss_clim
    M_poll = np.random.rand(N, d_poll) > miss_poll
    M_soc = np.random.rand(N, d_soc) > miss_soc

    # Zero-out missing entries (explicit masks are stored separately)
    climate *= M_clim
    pollution *= M_poll
    socio *= M_soc

    # ------------------------------------------------------------------
    # 5) Define region-level regression targets
    # ------------------------------------------------------------------
    # Targets depend on:
    #   - smooth spatial structure
    #   - modality aggregates
    #   - state-level heterogeneity
    #   - idiosyncratic noise
    y1 = (
        latent_global
        + 0.30 * climate.mean(axis=1)
        + 0.20 * pollution.mean(axis=1)
        + 0.15 * state_shift[state_id]
        + np.random.normal(0, 0.50, N)
    )

    y2 = (
        latent_secondary
        + 0.40 * socio.mean(axis=1)
        + 0.10 * climate.mean(axis=1)
        - 0.10 * state_shift[state_id]
        + np.random.normal(0, 0.50, N)
    )

    # ------------------------------------------------------------------
    # 6) Persist dataset artifacts
    # ------------------------------------------------------------------
    dataset_path = os.path.join(out_dir, f"{name}_seed{seed}")
    os.makedirs(dataset_path, exist_ok=True)

    torch.save(
        {
            "coords": torch.tensor(coords, dtype=torch.float32),
            "state_id": torch.tensor(state_id, dtype=torch.long),
        },
        os.path.join(dataset_path, "regions.pt"),
    )

    torch.save(
        {
            "climate": torch.tensor(climate, dtype=torch.float32),
            "pollution": torch.tensor(pollution, dtype=torch.float32),
            "socio": torch.tensor(socio, dtype=torch.float32),
        },
        os.path.join(dataset_path, "features.pt"),
    )

    torch.save(
        {
            "climate_mask": torch.tensor(M_clim, dtype=torch.bool),
            "pollution_mask": torch.tensor(M_poll, dtype=torch.bool),
            "socio_mask": torch.tensor(M_soc, dtype=torch.bool),
        },
        os.path.join(dataset_path, "masks.pt"),
    )

    torch.save(
        {
            "y1": torch.tensor(y1, dtype=torch.float32),
            "y2": torch.tensor(y2, dtype=torch.float32),
        },
        os.path.join(dataset_path, "targets.pt"),
    )

    with open(os.path.join(dataset_path, "meta.json"), "w") as f:
        json.dump({"seed": seed, "config": config}, f, indent=2)

    print(f"[OK] Dataset saved to: {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    generate(seed=7, config_path="configs/dataset.yaml", out_dir="data")
