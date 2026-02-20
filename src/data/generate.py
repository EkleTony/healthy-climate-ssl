import os
import json
import yaml
import torch
import numpy as np
from sklearn.cluster import KMeans

from src.utils.seed import seed_all


def smooth_field(coords: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Smooth latent spatial field over 2D coords."""
    x = coords[:, 0]
    y = coords[:, 1]
    return scale * (np.sin(0.5 * x) + np.cos(0.5 * y))


def generate(seed: int, config_path: str, out_dir: str) -> str:
    """
    Deterministic synthetic dataset generator.
    Writes dataset artifacts and returns the dataset path.
    """
    seed_all(seed)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    name = config["name"]
    N = int(config["N"])
    K = int(config["K_states"])

    d_clim = int(config["dims"]["climate"])
    d_poll = int(config["dims"]["pollution"])
    d_soc = int(config["dims"]["socio"])

    miss_clim = float(config["missingness"]["climate_rate"])
    miss_poll = float(config["missingness"]["pollution_rate"])
    miss_soc = float(config["missingness"]["socio_rate"])

    # --- 1) Coordinates: clustered geography (mixture of Gaussians) ---
    centers = np.random.uniform(-5, 5, size=(K, 2))
    cluster_ids = np.random.choice(K, size=N)
    coords = centers[cluster_ids] + np.random.normal(0, 0.8, size=(N, 2))

    # --- 2) Grouping variable: "state_id" via KMeans ---
    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(coords)
    state_id = kmeans.labels_  # [N]

    # --- 3) Latent spatial fields ---
    latent1 = smooth_field(coords, scale=1.0)
    latent2 = smooth_field(coords, scale=0.6)

    # --- 4) Modalities (3 blocks) ---
    # Climate: mostly smooth + mild noise
    climate = np.stack([latent1 + np.random.normal(0, 0.30, N) for _ in range(d_clim)], axis=1)

    # Pollution: smooth + mixture with latent1 + stronger noise
    pollution = np.stack([(latent2 + 0.5 * latent1) + np.random.normal(0, 0.40, N) for _ in range(d_poll)], axis=1)

    # Socio: state-level shift + local variation
    state_shift = np.random.normal(0, 0.6, size=(K,))
    socio = np.stack([state_shift[state_id] + np.random.normal(0, 0.50, N) for _ in range(d_soc)], axis=1)

    # --- 5) Missingness masks (random for now; we’ll make structured missingness next) ---
    M_clim = (np.random.rand(N, d_clim) > miss_clim)
    M_poll = (np.random.rand(N, d_poll) > miss_poll)
    M_soc = (np.random.rand(N, d_soc) > miss_soc)

    # Apply masks by zeroing missing entries (OK as long as masks are saved and used later)
    climate = climate * M_clim
    pollution = pollution * M_poll
    socio = socio * M_soc

    # --- 6) Targets (two continuous regression targets) ---
    # depend on both geography (latent fields) and feature blocks, plus noise
    y1 = latent1 + 0.30 * climate.mean(axis=1) + 0.20 * pollution.mean(axis=1) + 0.15 * state_shift[state_id] + np.random.normal(0, 0.50, N)
    y2 = latent2 + 0.40 * socio.mean(axis=1) + 0.10 * climate.mean(axis=1) - 0.10 * state_shift[state_id] + np.random.normal(0, 0.50, N)

    # --- 7) Save artifacts ---
    dataset_path = os.path.join(out_dir, f"{name}_seed{seed}")
    os.makedirs(dataset_path, exist_ok=True)

    torch.save(
        {"coords": torch.tensor(coords, dtype=torch.float32),
         "state_id": torch.tensor(state_id, dtype=torch.long)},
        os.path.join(dataset_path, "regions.pt")
    )

    torch.save(
        {"climate": torch.tensor(climate, dtype=torch.float32),
         "pollution": torch.tensor(pollution, dtype=torch.float32),
         "socio": torch.tensor(socio, dtype=torch.float32)},
        os.path.join(dataset_path, "features.pt")
    )

    torch.save(
        {"climate_mask": torch.tensor(M_clim, dtype=torch.bool),
         "pollution_mask": torch.tensor(M_poll, dtype=torch.bool),
         "socio_mask": torch.tensor(M_soc, dtype=torch.bool)},
        os.path.join(dataset_path, "masks.pt")
    )

    torch.save(
        {"y1": torch.tensor(y1, dtype=torch.float32),
         "y2": torch.tensor(y2, dtype=torch.float32)},
        os.path.join(dataset_path, "targets.pt")
    )

    with open(os.path.join(dataset_path, "meta.json"), "w") as f:
        json.dump({"seed": seed, "config": config}, f, indent=2)

    print(f"Dataset saved to: {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    generate(seed=7, config_path="configs/dataset.yaml", out_dir="data")
