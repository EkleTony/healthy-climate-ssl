import os
import yaml
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.utils.seed import seed_all


def build_knn_graph(coords: np.ndarray, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a kNN graph in PyTorch Geometric edge_index format.
    Returns:
      edge_index: LongTensor [2, E]
      edge_weight: FloatTensor [E] (inverse distance)
    """
    assert coords.ndim == 2 and coords.shape[1] == 2, "coords must be [N,2]"
    N = coords.shape[0]
    k = int(k)
    if k >= N:
        raise ValueError(f"k must be < N, got k={k}, N={N}")

    # k+1 because first neighbor is itself (distance 0)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    dists, nbrs = nn.kneighbors(coords)  # [N,k+1], [N,k+1]

    src_list = []
    dst_list = []
    w_list = []

    for i in range(N):
        for j in range(1, k + 1):  # skip self at j=0
            nb = int(nbrs[i, j])
            dist = float(dists[i, j])

            src_list.append(i)
            dst_list.append(nb)

            # inverse distance weight (avoid divide by 0)
            w_list.append(1.0 / (dist + 1e-8))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(w_list, dtype=torch.float32)
    return edge_index, edge_weight


def save_graph(seed: int, config_path: str, dataset_dir: str) -> str:
    """
    Loads regions.pt from dataset_dir and writes graph.pt into dataset_dir.
    Returns path to graph.pt
    """
    seed_all(seed)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    k = int(config.get("k_graph", 10))

    regions_path = os.path.join(dataset_dir, "regions.pt")
    if not os.path.exists(regions_path):
        raise FileNotFoundError(f"Missing {regions_path}. Generate dataset first.")

    regions = torch.load(regions_path)
    coords = regions["coords"].cpu().numpy()

    edge_index, edge_weight = build_knn_graph(coords, k=k)

    out_path = os.path.join(dataset_dir, "graph.pt")
    torch.save(
        {"edge_index": edge_index,
         "edge_weight": edge_weight,
         "k": k,
         "seed": seed},
        out_path
    )

    print(f"Graph saved to: {out_path} | edge_index={tuple(edge_index.shape)}")
    return out_path


if __name__ == "__main__":
    # Example:
    # python src/data/graph.py data/v1_seed7 configs/dataset.yaml 7
    import sys
    if len(sys.argv) != 4:
        print("Usage: python src/data/graph.py <dataset_dir> <config_path> <seed>")
        raise SystemExit(1)

    dataset_dir = sys.argv[1]
    config_path = sys.argv[2]
    seed = int(sys.argv[3])
    save_graph(seed=seed, config_path=config_path, dataset_dir=dataset_dir)
