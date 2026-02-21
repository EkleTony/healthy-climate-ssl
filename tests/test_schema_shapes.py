from pathlib import Path
import torch


DATASET_DIR = Path("data/v1_seed7")


def test_expected_files_exist():
    expected = [
        "regions.pt",
        "features.pt",
        "masks.pt",
        "targets.pt",
        "graph.pt",
        "embeddings.pt",
        "geomodrank.pt",
    ]
    for f in expected:
        assert (DATASET_DIR / f).exists(), f"Missing file: {f}"


def test_schema_and_shapes():
    feats = torch.load(DATASET_DIR / "features.pt")
    masks = torch.load(DATASET_DIR / "masks.pt")
    regions = torch.load(DATASET_DIR / "regions.pt")
    graph = torch.load(DATASET_DIR / "graph.pt")
    emb = torch.load(DATASET_DIR / "embeddings.pt")

    # keys
    assert set(feats.keys()) == {"climate", "pollution", "socio"}
    assert set(masks.keys()) == {"climate_mask",
                                 "pollution_mask", "socio_mask"}
    assert set(graph.keys()) >= {"edge_index", "edge_weight", "k", "seed"}

    # shapes
    N = regions["coords"].shape[0]
    assert regions["coords"].shape[1] == 2

    assert feats["climate"].shape[0] == N
    assert feats["pollution"].shape[0] == N
    assert feats["socio"].shape[0] == N

    assert masks["climate_mask"].shape == feats["climate"].shape
    assert masks["pollution_mask"].shape == feats["pollution"].shape
    assert masks["socio_mask"].shape == feats["socio"].shape

    # embeddings match N
    assert emb.shape[0] == N
    assert emb.ndim == 2

    # edge_index validity
    edge_index = graph["edge_index"]
    assert edge_index.shape[0] == 2
    assert edge_index.dtype in (torch.int64, torch.long)

    assert edge_index.min().item() >= 0
    assert edge_index.max().item() < N
