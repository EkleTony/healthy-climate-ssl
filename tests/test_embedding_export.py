from pathlib import Path
import torch


DATASET_DIR = Path("data/v1_seed7")


def test_embedding_export_shape_and_order():
    feats = torch.load(DATASET_DIR / "features.pt")
    emb = torch.load(DATASET_DIR / "embeddings.pt")

    N = feats["climate"].shape[0]
    assert emb.shape[0] == N
    assert emb.ndim == 2
    assert emb.shape[1] == 192  # 64+64+64

    # sanity: embeddings shouldn't be all zeros
    assert emb.abs().mean().item() > 0.0
