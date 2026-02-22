from pathlib import Path
import torch


def load_pt(path):
    """
    Safe loader for .pt files.

    - Uses weights_only=True for tensor-only artifacts
    - Falls back to full pickle loading for objects like PyG Data (graph.pt)
    """
    path = Path(path)

    try:
        # Try safe tensor-only load first
        return torch.load(path, weights_only=True)
    except Exception:
        # Fallback for objects (e.g., PyG Data)
        return torch.load(path, weights_only=False)
