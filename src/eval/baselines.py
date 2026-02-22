# src/eval/baselines.py
from __future__ import annotations

import torch


def idw_predict(
    train_coords: torch.Tensor,
    train_y: torch.Tensor,
    test_coords: torch.Tensor,
    power: float = 2.0,
    eps: float = 1e-12,
    k: int | None = 64,
) -> torch.Tensor:
    """
    Inverse Distance Weighting (IDW) baseline.

    Args:
        train_coords: [Ntr, 2]
        train_y:      [Ntr] or [Ntr, T]
        test_coords:  [Nte, 2]
        power:        IDW power (p)
        eps:          numerical stability
        k:            use k nearest neighbors for speed (None -> use all)

    Returns:
        pred_y: [Nte] or [Nte, T]
    """
    assert train_coords.ndim == 2 and train_coords.shape[1] == 2
    assert test_coords.ndim == 2 and test_coords.shape[1] == 2
    assert train_y.shape[0] == train_coords.shape[0]

    # Ensure 2D targets for unified handling
    squeeze = False
    if train_y.ndim == 1:
        train_y = train_y[:, None]
        squeeze = True

    # Pairwise distances: [Nte, Ntr]
    d = torch.cdist(test_coords, train_coords)  # Euclidean

    # Handle exact matches: if a test point equals a train point, copy its label
    exact = d <= eps
    if exact.any():
        pred = torch.empty(
            (test_coords.shape[0], train_y.shape[1]), device=train_y.device, dtype=train_y.dtype)
        has_exact = exact.any(dim=1)
        # For rows with exact match, take the first matched train index
        idx_first = exact.float().argmax(dim=1)
        pred[has_exact] = train_y[idx_first[has_exact]]
        # For others, do IDW normally
        remaining = ~has_exact
        if remaining.any():
            pred[remaining] = _idw_core(
                d[remaining], train_y, power=power, eps=eps, k=k)
    else:
        pred = _idw_core(d, train_y, power=power, eps=eps, k=k)

    if squeeze:
        pred = pred[:, 0]
    return pred


def _idw_core(d: torch.Tensor, train_y_2d: torch.Tensor, power: float, eps: float, k: int | None):
    # Optional kNN restriction
    if k is not None and k < d.shape[1]:
        knn_d, knn_idx = torch.topk(d, k=k, dim=1, largest=False)
        w = 1.0 / (knn_d.clamp_min(eps) ** power)  # [Nte, k]
        y = train_y_2d[knn_idx]                   # [Nte, k, T]
        w_sum = w.sum(dim=1, keepdim=True).clamp_min(eps)  # [Nte, 1]
        pred = (w[:, :, None] * y).sum(dim=1) / w_sum      # [Nte, T]
        return pred
    else:
        w = 1.0 / (d.clamp_min(eps) ** power)               # [Nte, Ntr]
        w_sum = w.sum(dim=1, keepdim=True).clamp_min(eps)   # [Nte, 1]
        pred = (w @ train_y_2d) / w_sum                      # [Nte, T]
        return pred
