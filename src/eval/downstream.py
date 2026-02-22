# src/eval/downstream.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class Metrics:
    mse: float
    rmse: float
    r2: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    # R2: 1 - SSE/SST (handle constant y)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else 0.0

    return Metrics(mse=mse, rmse=rmse, r2=r2)


def ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
) -> Dict:
    """
    Simple ridge regression with alpha tuned on validation set.

    Returns:
        dict with best_alpha, yhat_test, yhat_val
    """
    # Standardize using train only
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    Xtr = (X_train - mu) / sd
    Xva = (X_val - mu) / sd
    Xte = (X_test - mu) / sd

    # Closed-form ridge: w = (X^T X + a I)^-1 X^T y
    # Add bias via column of ones
    Xtr_b = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xva_b = np.concatenate([Xva, np.ones((Xva.shape[0], 1))], axis=1)
    Xte_b = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)

    best = None
    d = Xtr_b.shape[1]
    I = np.eye(d, dtype=np.float64)
    I[-1, -1] = 0.0  # don't regularize bias

    for a in alphas:
        A = Xtr_b.T @ Xtr_b + a * I
        b = Xtr_b.T @ y_train
        w = np.linalg.solve(A, b)

        yhat_val = Xva_b @ w
        val_mse = np.mean((y_val - yhat_val) ** 2)

        if best is None or val_mse < best["val_mse"]:
            best = {"alpha": float(a), "val_mse": float(val_mse), "w": w}

    w = best["w"]
    return {
        "best_alpha": best["alpha"],
        "yhat_val": Xva_b @ w,
        "yhat_test": Xte_b @ w,
    }


def torch_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
