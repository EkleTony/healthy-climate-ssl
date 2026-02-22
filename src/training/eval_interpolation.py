# src/training/eval_interpolation.py
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
# REMOVED: from sklearn.kernel_ridge import KernelRidge


# -----------------------------
# Utility
# -----------------------------
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def fit_scaler_train_only(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# -----------------------------
# Baselines (coords-only)
# -----------------------------
def mean_train_predict(y_train, n_out):
    mu = float(np.mean(y_train))
    return np.full(shape=(n_out,), fill_value=mu, dtype=np.float64)


def idw_predict(train_coords, train_y, query_coords, power=2.0, eps=1e-12):
    # weights = 1 / dist^p
    d = np.linalg.norm(query_coords[:, None, :] -
                       train_coords[None, :, :], axis=-1)
    w = 1.0 / np.power(d + eps, power)
    w_sum = np.sum(w, axis=1, keepdims=True) + eps
    y_hat = (w @ train_y.reshape(-1, 1)) / w_sum
    return y_hat.squeeze(1)


def tune_knn_coords(train_coords, train_y, val_coords, val_y, k_grid):
    best = None
    for k in k_grid:
        model = KNeighborsRegressor(n_neighbors=k, weights="distance")
        model.fit(train_coords, train_y)
        pred = model.predict(val_coords)
        score = rmse(val_y, pred)
        if (best is None) or (score < best["rmse_val"]):
            best = {"k": int(k), "rmse_val": float(score)}
    return best


# -----------------------------
# Simple Kernel Regression (coords-only)
# Nadaraya–Watson Kernel Regression (NWKR) with RBF kernel
# -----------------------------
def nwkr_predict(train_coords, train_y, query_coords, bandwidth, eps=1e-12):
    """
    Nadaraya–Watson kernel regression (Gaussian/RBF).

    pred(x) = sum_i exp(-||x-x_i||^2 / (2*h^2)) * y_i / sum_i exp(-||x-x_i||^2 / (2*h^2))

    train_coords: [Ntr, d]
    train_y:      [Ntr]
    query_coords: [Nq, d]
    bandwidth (h) > 0
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")

    # Pairwise squared distances: [Nq, Ntr]
    x2 = np.sum(query_coords ** 2, axis=1, keepdims=True)        # [Nq,1]
    t2 = np.sum(train_coords ** 2, axis=1, keepdims=True).T      # [1,Ntr]
    d2 = x2 + t2 - 2.0 * (query_coords @ train_coords.T)         # [Nq,Ntr]
    d2 = np.maximum(d2, 0.0)

    w = np.exp(-d2 / (2.0 * (bandwidth ** 2)))
    denom = np.sum(w, axis=1) + eps
    pred = (w @ train_y) / denom
    return pred


def tune_nwkr_rbf_coords(train_coords, train_y, val_coords, val_y, bandwidth_grid):
    best = None
    for bw in bandwidth_grid:
        pred = nwkr_predict(train_coords, train_y, val_coords, bandwidth=bw)
        score = rmse(val_y, pred)
        if (best is None) or (score < best["rmse_val"]):
            best = {"bandwidth": float(bw), "rmse_val": float(score)}
    return best


# -----------------------------
# Embedding-based predictors
# -----------------------------
def tune_ridge_embeddings(X_train, y_train, X_val, y_val, alpha_grid):
    best = None
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = rmse(y_val, pred)
        if (best is None) or (score < best["rmse_val"]):
            best = {"alpha": float(alpha), "rmse_val": float(score)}
    return best


class TinyMLP(nn.Module):
    def __init__(self, d_in, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_mlp_with_early_stopping(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden=128,
    dropout=0.1,
    lr=1e-3,
    weight_decay=0.0,
    max_epochs=200,
    patience=20,
    seed=0,
):
    torch.manual_seed(seed)
    model = TinyMLP(d_in=X_train.shape[1], hidden=hidden, dropout=dropout)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).float()
    Xva = torch.from_numpy(X_val).float()
    yva = torch.from_numpy(y_val).float()

    best_state, best_val = None, float("inf")
    bad = 0

    for _epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_loss = loss_fn(val_pred, yva).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_val)


def _train_mlp_fixed_epochs(
    X_train,
    y_train,
    hidden=128,
    dropout=0.1,
    lr=1e-3,
    weight_decay=0.0,
    max_epochs=200,
    seed=0,
):
    torch.manual_seed(seed)
    model = TinyMLP(d_in=X_train.shape[1], hidden=hidden, dropout=dropout)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).float()

    for _epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        opt.step()

    return model


# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--use_mlp", action="store_true")
    ap.add_argument(
        "--mode",
        type=str,
        default="val",
        choices=["val", "test"],
        help="val=tune/compare on validation (run many times); test=final report (run once)",
    )
    args = ap.parse_args()

    d = Path(args.data_dir)

    # Plain torch.load (keep warnings)
    #regions = torch.load(d / "regions.pt")
    regions = torch.load(
        d / "regions.pt", weights_only=True, map_location="cpu")
    splits = torch.load(
        d / f"splits_seed{args.seed}.pt", weights_only=True, map_location="cpu")
    emb = torch.load(d / "embeddings.pt",
                     weights_only=True, map_location="cpu")
    targets = torch.load(
        d / "targets.pt", weights_only=True, map_location="cpu")

    coords = to_np(regions["coords"]).astype(np.float64)  # [N,2]

    # splits = torch.load(d / f"splits_seed{args.seed}.pt")
    tr = to_np(splits["train_idx"]).astype(np.int64)
    va = to_np(splits["val_idx"]).astype(np.int64)
    te = to_np(splits["test_idx"]).astype(np.int64)

    # emb = to_np(torch.load(d / "embeddings.pt")).astype(np.float64)  # [N,D]
    # targets = torch.load(d / "targets.pt")

    # Grids (simple)
    #knn_k_grid = [3, 5, 10, 20, 30] 
    knn_k_grid = [3, 5, 10, 20, 30, 50, 75, 100]
    # REMOVED: KRR grids
    nwkr_bw_grid =  [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0] #[0.25, 0.5, 1.0, 2.0, 4.0]   # after coord standardization
    ridge_alpha_grid = [1e-2, 1e-1, 1.0, 10.0, 100.0]
    idw_p = 2.0

    # Fit scalers on TRAIN only (no leakage)
    emb_scaler = fit_scaler_train_only(emb[tr])
    emb_tr = emb_scaler.transform(emb[tr])
    emb_va = emb_scaler.transform(emb[va])
    emb_te = emb_scaler.transform(emb[te])

    coord_scaler = fit_scaler_train_only(coords[tr])
    c_tr = coord_scaler.transform(coords[tr])
    c_va = coord_scaler.transform(coords[va])
    c_te = coord_scaler.transform(coords[te])

    split_name = "VAL (for tuning)" if args.mode == "val" else "TEST (final report; run once)"
    print(f"\n=== Downstream Interpolation ({split_name}) ===")

    results = {}

    for name, y in targets.items():
        y = to_np(y).astype(np.float64)
        y_tr, y_va, y_te = y[tr], y[va], y[te]

        if args.mode == "val":
            y_eval = y_va
            c_eval = c_va
            emb_eval = emb_va
        else:
            y_eval = y_te
            c_eval = c_te
            emb_eval = emb_te

        # ---------------- Baselines ----------------
        pred_mean = mean_train_predict(y_tr, n_out=len(y_eval))
        mean_rmse, mean_r2 = rmse(
            y_eval, pred_mean), r2_score(y_eval, pred_mean)

        pred_idw = idw_predict(c_tr, y_tr, c_eval, power=idw_p)
        idw_rmse, idw_r2 = rmse(y_eval, pred_idw), r2_score(y_eval, pred_idw)

        knn_best = tune_knn_coords(c_tr, y_tr, c_va, y_va, knn_k_grid)
        knn_model = KNeighborsRegressor(
            n_neighbors=knn_best["k"], weights="distance")
        knn_model.fit(c_tr, y_tr)  # strict: train only
        pred_knn = knn_model.predict(c_eval)
        knn_rmse, knn_r2 = rmse(y_eval, pred_knn), r2_score(y_eval, pred_knn)

        # NEW: NW Kernel Regression baseline (replaces KRR)
        nwkr_best = tune_nwkr_rbf_coords(c_tr, y_tr, c_va, y_va, nwkr_bw_grid)
        pred_nwkr = nwkr_predict(
            c_tr, y_tr, c_eval, bandwidth=nwkr_best["bandwidth"])
        nwkr_rmse, nwkr_r2 = rmse(
            y_eval, pred_nwkr), r2_score(y_eval, pred_nwkr)

        # ---------------- Embedding-based ----------------
        ridge_best = tune_ridge_embeddings(
            emb_tr, y_tr, emb_va, y_va, ridge_alpha_grid)
        ridge_model = Ridge(alpha=ridge_best["alpha"])
        ridge_model.fit(emb_tr, y_tr)  # strict: train only
        pred_ridge = ridge_model.predict(emb_eval)
        ridge_rmse, ridge_r2 = rmse(
            y_eval, pred_ridge), r2_score(y_eval, pred_ridge)

        mlp_out = None
        if args.use_mlp:
            hidden_grid = [64, 128, 256]
            dropout_grid = [0.0, 0.1]
            wd_grid = [0.0, 1e-4, 1e-3]

            best_val_loss = None
            best_cfg = None

            for h in hidden_grid:
                for dr in dropout_grid:
                    for wd in wd_grid:
                        _m, val_loss = _train_mlp_with_early_stopping(
                            emb_tr, y_tr, emb_va, y_va,
                            hidden=h, dropout=dr, lr=1e-3, weight_decay=wd,
                            max_epochs=200, patience=20, seed=args.seed
                        )
                        if (best_val_loss is None) or (val_loss < best_val_loss):
                            best_val_loss = val_loss
                            best_cfg = {"hidden": int(h), "dropout": float(
                                dr), "weight_decay": float(wd)}

            final_mlp = _train_mlp_fixed_epochs(
                emb_tr, y_tr,
                hidden=best_cfg["hidden"],
                dropout=best_cfg["dropout"],
                lr=1e-3,
                weight_decay=best_cfg["weight_decay"],
                max_epochs=200,
                seed=args.seed
            )
            final_mlp.eval()
            with torch.no_grad():
                pred_mlp = final_mlp(torch.from_numpy(
                    emb_eval).float()).cpu().numpy()

            mlp_out = {
                "rmse": float(rmse(y_eval, pred_mlp)),
                "r2": float(r2_score(y_eval, pred_mlp)),
                **best_cfg,
            }

        results[name] = {
            "MeanTrain": {"rmse": float(mean_rmse), "r2": float(mean_r2)},
            "IDW": {"rmse": float(idw_rmse), "r2": float(idw_r2), "power": float(idw_p)},
            "kNN(coords)": {"rmse": float(knn_rmse), "r2": float(knn_r2), "k": int(knn_best["k"])},
            "NWKR(RBF coords)": {"rmse": float(nwkr_rmse), "r2": float(nwkr_r2), **nwkr_best},
            "GeoModRank+Ridge": {"rmse": float(ridge_rmse), "r2": float(ridge_r2), "alpha": float(ridge_best["alpha"])},
        }
        if mlp_out is not None:
            results[name]["GeoModRank+MLP"] = mlp_out

        print(f"\nTarget: {name}")
        ordered = ["MeanTrain", "IDW",
                   "kNN(coords)", "NWKR(RBF coords)", "GeoModRank+Ridge"]
        if mlp_out is not None:
            ordered.append("GeoModRank+MLP")

        for model_name in ordered:
            r = results[name][model_name]
            note = ""
            if model_name == "kNN(coords)":
                note = f"k={r['k']}"
            elif model_name == "NWKR(RBF coords)":
                note = f"bw={r['bandwidth']}"
            elif model_name == "GeoModRank+Ridge":
                note = f"alpha={r['alpha']}"
            elif model_name == "IDW":
                note = f"p={r['power']}"
            elif model_name == "GeoModRank+MLP":
                note = f"hidden={r['hidden']} dropout={r['dropout']} wd={r['weight_decay']}"

            print(
                f"  {model_name:18s}  RMSE={r['rmse']:.4f}  R2={r['r2']:.4f}  {note}")

    out = d / f"downstream_results_seed{args.seed}_{args.mode}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results -> {out}\n")


if __name__ == "__main__":
    main()
