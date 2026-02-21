import os
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from src.models.geomodrank import GeoModRank


@dataclass
class TrainConfig:
    dataset_dir: str = "data/v1_seed7"
    seed: int = 7

    # corruption
    mask_ratio: float = 0.3
    noise_std: float = 0.0

    # model
    z_clim: int = 64
    z_poll: int = 64
    z_soc: int = 64
    gnn_hidden: int = 192
    num_gnn_layers: int = 2
    dropout: float = 0.1

    # optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50

    # outputs
    save_ckpt: bool = True
    ckpt_name: str = "geomodrank.pt"
    emb_name: str = "embeddings.pt"


def set_seed(seed: int):
    """Deterministic settings (best-effort on CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def make_corruption_mask(obs_mask: torch.Tensor, mask_ratio: float, g: torch.Generator) -> torch.Tensor:
    """
    obs_mask: Bool tensor [N, d]
    Returns corrupt_mask: Bool tensor [N, d] (True means this observed entry is corrupted).
    """
    rand = torch.rand(obs_mask.shape, generator=g, device=obs_mask.device)
    return (rand < mask_ratio) & obs_mask


def apply_corruption(x: torch.Tensor, corrupt_mask: torch.Tensor, noise_std: float, g: torch.Generator) -> torch.Tensor:
    """Set corrupted entries to 0, optionally add noise."""
    x2 = x.clone()
    x2[corrupt_mask] = 0.0
    if noise_std > 0.0:
        x2 = x2 + torch.randn_like(x2, generator=g) * noise_std
    return x2


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE computed only over mask==True entries; safe if mask has no True entries."""
    if mask.sum().item() == 0:
        return pred.new_tensor(0.0)
    return ((pred - target) ** 2)[mask].mean()


def geomodrank_loss(
    xhat_clim, xhat_poll, xhat_soc,
    x_clim, x_poll, x_soc,
    obs_clim, obs_poll, obs_soc,
    corr_clim, corr_poll, corr_soc,
) -> torch.Tensor:
    """
    Rubric-compliant:
      Loss only on (observed AND corrupted) entries.
    """
    m_clim = obs_clim & corr_clim
    m_poll = obs_poll & corr_poll
    m_soc = obs_soc & corr_soc

    loss = masked_mse(xhat_clim, x_clim, m_clim)
    loss = loss + masked_mse(xhat_poll, x_poll, m_poll)
    loss = loss + masked_mse(xhat_soc,  x_soc,  m_soc)
    return loss


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    # keep simple/reproducible; your Mac is CPU anyway
    device = torch.device("cpu")

    d = Path(cfg.dataset_dir)

    # --- Load artifacts ---
    feats = torch.load(d / "features.pt")
    masks = torch.load(d / "masks.pt")
    graph = torch.load(d / "graph.pt")

    x_clim = feats["climate"].float().to(device)
    x_poll = feats["pollution"].float().to(device)
    x_soc = feats["socio"].float().to(device)

    obs_clim = masks["climate_mask"].bool().to(device)
    obs_poll = masks["pollution_mask"].bool().to(device)
    obs_soc = masks["socio_mask"].bool().to(device)

    edge_index = graph["edge_index"].long().to(device)

    # --- Model ---
    model = GeoModRank(
        d_clim=x_clim.shape[1],
        d_poll=x_poll.shape[1],
        d_soc=x_soc.shape[1],
        z_clim=cfg.z_clim,
        z_poll=cfg.z_poll,
        z_soc=cfg.z_soc,
        gnn_hidden=cfg.gnn_hidden,
        num_gnn_layers=cfg.num_gnn_layers,
        dropout=cfg.dropout,
    ).to(device)

    opt = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # corruption RNG
    g = torch.Generator(device=device).manual_seed(cfg.seed + 12345)

    # --- Training loop (full-batch) ---
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        # Sample corruption masks each epoch (still deterministic due to fixed Generator)
        corr_clim = make_corruption_mask(obs_clim, cfg.mask_ratio, g)
        corr_poll = make_corruption_mask(obs_poll, cfg.mask_ratio, g)
        corr_soc = make_corruption_mask(obs_soc,  cfg.mask_ratio, g)

        x_clim_c = apply_corruption(x_clim, corr_clim, cfg.noise_std, g)
        x_poll_c = apply_corruption(x_poll, corr_poll, cfg.noise_std, g)
        x_soc_c = apply_corruption(x_soc,  corr_soc,  cfg.noise_std, g)

        opt.zero_grad(set_to_none=True)

        z, (xh_clim, xh_poll, xh_soc) = model(
            x_clim_c, x_poll_c, x_soc_c, edge_index)

        loss = geomodrank_loss(
            xh_clim, xh_poll, xh_soc,
            x_clim, x_poll, x_soc,
            obs_clim, obs_poll, obs_soc,
            corr_clim, corr_poll, corr_soc,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Loss became NaN/Inf at epoch {epoch}: {loss.item()}")

        loss.backward()
        opt.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"[GeoModRank] epoch={epoch:03d} loss={loss.item():.6f}")

    # --- Save checkpoint ---
    if cfg.save_ckpt:
        ckpt_path = d / cfg.ckpt_name
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "graph_seed": int(graph.get("seed", -1)),
                "k": int(graph.get("k", -1)),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint -> {ckpt_path}")

    # --- Export embeddings (no corruption) ---
    model.eval()
    with torch.no_grad():
        z_clean = model.encode(x_clim, x_poll, x_soc, edge_index).cpu()

    emb_path = d / cfg.emb_name
    torch.save(z_clean, emb_path)
    print(f"Saved embeddings -> {emb_path}  shape={tuple(z_clean.shape)}")

    # Save a small json log for reproducibility
    run_meta = {
        "seed": cfg.seed,
        "dataset_dir": cfg.dataset_dir,
        "mask_ratio": cfg.mask_ratio,
        "noise_std": cfg.noise_std,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "embedding_dim": int(z_clean.shape[1]),
    }
    with open(d / "train_ssl_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)
    print(f"Saved training metadata -> {d / 'train_ssl_meta.json'}")


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)
