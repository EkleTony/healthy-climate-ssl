from pathlib import Path
import torch
from torch.optim import Adam

from src.models.geomodrank import GeoModRank


DATASET_DIR = Path("data/v1_seed7")


def test_one_step_training_no_nan():
    feats = torch.load(DATASET_DIR / "features.pt")
    masks = torch.load(DATASET_DIR / "masks.pt")
    graph = torch.load(DATASET_DIR / "graph.pt")

    x_clim = feats["climate"].float()
    x_poll = feats["pollution"].float()
    x_soc = feats["socio"].float()

    obs_clim = masks["climate_mask"].bool()
    obs_poll = masks["pollution_mask"].bool()
    obs_soc = masks["socio_mask"].bool()

    edge_index = graph["edge_index"].long()

    model = GeoModRank(
        d_clim=x_clim.shape[1],
        d_poll=x_poll.shape[1],
        d_soc=x_soc.shape[1],
        z_clim=64,
        z_poll=64,
        z_soc=64,
        gnn_hidden=192,
        num_gnn_layers=2,
        dropout=0.1,
    )

    opt = Adam(model.parameters(), lr=1e-3)

    # simple corruption: mask 10% of observed entries
    torch.manual_seed(0)
    corr_clim = (torch.rand(x_clim.shape) < 0.1) & obs_clim
    corr_poll = (torch.rand(x_poll.shape) < 0.1) & obs_poll
    corr_soc = (torch.rand(x_soc.shape) < 0.1) & obs_soc
    x_clim_c = x_clim.clone()
    x_poll_c = x_poll.clone()
    x_soc_c = x_soc.clone()

    x_clim_c[corr_clim] = 0.0
    x_poll_c[corr_poll] = 0.0
    x_soc_c[corr_soc] = 0.0

    model.train()
    opt.zero_grad(set_to_none=True)

    z, (xh_clim, xh_poll, xh_soc) = model(
        x_clim_c, x_poll_c, x_soc_c, edge_index)

    # loss only on observed & corrupted
    def masked_mse(pred, target, mask):
        if mask.sum().item() == 0:
            return pred.new_tensor(0.0)
        return ((pred - target) ** 2)[mask].mean()

    loss = masked_mse(xh_clim, x_clim, obs_clim & corr_clim)
    loss = loss + masked_mse(xh_poll, x_poll, obs_poll & corr_poll)
    loss = loss + masked_mse(xh_soc, x_soc, obs_soc & corr_soc)

    assert torch.isfinite(loss).item(), f"Loss is NaN/Inf: {loss.item()}"

    loss.backward()
    opt.step()

    # check at least one grad exists
    has_grad = any(p.grad is not None for p in model.parameters()
                   if p.requires_grad)
    assert has_grad, "No gradients found after backward()"
