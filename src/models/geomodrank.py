import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class MLP(nn.Module):
    """Small MLP used for modality-specific encoders/decoders."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GeoModRank(nn.Module):
    """
    GeoModRank (Part B):
      - Modality-specific encoders
      - GraphSAGE trunk (PyG) for spatial inductive bias
      - Explicit partitioned embedding: [z_clim | z_poll | z_soc]
      - Modality-specific decoders (one per modality)
    """

    def __init__(
        self,
        d_clim: int,
        d_poll: int,
        d_soc: int,
        z_clim: int = 64,
        z_poll: int = 64,
        z_soc: int = 64,
        gnn_hidden: int = 192,
        num_gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- Encoders (explicit modality awareness) ---
        self.enc_clim = MLP(d_clim, z_clim, hidden_dim=128, dropout=dropout)
        self.enc_poll = MLP(d_poll, z_poll, hidden_dim=128, dropout=dropout)
        self.enc_soc = MLP(d_soc, z_soc, hidden_dim=128, dropout=dropout)

        # Embedding partition sizes
        self.z_clim, self.z_poll, self.z_soc = z_clim, z_poll, z_soc
        self.z_dim = z_clim + z_poll + z_soc

        # Pre-GNN projection
        self.pre = nn.Linear(self.z_dim, gnn_hidden)

        # --- GraphSAGE trunk ---
        self.convs = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(SAGEConv(gnn_hidden, gnn_hidden))

        self.dropout = nn.Dropout(dropout)

        # Post-GNN projection back to partitioned embedding space
        self.post = nn.Linear(gnn_hidden, self.z_dim)

        # --- Decoders (explicit modality awareness) ---
        self.dec_clim = MLP(z_clim, d_clim, hidden_dim=128, dropout=dropout)
        self.dec_poll = MLP(z_poll, d_poll, hidden_dim=128, dropout=dropout)
        self.dec_soc = MLP(z_soc, d_soc, hidden_dim=128, dropout=dropout)

    def encode(
        self,
        x_clim: torch.Tensor,
        x_poll: torch.Tensor,
        x_soc: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        # Modality encoders
        h_clim = self.enc_clim(x_clim)
        h_poll = self.enc_poll(x_poll)
        h_soc = self.enc_soc(x_soc)

        # Concatenate modality embeddings
        h = torch.cat([h_clim, h_poll, h_soc], dim=-1)

        # Project + message passing
        h = self.pre(h)
        h = F.gelu(h)
        h = self.dropout(h)

        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.gelu(h)
            h = self.dropout(h)

        # Final region embedding (partitioned)
        z = self.post(h)
        return z

    def decode(self, z: torch.Tensor):
        # Explicit partitioning
        zc = z[:, : self.z_clim]
        zp = z[:, self.z_clim: self.z_clim + self.z_poll]
        zs = z[:, self.z_clim + self.z_poll:]

        # Modality-specific reconstruction
        xhat_clim = self.dec_clim(zc)
        xhat_poll = self.dec_poll(zp)
        xhat_soc = self.dec_soc(zs)
        return xhat_clim, xhat_poll, xhat_soc

    def forward(
        self,
        x_clim: torch.Tensor,
        x_poll: torch.Tensor,
        x_soc: torch.Tensor,
        edge_index: torch.Tensor,
    ):
        z = self.encode(x_clim, x_poll, x_soc, edge_index)
        xhat = self.decode(z)
        return z, xhat
