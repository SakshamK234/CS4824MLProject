"""GAT model for multi-label compartment prediction.

GATLocalizer: 2 GATv2Conv layers, hidden 128, 4 heads, edge_dim=2,
mean-pool + attention-pool, MLP head, 10-way logits (sigmoid via BCE in train).
MeanPoolMLP: row-2 ablation that mean-pools per-residue ESM2 to a 640-d vector
then runs a 2-layer MLP.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import AttentionalAggregation, GATv2Conv, global_mean_pool
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False


def _edge_attr_from_type(edge_type: torch.Tensor) -> torch.Tensor:
    """[E] long -> [E, 2] float one-hot."""
    return F.one_hot(edge_type.long(), num_classes=2).float()


class GATLocalizer(nn.Module):
    def __init__(self, in_dim: int = 640, hidden: int = 128, heads: int = 4,
                 num_classes: int = 10, num_layers: int = 2, dropout: float = 0.1,
                 edge_dim: int = 2):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("torch_geometric is required for GATLocalizer")
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        per_head = hidden // heads
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(hidden, per_head, heads=heads,
                                        edge_dim=edge_dim, dropout=dropout,
                                        add_self_loops=True))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.dropout = dropout
        gate_nn = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                nn.Linear(hidden, 1))
        self.attn_pool = AttentionalAggregation(gate_nn=gate_nn)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self._last_attention: list = []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor, batch: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        h = self.in_proj(x.float())
        edge_attr = _edge_attr_from_type(edge_type)
        attn_records = []
        for conv, norm in zip(self.convs, self.norms):
            if return_attention:
                h_new, (ei_a, alpha) = conv(h, edge_index, edge_attr=edge_attr,
                                             return_attention_weights=True)
                attn_records.append((ei_a.detach().cpu(), alpha.detach().cpu()))
            else:
                h_new = conv(h, edge_index, edge_attr=edge_attr)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = norm(h_new + h)  # residual + LayerNorm
        # Concatenate mean-pool and attention-pool readouts for the graph-level head.
        mean = global_mean_pool(h, batch)
        attn = self.attn_pool(h, batch)
        z = torch.cat([mean, attn], dim=-1)
        if return_attention:
            self._last_attention = attn_records
        return self.head(z)


class MeanPoolMLP(nn.Module):
    """Row-2 ablation."""

    def __init__(self, in_dim: int = 640, hidden: int = 256, num_classes: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if _HAS_PYG:
            pooled = global_mean_pool(x.float(), batch)
        else:
            pooled = torch.zeros(int(batch.max().item()) + 1, x.shape[1],
                                 device=x.device, dtype=x.dtype)
            counts = torch.zeros(pooled.shape[0], device=x.device)
            pooled.index_add_(0, batch, x.float())
            counts.index_add_(0, batch, torch.ones(x.shape[0], device=x.device))
            pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
        return self.net(pooled.float())
