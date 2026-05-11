"""Stage 4 unit test: synthetic batch through GATLocalizer and MeanPoolMLP."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyg = pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data  # noqa: E402

from halo.gnn.model import GATLocalizer, MeanPoolMLP  # noqa: E402


def _random_graph(L: int = 32, in_dim: int = 640, seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(L, in_dim, generator=g)
    # Sequence-adjacency edges
    if L > 1:
        seq_src = torch.cat([torch.arange(L - 1), torch.arange(1, L)])
        seq_dst = torch.cat([torch.arange(1, L), torch.arange(L - 1)])
    else:
        seq_src = torch.zeros(0, dtype=torch.long)
        seq_dst = torch.zeros(0, dtype=torch.long)
    # A few random contact edges
    n_ct = max(1, L // 4)
    ct_src = torch.randint(0, L, (n_ct,), generator=g)
    ct_dst = torch.randint(0, L, (n_ct,), generator=g)
    src = torch.cat([seq_src, ct_src])
    dst = torch.cat([seq_dst, ct_dst])
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.cat([
        torch.zeros(seq_src.numel(), dtype=torch.long),
        torch.ones(ct_src.numel(), dtype=torch.long),
    ])
    y = torch.bernoulli(torch.full((1, 10), 0.3), generator=g)
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)


def test_gat_localizer_shape_and_finite():
    graphs = [_random_graph(L=L, seed=i) for i, L in enumerate([20, 35, 12, 50])]
    batch = Batch.from_data_list(graphs)
    model = GATLocalizer(in_dim=640, hidden=128, heads=4, num_layers=2,
                        num_classes=10, dropout=0.0)
    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
    assert out.shape == (4, 10), out.shape
    assert torch.isfinite(out).all()


def test_mean_pool_mlp_shape_and_finite():
    graphs = [_random_graph(L=L, seed=i) for i, L in enumerate([20, 35, 12, 50])]
    batch = Batch.from_data_list(graphs)
    model = MeanPoolMLP(in_dim=640, num_classes=10, dropout=0.0)
    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.batch)
    assert out.shape == (4, 10), out.shape
    assert torch.isfinite(out).all()
