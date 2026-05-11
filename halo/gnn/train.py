"""Train the GAT multi-label localizer (and the row-2 MeanPoolMLP ablation).

Splits respect the DeepLoc 2.0 partitioning shipped in Data/. The test set
is touched exactly once, at the end of training.

Loss: BCEWithLogitsLoss with per-class pos_weight from training-set positive
rates (rare compartments get up-weighted).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from halo.data.labels import LABEL_COLUMNS, NUM_LABELS
from halo.spec.classical_models import evaluate_binary

log = logging.getLogger(__name__)


# ---- Dataset ---------------------------------------------------------------

def _split_accs(csv_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path).dropna(subset=LABEL_COLUMNS)
    col = "ACC" if "ACC" in df.columns else "AccessionID"
    accs = []
    labels = {}
    for _, row in df.iterrows():
        a = str(row[col]).strip()
        if not a or a.lower() == "nan":
            continue
        labels[a] = np.array([float(row[L]) for L in LABEL_COLUMNS], dtype=np.float32)
        accs.append(a)
    return accs, labels


def _filter_existing(accs: list[str], graph_dir: Path) -> list[str]:
    return [a for a in accs if (graph_dir / f"{a}.pt").exists()]


def _filter_existing_emb(accs: list[str], emb_dir: Path) -> list[str]:
    return [a for a in accs if (emb_dir / f"{a}.pt").exists()]


def _load_graph(graph_dir: Path, acc: str, edges: str = "both"):
    """Load a saved graph dict and turn it into a torch_geometric Data object,
    optionally masking edge_type."""
    from torch_geometric.data import Data
    obj = torch.load(graph_dir / f"{acc}.pt", map_location="cpu", weights_only=False)
    edge_index = obj["edge_index"]
    edge_type = obj["edge_type"]
    if edges == "sequence":
        keep = edge_type == 0
    elif edges == "contact":
        keep = edge_type == 1
    else:
        keep = torch.ones_like(edge_type, dtype=torch.bool)
    edge_index = edge_index[:, keep]
    edge_type = edge_type[keep]
    data = Data(
        x=obj["x"].float(),
        edge_index=edge_index,
        edge_type=edge_type,
        y=obj["y"].unsqueeze(0),  # [1, 10]
        acc=obj["acc"],
        n_residues=int(obj["n_residues"]),
    )
    return data


def _load_mlp_record(emb_dir: Path, acc: str, label_vec: np.ndarray):
    """Load a (mean_x, y) item for the row-2 MLP path. Build a degenerate
    graph with one node = mean(esm) so we can re-use the same DataLoader."""
    from torch_geometric.data import Data
    obj = torch.load(emb_dir / f"{acc}.pt", map_location="cpu", weights_only=False)
    x = obj["x"].float().mean(dim=0, keepdim=True)  # [1, 640]
    return Data(
        x=x,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_type=torch.zeros(0, dtype=torch.long),
        y=torch.from_numpy(label_vec).unsqueeze(0),
        acc=acc,
        n_residues=1,
    )


class GraphIterableDataset(torch.utils.data.Dataset):
    def __init__(self, accs: list[str], graph_dir: Path, edges: str = "both",
                 mode: str = "gat", emb_dir: Path | None = None,
                 labels: dict[str, np.ndarray] | None = None):
        self.accs = accs
        self.graph_dir = graph_dir
        self.edges = edges
        self.mode = mode
        self.emb_dir = emb_dir
        self.labels = labels or {}

    def __len__(self):
        return len(self.accs)

    def __getitem__(self, i):
        a = self.accs[i]
        if self.mode == "gat":
            return _load_graph(self.graph_dir, a, edges=self.edges)
        return _load_mlp_record(self.emb_dir, a, self.labels[a])


# ---- Eval ------------------------------------------------------------------

def _per_label_metrics(y_true: np.ndarray, y_score: np.ndarray) -> list[dict]:
    rows = []
    for j, lab in enumerate(LABEL_COLUMNS):
        m = evaluate_binary(y_true[:, j].astype(int), y_score[:, j])
        rows.append({"label": lab, **m})
    return rows


def _evaluate(model, loader, device, model_kind: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if model_kind == "gat":
                logit = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            else:
                logit = model(batch.x, batch.batch)
            score = torch.sigmoid(logit).float().cpu().numpy()
            ys.append(batch.y.cpu().numpy())
            ss.append(score)
    if not ys:
        return np.zeros((0, NUM_LABELS)), np.zeros((0, NUM_LABELS))
    return np.concatenate(ys, 0), np.concatenate(ss, 0)


# ---- Train -----------------------------------------------------------------

@dataclass
class TrainConfig:
    model: str = "gat"
    edges: str = "both"
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    seed: int = 0


def _build_model(cfg: TrainConfig):
    from halo.gnn.model import GATLocalizer, MeanPoolMLP
    if cfg.model == "gat":
        return GATLocalizer(in_dim=640, hidden=cfg.hidden, heads=cfg.heads,
                            num_layers=cfg.layers, dropout=cfg.dropout,
                            num_classes=NUM_LABELS)
    elif cfg.model == "mlp_pool":
        return MeanPoolMLP(in_dim=640, hidden=max(cfg.hidden, 256),
                           dropout=cfg.dropout, num_classes=NUM_LABELS)
    else:
        raise ValueError(cfg.model)


def _pos_weight_from_labels(labels: dict[str, np.ndarray]) -> torch.Tensor:
    # Per-class neg/pos ratio used as BCEWithLogitsLoss pos_weight to up-weight rare compartments.
    ys = np.stack(list(labels.values()), axis=0)
    pos = ys.sum(0).clip(min=1.0)
    neg = (1.0 - ys).sum(0).clip(min=1.0)
    pw = neg / pos
    return torch.from_numpy(pw.astype(np.float32))


def fit(cfg: TrainConfig, graph_dir: Path, emb_dir: Path,
        train_csv: Path, val_csv: Path, test_csv: Path, out_dir: Path,
        device: str = "cuda", amp: bool = True,
        train_on_train_plus_val: bool = False) -> dict:
    from torch_geometric.loader import DataLoader as PygLoader

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    train_accs, train_labels = _split_accs(train_csv)
    val_accs, val_labels = _split_accs(val_csv)
    test_accs, test_labels = _split_accs(test_csv)

    if cfg.model == "gat":
        train_accs = _filter_existing(train_accs, graph_dir)
        val_accs = _filter_existing(val_accs, graph_dir)
        test_accs = _filter_existing(test_accs, graph_dir)
    else:
        train_accs = _filter_existing_emb(train_accs, emb_dir)
        val_accs = _filter_existing_emb(val_accs, emb_dir)
        test_accs = _filter_existing_emb(test_accs, emb_dir)

    log.info("filtered: train=%d val=%d test=%d", len(train_accs), len(val_accs), len(test_accs))

    # Stage 5d refit path: fold val into train and evaluate test once at the end.
    if train_on_train_plus_val:
        train_accs = train_accs + val_accs
        train_labels = {**train_labels, **val_labels}

    train_ds = GraphIterableDataset(train_accs, graph_dir, edges=cfg.edges,
                                    mode=cfg.model, emb_dir=emb_dir, labels=train_labels)
    val_ds = GraphIterableDataset(val_accs, graph_dir, edges=cfg.edges,
                                  mode=cfg.model, emb_dir=emb_dir, labels=val_labels)
    test_ds = GraphIterableDataset(test_accs, graph_dir, edges=cfg.edges,
                                   mode=cfg.model, emb_dir=emb_dir, labels=test_labels)

    train_loader = PygLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=0, pin_memory=(dev.type == "cuda"))
    val_loader = PygLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = PygLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = _build_model(cfg).to(dev)
    pw = _pos_weight_from_labels(train_labels).to(dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, cfg.epochs))

    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metrics_rows = []
    best_val = -math.inf
    best_state = None
    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0
        n = 0
        for batch in train_loader:
            batch = batch.to(dev)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if cfg.model == "gat":
                    logit = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                else:
                    logit = model(batch.x, batch.batch)
                loss = criterion(logit, batch.y.float())
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optim); scaler.update()
            else:
                loss.backward(); optim.step()
            loss_sum += float(loss.item()) * batch.num_graphs
            n += batch.num_graphs
        sched.step()
        if len(val_ds) > 0:
            yv, sv = _evaluate(model, val_loader, dev, cfg.model)
            val_per = _per_label_metrics(yv, sv)
            val_roc = float(np.nanmean([r["roc_auc"] for r in val_per]))
        else:
            val_roc = float("nan")
        metrics_rows.append({"epoch": epoch, "train_loss": loss_sum / max(1, n),
                             "val_roc_auc_macro": val_roc})
        log.info("epoch %d  loss=%.4f  val_roc=%.4f", epoch, loss_sum / max(1, n), val_roc)
        if val_roc > best_val:
            best_val = val_roc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Restore best-val checkpoint before the single test eval.
    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)

    # --- final test eval ---
    yt, st = _evaluate(model, test_loader, dev, cfg.model)
    test_per = _per_label_metrics(yt, st)
    model_name = f"{cfg.model}_{cfg.edges}"
    rows = [{"model": model_name, **r} for r in test_per]
    pd.DataFrame(rows).to_csv(out_dir / "test_results.csv", index=False)
    np.savez_compressed(out_dir / "test_curves.npz",
                        **{f"{model_name}__{LABEL_COLUMNS[j]}__y_true": yt[:, j].astype(int)
                           for j in range(NUM_LABELS)},
                        **{f"{model_name}__{LABEL_COLUMNS[j]}__y_score": st[:, j]
                           for j in range(NUM_LABELS)})
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__,
                "labels": LABEL_COLUMNS}, out_dir / "checkpoint.pt")
    summary = {"best_val_roc_macro": best_val,
               "test_roc_macro": float(np.nanmean([r["roc_auc"] for r in test_per])),
               "test_pr_macro": float(np.nanmean([r["pr_auc"] for r in test_per])),
               "test_f1_macro": float(np.nanmean([r["f1"] for r in test_per])),
               "n_train": len(train_accs), "n_val": len(val_accs), "n_test": len(test_accs)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done: %s", summary)
    return summary


def _cfg_from_args(args, json_path: Path | None = None) -> TrainConfig:
    cfg = TrainConfig(model=args.model, edges=args.edges, epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr,
                      weight_decay=args.weight_decay, seed=args.seed)
    if json_path is not None and json_path.exists():
        d = json.loads(json_path.read_text())
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", type=Path, default=Path("gnnData/graphs"))
    p.add_argument("--embedding-dir", type=Path, default=Path("gnnData/embeddings"))
    p.add_argument("--csvs-train", type=Path, default=Path("Data/deeploc_train.csv"))
    p.add_argument("--csvs-val", type=Path, default=Path("Data/deeploc_validation.csv"))
    p.add_argument("--csvs-test", type=Path, default=Path("Data/deeploc_test.csv"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", choices=["gat", "mlp_pool"], default="gat")
    p.add_argument("--edges", choices=["sequence", "contact", "both"], default="both")
    p.add_argument("--config-json", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-plus-val", action="store_true",
                   help="Stage 5d refit: train on train+val, eval on test once.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = _cfg_from_args(args, args.config_json)
    out = fit(cfg, args.graph_dir, args.embedding_dir, args.csvs_train,
              args.csvs_val, args.csvs_test, args.out,
              device=args.device, amp=args.amp,
              train_on_train_plus_val=args.train_plus_val)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
