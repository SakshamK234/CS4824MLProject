"""Post-hoc interpretability via GNNExplainer + raw attention weights.

Run on the headline GAT checkpoint only. For each compartment, sample N test
proteins predicted positive with high confidence; produce per-residue
node-importance bars, raw GATv2 attention, and a motif-overlap summary.

Motifs:
- Peroxisome  : PTS1 = [SAC]K[LMI] in C-terminal 6 residues
- ER          : KDEL or HDEL in C-terminal 4 residues
- Mitochondrion / Plastid : top-10% importance lies in the first 70 residues
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from halo.data.labels import LABEL_COLUMNS

log = logging.getLogger(__name__)

PEROXISOME_RE = re.compile(r"[SAC]K[LMI]$")
ER_RE = re.compile(r"(KDEL|HDEL)$")
N_TERMINAL_WINDOW = 70


def _has_motif(label: str, seq: str) -> bool:
    if not seq:
        return False
    if label == "Peroxisome":
        return bool(PEROXISOME_RE.search(seq[-6:]))
    if label == "Endoplasmic reticulum":
        return bool(ER_RE.search(seq[-4:]))
    return False


def _topk_in_window(node_importance: np.ndarray, k_frac: float = 0.1,
                    window: int = N_TERMINAL_WINDOW) -> float:
    if len(node_importance) == 0:
        return 0.0
    k = max(1, int(round(k_frac * len(node_importance))))
    top_idx = np.argsort(node_importance)[-k:]
    return float(np.mean(top_idx < window))


def _explain_one(model, data, target_idx: int, device, epochs: int = 100):
    """Run GNNExplainer for a single graph and target class. Returns
    node-importance numpy array of length L."""
    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
    try:
        explanation = explainer(
            x=data.x.to(device),
            edge_index=data.edge_index.to(device),
            edge_type=data.edge_type.to(device),
            batch=batch,
            target=torch.tensor([target_idx], device=device),
        )
        node_mask = explanation.node_mask
        if node_mask is None:
            return None
        # If node_mask is per-feature, collapse to per-node.
        if node_mask.dim() == 2:
            return node_mask.detach().abs().sum(dim=-1).cpu().numpy()
        return node_mask.detach().abs().cpu().numpy()
    except Exception as e:
        log.warning("explainer failed: %s", e)
        return None


def _attention_node_importance(model, data, device) -> np.ndarray:
    """Approximate per-node importance from GATv2 attention by averaging
    incoming attention across heads and layers."""
    model.eval()
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(data.x.to(device), data.edge_index.to(device),
                  data.edge_type.to(device), batch, return_attention=True)
    L = data.x.shape[0]
    importance = np.zeros(L, dtype=np.float32)
    for ei, alpha in model._last_attention:
        # alpha: [E_total, heads]; ei: [2, E_total]; sum incoming
        a = alpha.float().mean(dim=-1).numpy()
        dst = ei[1].numpy()
        for j, d in enumerate(dst):
            if 0 <= d < L:
                importance[d] += float(a[j])
    if importance.max() > 0:
        importance = importance / importance.max()
    return importance


def _per_protein_plot(acc: str, label: str, importance: np.ndarray, seq: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.bar(np.arange(len(importance)), importance, width=1.0, color="steelblue")
    ax.set_xlabel("residue index")
    ax.set_ylabel("importance")
    ax.set_title(f"{acc} - {label}")
    if label == "Peroxisome" and seq:
        m = PEROXISOME_RE.search(seq[-6:])
        if m:
            start = len(seq) - 6 + m.start()
            ax.axvspan(start, start + (m.end() - m.start()), color="red", alpha=0.3, label="PTS1")
            ax.legend(loc="upper left", fontsize=8)
    elif label == "Endoplasmic reticulum" and seq:
        m = ER_RE.search(seq[-4:])
        if m:
            start = len(seq) - 4 + m.start()
            ax.axvspan(start, start + 4, color="red", alpha=0.3, label="KDEL/HDEL")
            ax.legend(loc="upper left", fontsize=8)
    elif label in ("Mitochondrion", "Plastid"):
        ax.axvspan(0, min(N_TERMINAL_WINDOW, len(importance)), color="orange", alpha=0.2,
                   label="N-term window")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def explain_all(checkpoint_path: Path, graph_dir: Path, emb_dir: Path,
                test_csv: Path, out_dir: Path, proteins_per_class: int = 50,
                device: str = "cuda", epochs: int = 100) -> dict:
    from halo.gnn.model import GATLocalizer
    from halo.gnn.train import _load_graph, _split_accs, _filter_existing

    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = ckpt["config"]
    model = GATLocalizer(in_dim=640, hidden=cfg.get("hidden", 128),
                         heads=cfg.get("heads", 4),
                         num_layers=cfg.get("layers", 2),
                         dropout=cfg.get("dropout", 0.1))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)

    test_accs, test_labels = _split_accs(test_csv)
    test_accs = _filter_existing(test_accs, graph_dir)

    # Build sequence lookup from embeddings cache.
    def _seq_for(acc: str) -> str:
        p = emb_dir / f"{acc}.pt"
        if not p.exists():
            return ""
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            return obj.get("seq", "")
        except Exception:
            return ""

    # Score every test graph once to pick high-confidence positives per class.
    log.info("scoring %d test graphs to pick top proteins per compartment", len(test_accs))
    scores = np.zeros((len(test_accs), len(LABEL_COLUMNS)), dtype=np.float32)
    truths = np.zeros((len(test_accs), len(LABEL_COLUMNS)), dtype=np.float32)
    for i, acc in enumerate(test_accs):
        try:
            data = _load_graph(graph_dir, acc, edges=cfg.get("edges", "both"))
        except Exception:
            continue
        batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=dev)
        with torch.no_grad():
            logit = model(data.x.to(dev), data.edge_index.to(dev),
                          data.edge_type.to(dev), batch)
        scores[i] = torch.sigmoid(logit).cpu().numpy()[0]
        truths[i] = data.y.numpy()[0]

    summaries = {}
    attention_records = {}
    for ci, label in enumerate(LABEL_COLUMNS):
        comp_dir = out_dir / label.replace("/", "_").replace(" ", "_")
        comp_dir.mkdir(parents=True, exist_ok=True)
        # Pick proteins where the model predicts positive AND ground truth is positive.
        cand = np.where((truths[:, ci] > 0.5) & (scores[:, ci] > 0.5))[0]
        # Highest-confidence true positives first.
        order = np.argsort(-scores[cand, ci])[:proteins_per_class]
        chosen = [test_accs[k] for k in cand[order]]

        motif_hits = 0
        nterm_hits = 0
        importances_topidx = []
        for acc in chosen:
            try:
                data = _load_graph(graph_dir, acc, edges=cfg.get("edges", "both"))
            except Exception:
                continue
            seq = _seq_for(acc)
            imp = _explain_one(model, data, ci, dev, epochs=epochs)
            if imp is None:
                imp = _attention_node_importance(model, data, dev)
            attn = _attention_node_importance(model, data, dev)
            attention_records[f"{label}__{acc}"] = attn.astype(np.float32)
            _per_protein_plot(acc, label, imp, seq, comp_dir / f"{acc}.png")
            if label in ("Peroxisome", "Endoplasmic reticulum") and _has_motif(label, seq):
                motif_hits += 1
            if label in ("Mitochondrion", "Plastid") and _topk_in_window(imp) > 0.5:
                nterm_hits += 1
            if len(imp):
                k = max(1, int(round(0.05 * len(imp))))
                importances_topidx.append(np.argsort(imp)[-k:].tolist())

        summary = {
            "label": label,
            "n_explained": len(chosen),
            "motif_hits": motif_hits,
            "nterm_hits": nterm_hits,
            "motif_overlap_rate": (motif_hits / len(chosen)) if chosen and label in
                ("Peroxisome", "Endoplasmic reticulum") else None,
            "nterm_overlap_rate": (nterm_hits / len(chosen)) if chosen and label in
                ("Mitochondrion", "Plastid") else None,
        }
        (comp_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries[label] = summary
        log.info("%s: %s", label, summary)

    np.savez_compressed(out_dir / "attention_weights.npz", **attention_records)
    (out_dir / "_summary.json").write_text(json.dumps(summaries, indent=2))
    return summaries


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph-dir", type=Path, default=Path("gnnData/graphs"))
    p.add_argument("--embedding-dir", type=Path, default=Path("gnnData/embeddings"))
    p.add_argument("--csv-test", type=Path, default=Path("Data/deeploc_test.csv"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--proteins-per-class", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    explain_all(args.checkpoint, args.graph_dir, args.embedding_dir, args.csv_test,
                args.out, proteins_per_class=args.proteins_per_class,
                device=args.device, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
