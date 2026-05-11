"""Figures for the GNN track.

Required outputs (PLAN_GNN.md Stage 7):
  1. ablation_bar.png       five-row bar chart, mean ROC-AUC across compartments
  2. heatmap_with_gat.png   ROC-AUC heatmap including GAT row
  3. curves_with_gat_<L>.png ROC and PR overlay per compartment, with GAT line
  4. sweep_val_curves.png   val ROC-AUC across the 8 tuning configs
  5. importance_summary.png per-compartment top-importance-position histogram
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

log = logging.getLogger(__name__)


def _read_classical_results(classical_dir: Path) -> pd.DataFrame:
    p = classical_dir / "results.csv"
    if not p.exists():
        return pd.DataFrame(columns=["model", "label", "roc_auc", "pr_auc", "f1"])
    df = pd.read_csv(p)
    return df[["model", "label", "roc_auc", "pr_auc", "f1"]].copy()


def _read_gnn_row(row_dir: Path) -> pd.DataFrame:
    p = row_dir / "test_results.csv"
    if not p.exists():
        return pd.DataFrame(columns=["model", "label", "roc_auc", "pr_auc", "f1"])
    df = pd.read_csv(p)
    return df[["model", "label", "roc_auc", "pr_auc", "f1"]].copy()


def ablation_bar(classical_df: pd.DataFrame, ablation_rows, out_path: Path):
    means = []
    if not classical_df.empty:
        best = classical_df.groupby("model")["roc_auc"].mean().idxmax()
        cdf = classical_df[classical_df["model"] == best]
        means.append((f"classical:{best}", float(cdf["roc_auc"].mean())))
    for name, df in ablation_rows:
        if df.empty:
            continue
        means.append((name, float(df["roc_auc"].mean())))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([m[0] for m in means], [m[1] for m in means], color="steelblue")
    ax.set_ylabel("Mean test ROC-AUC across 10 compartments")
    ax.set_ylim(0, 1)
    ax.set_title("Ablation rows")
    for i, v in enumerate([m[1] for m in means]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def heatmap_with_gat(combined: pd.DataFrame, out_path: Path):
    if combined.empty:
        return
    combined = combined.copy()
    combined["roc_auc"] = pd.to_numeric(combined["roc_auc"], errors="coerce")
    pivot = combined.pivot_table(index="model", columns="label", values="roc_auc", aggfunc="mean")
    pivot = pivot.astype(float)
    fig, ax = plt.subplots(figsize=(11, max(3, 0.5 * pivot.shape[0] + 1)))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "-",
                    ha="center", va="center", fontsize=8,
                    color="white" if (np.isfinite(v) and v < 0.75) else "black")
    fig.colorbar(im, ax=ax, fraction=0.025).set_label("ROC-AUC")
    ax.set_title("ROC-AUC heatmap (model x compartment) with GAT row")
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _load_curves(npz_path: Path) -> dict:
    if not npz_path.exists():
        return {}
    d = np.load(npz_path)
    out: dict = {}
    for k in d.files:
        if k.endswith("__y_true"):
            base = k[:-len("__y_true")]
            mdl, lbl = base.split("__", 1)
            out.setdefault(mdl, {}).setdefault(lbl, {})["y_true"] = d[k]
        elif k.endswith("__y_score"):
            base = k[:-len("__y_score")]
            mdl, lbl = base.split("__", 1)
            out.setdefault(mdl, {}).setdefault(lbl, {})["y_score"] = d[k]
    return out


def curves_with_gat(classical_dir: Path, gnn_curves, out_dir: Path, labels):
    classical = _load_curves(classical_dir / "curves.npz")
    gnn_all: dict = {}
    for p in gnn_curves:
        gnn_all.update(_load_curves(p))
    for label in labels:
        fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(11, 4.2))
        any_data = False
        for mdl, ld in classical.items():
            if label not in ld:
                continue
            yt, ys = ld[label].get("y_true"), ld[label].get("y_score")
            if yt is None or ys is None or len(np.unique(yt)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yt, ys); ax_r.plot(fpr, tpr, label=mdl, lw=1.0, alpha=0.7)
            prec, rec, _ = precision_recall_curve(yt, ys); ax_p.plot(rec, prec, label=mdl, lw=1.0, alpha=0.7)
            any_data = True
        for mdl, ld in gnn_all.items():
            if label not in ld:
                continue
            yt, ys = ld[label].get("y_true"), ld[label].get("y_score")
            if yt is None or ys is None or len(np.unique(yt)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yt, ys); ax_r.plot(fpr, tpr, label=mdl, lw=1.6)
            prec, rec, _ = precision_recall_curve(yt, ys); ax_p.plot(rec, prec, label=mdl, lw=1.6)
            any_data = True
        if not any_data:
            plt.close(fig); continue
        ax_r.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR"); ax_r.set_title(f"ROC - {label}"); ax_r.legend(fontsize=7)
        ax_p.set_xlabel("Recall"); ax_p.set_ylabel("Precision"); ax_p.set_title(f"PR - {label}"); ax_p.legend(fontsize=7)
        slug = label.replace("/", "_").replace(" ", "_")
        plt.tight_layout(); fig.savefig(out_dir / f"curves_with_gat_{slug}.png", dpi=150); plt.close(fig)


def sweep_val_curves(sweep_dir: Path, out_path: Path):
    p = sweep_dir / "configs.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(df["config_id"].astype(str), df["val_roc_macro"], color="steelblue")
    chosen = sweep_dir / "chosen_config.json"
    if chosen.exists():
        c = json.loads(chosen.read_text())
        cid = c.get("config_id", -1)
        for b, row in zip(bars, df.itertuples()):
            if int(row.config_id) == int(cid):
                b.set_color("orange")
    ax.set_xlabel("config_id"); ax.set_ylabel("Macro val ROC-AUC")
    ax.set_title("Random search: val ROC-AUC across configs")
    ax.set_ylim(0, 1)
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def importance_summary(explanations_dir: Path, out_path: Path):
    npz = explanations_dir / "attention_weights.npz"
    if not npz.exists():
        return
    d = np.load(npz)
    by_label: dict = {}
    for k in d.files:
        label, _ = k.split("__", 1)
        by_label.setdefault(label, []).append(d[k])
    if not by_label:
        return
    n = max(1, len(by_label))
    cols = min(5, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.2 * rows), sharey=True)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).ravel()
    for ax, (label, arrs) in zip(axes_flat, by_label.items()):
        bins = np.zeros(20)
        for a in arrs:
            if len(a) == 0:
                continue
            pos = np.linspace(0, 1, len(a))
            idx = np.clip((pos * 20).astype(int), 0, 19)
            for j, v in zip(idx, a):
                bins[j] += float(v)
        if bins.sum() > 0:
            bins = bins / bins.sum()
        ax.bar(np.arange(20), bins, color="steelblue")
        ax.set_title(label, fontsize=8)
        ax.set_xticks([0, 10, 19]); ax.set_xticklabels(["N", "mid", "C"], fontsize=7)
    for ax in axes_flat[len(by_label):]:
        ax.axis("off")
    fig.suptitle("Per-compartment node-importance distribution along sequence")
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classical-dir", type=Path, default=Path("runs/spec_classical"))
    p.add_argument("--gnn-dir", type=Path, default=Path("runs/gnn"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    classical_df = _read_classical_results(args.classical_dir)
    ablation_rows = []
    for sub in ["mlp_pool", "gat_seq", "gat_contact"]:
        ablation_rows.append((f"ablation:{sub}", _read_gnn_row(args.gnn_dir / "ablation" / sub)))
    headline_df = _read_gnn_row(args.gnn_dir / "headline")
    if not headline_df.empty:
        ablation_rows.append(("headline:gat_both", headline_df))

    ablation_bar(classical_df, ablation_rows, out / "ablation_bar.png")

    parts = [classical_df]
    for name, df in ablation_rows:
        if not df.empty:
            df = df.copy(); df["model"] = name
            parts.append(df[["model", "label", "roc_auc", "pr_auc", "f1"]])
    if parts:
        combined = pd.concat(parts, ignore_index=True)
        heatmap_with_gat(combined, out / "heatmap_with_gat.png")
        combined.to_csv(out / "combined_results.csv", index=False)

    gnn_curves = []
    for sub in ["mlp_pool", "gat_seq", "gat_contact"]:
        gnn_curves.append(args.gnn_dir / "ablation" / sub / "test_curves.npz")
    gnn_curves.append(args.gnn_dir / "headline" / "test_curves.npz")
    curves_with_gat(args.classical_dir, gnn_curves, out, ["Extracellular", "Peroxisome"])

    sweep_val_curves(args.gnn_dir / "sweep", out / "sweep_val_curves.png")
    importance_summary(args.gnn_dir / "explanations", out / "importance_summary.png")
    log.info("figures written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
