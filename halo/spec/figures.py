"""Figures for the spec workflow report.

Reads classical_models.run output (results.csv + curves.npz) and produces:
  - bar charts of ROC-AUC / F1 / PR-AUC per model, per label
  - overlaid ROC curves per label
  - overlaid PR curves per label
  - a model-vs-label heatmap of ROC-AUC

Usage:
    python -m halo.spec.figures --run-dir runs/spec_classical --out runs/spec_classical/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def bar_by_model(results_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(results_csv)
    for metric in ("roc_auc", "pr_auc", "f1"):
        pivot = df.pivot(index="label", columns="model", values=metric)
        ax = pivot.plot(kind="bar", figsize=(11, 5))
        ax.set_ylabel(metric)
        ax.set_title(f"Test {metric} per compartment, per model")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / f"bar_{metric}.png", dpi=150)
        plt.close()


def heatmap(results_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(results_csv)
    pivot = df.pivot(index="model", columns="label", values="roc_auc")
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "—",
                    ha="center", va="center", fontsize=8,
                    color="white" if (np.isfinite(v) and v < 0.75) else "black")
    fig.colorbar(im, ax=ax, fraction=0.025).set_label("ROC-AUC")
    ax.set_title("ROC-AUC heatmap (model × compartment)")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_roc_auc.png", dpi=150)
    plt.close()


def roc_overlay(curves_npz: Path, out_dir: Path) -> None:
    data = np.load(curves_npz)
    keys = sorted({"__".join(k.split("__")[:-1]) for k in data.files})
    by_label: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    for k in keys:
        model, label = k.split("__")
        y_true = data[f"{k}__y_true"]
        y_score = data[f"{k}__y_score"]
        if len(np.unique(y_true)) < 2:
            continue
        by_label.setdefault(label, []).append((model, y_true, y_score))
    for label, group in by_label.items():
        fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(11, 4.2))
        for model, y_true, y_score in group:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax_r.plot(fpr, tpr, label=model, lw=1.2)
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            ax_p.plot(rec, prec, label=model, lw=1.2)
        ax_r.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR"); ax_r.set_title(f"ROC — {label}"); ax_r.legend(fontsize=8)
        ax_p.set_xlabel("Recall"); ax_p.set_ylabel("Precision"); ax_p.set_title(f"PR — {label}"); ax_p.legend(fontsize=8)
        fig.tight_layout()
        safe = label.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"curves_{safe}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    run = Path(args.run_dir)
    bar_by_model(run / "results.csv", out)
    heatmap(run / "results.csv", out)
    if (run / "curves.npz").exists():
        roc_overlay(run / "curves.npz", out)
    print(f"figures → {out}")


if __name__ == "__main__":
    main()
