"""K-features sweep on top of classical_models.run.

For each K in the sweep, runs the classical panel (subset of models / labels),
collects results into a single table keyed by (K, model, label), writes a
combined CSV and a bar chart of mean ROC-AUC across compartments vs K.

Usage:
    python -m halo.spec.kfeatures_sweep \
        --train specData/sequence_features_train.csv \
        --val   specData/sequence_features_validation.csv \
        --test  specData/sequence_features_test.csv \
        --out   runs/spec_kfeatures_sweep \
        --models logreg rf \
        --k-values 10 20 30
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from halo.spec.classical_models import run

log = logging.getLogger(__name__)


def sweep(train_csv: Path, val_csv: Path, test_csv: Path, out_dir: Path,
          k_values: list[int], models: list[str], cv_folds: int = 5,
          labels: list[str] | None = None, seed: int = 0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined: list[pd.DataFrame] = []
    for k in k_values:
        sub_dir = out_dir / f"k{k}"
        log.info("=== K=%d ===", k)
        run(train_csv, val_csv, test_csv, sub_dir,
            k_features=k, cv_folds=cv_folds, models=models, labels=labels, seed=seed)
        df = pd.read_csv(sub_dir / "results.csv")
        df["k_features"] = k
        combined.append(df)
    full = pd.concat(combined, ignore_index=True)
    full.to_csv(out_dir / "results.csv", index=False)

    summary = (full.groupby(["k_features", "model"])[["roc_auc", "pr_auc", "f1"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(out_dir / "summary.csv")
    return out_dir / "results.csv"


def bar_chart(results_csv: Path, out_dir: Path) -> Path:
    df = pd.read_csv(results_csv)
    pivot = (df.groupby(["k_features", "model"])["roc_auc"].mean().unstack("model"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean ROC-AUC across compartments")
    ax.set_xlabel("K (SelectKBest)")
    ax.set_title("Mean ROC-AUC vs feature-selection K")
    ax.set_ylim(0.5, 1.0)
    ax.legend(title="Model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = out_dir / "bar_kfeatures_roc_auc.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", default="runs/spec_kfeatures_sweep")
    p.add_argument("--k-values", nargs="+", type=int, default=[10, 20, 30])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    results_csv = sweep(Path(args.train), Path(args.val), Path(args.test), out,
                        args.k_values, args.models, args.cv_folds, args.labels, args.seed)
    bar_path = bar_chart(results_csv, out)
    print(json.dumps({"results_csv": str(results_csv), "bar_chart": str(bar_path)}, indent=2))


if __name__ == "__main__":
    main()
