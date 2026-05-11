"""Classical multi-label classifiers for the spec workflow.

Trains a panel of sklearn models, with K-fold CV hyperparameter tuning on the
training set, and reports test-set ROC-AUC / F1 / PR-AUC per compartment.

Workflow alignment (Example_Workflow.docx):
  Step 5: StandardScaler fit on TRAIN ONLY, applied to TEST.
  Step 6: Feature selection (variance threshold + optional univariate top-K) +
          K-fold CV via GridSearchCV.
  Step 7: Final evaluation on the held-out test set.

Usage:
    python -m halo.spec.classical_models \
        --train specData/sequence_features_train.csv \
        --val   specData/sequence_features_validation.csv \
        --test  specData/sequence_features_test.csv \
        --out   runs/spec_classical
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from halo.data.labels import LABEL_COLUMNS
from halo.spec.feature_engineering import FEATURE_COLUMNS

log = logging.getLogger(__name__)


def _model_grid() -> dict[str, tuple[Any, dict]]:
    """Each entry: (estimator, GridSearchCV param grid keyed off pipeline step name)."""
    return {
        "logreg": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "svm_rbf": (
            SVC(kernel="rbf", probability=True, class_weight="balanced"),
            {"clf__C": [0.5, 1.0, 4.0], "clf__gamma": ["scale", 0.01, 0.001]},
        ),
        "rf": (
            RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced",
                                   random_state=0),
            {"clf__max_depth": [None, 8, 16], "clf__min_samples_leaf": [1, 4]},
        ),
        "mlp": (
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=0,
                          early_stopping=True),
            {"clf__alpha": [1e-4, 1e-3], "clf__learning_rate_init": [1e-3, 1e-2]},
        ),
    }


def make_pipeline(estimator, k_features: int | str = 30) -> Pipeline:
    """Step 5 (impute + scale) and Step 6 (variance + univariate top-K) live here."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("var", VarianceThreshold(threshold=0.0)),
        ("select", SelectKBest(score_func=mutual_info_classif,
                               k="all" if k_features == "all" else int(k_features))),
        ("clf", estimator),
    ])


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "f1": float(f1_score(y_true, y_pred, zero_division=1)),
                "n_pos": int(y_true.sum()), "n": len(y_true)}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=1)),
        "n_pos": int(y_true.sum()),
        "n": len(y_true),
    }


def run(train_csv: Path, val_csv: Path, test_csv: Path, out_dir: Path,
        k_features: int = 30, cv_folds: int = 5, models: list[str] | None = None,
        labels: list[str] | None = None, seed: int = 0) -> dict:
    """Drive the full 7-step pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(train_csv).dropna(subset=LABEL_COLUMNS)
    val_df = pd.read_csv(val_csv).dropna(subset=LABEL_COLUMNS)
    test_df = pd.read_csv(test_csv).dropna(subset=LABEL_COLUMNS)
    # Step 4 confirmation: training and test pools come from disjoint files; partition column
    # in DeepLoc was constructed by 30%-IDY clustering. We do not re-shuffle.

    # Train+val pooled: GridSearchCV does its own internal K-fold split for tuning.
    full_train = pd.concat([train_df, val_df], ignore_index=True)
    feat_cols = [c for c in FEATURE_COLUMNS if c in full_train.columns]
    X_tr = full_train[feat_cols].to_numpy()
    X_te = test_df[feat_cols].to_numpy()

    label_set = labels or LABEL_COLUMNS
    grid = _model_grid()
    selected = list(grid) if models is None else models
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    all_results: list[dict] = []
    curves: dict[str, dict[str, dict]] = {}  # model -> label -> {fpr,tpr,prec,rec,scores,labels}
    best_params: dict[str, dict[str, dict]] = {}

    for label in label_set:
        log.info("--- label: %s ---", label)
        y_tr = full_train[label].astype(int).to_numpy()
        y_te = test_df[label].astype(int).to_numpy()
        for model_name in selected:
            est, param_grid = grid[model_name]
            pipe = make_pipeline(est, k_features=k_features)
            search = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=skf,
                                  n_jobs=-1, refit=True, error_score="raise")
            search.fit(X_tr, y_tr)
            try:
                y_score = search.predict_proba(X_te)[:, 1]
            except AttributeError:
                # SVC without probability=True: use the unnormalized margin instead.
                y_score = search.decision_function(X_te)
            metrics = evaluate_binary(y_te, y_score)
            metrics.update({"label": label, "model": model_name,
                            "best_cv_score": float(search.best_score_),
                            **{f"hp_{k}": v for k, v in search.best_params_.items()}})
            all_results.append(metrics)

            curves.setdefault(model_name, {})[label] = {
                "y_true": y_te.tolist(),
                "y_score": y_score.tolist(),
            }
            best_params.setdefault(model_name, {})[label] = {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in search.best_params_.items()
            }
            log.info("%-8s  %-25s  ROC=%.3f  PR=%.3f  F1=%.3f",
                     model_name, label, metrics["roc_auc"], metrics["pr_auc"], metrics["f1"])

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "results.csv", index=False)
    np.savez_compressed(out_dir / "curves.npz",
                        **{f"{m}__{l}__y_true": np.asarray(v["y_true"])
                           for m, ld in curves.items() for l, v in ld.items()},
                        **{f"{m}__{l}__y_score": np.asarray(v["y_score"])
                           for m, ld in curves.items() for l, v in ld.items()})
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    summary = (results_df.groupby("model")[["roc_auc", "pr_auc", "f1"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(out_dir / "summary_by_model.csv")
    return {"results_csv": str(out_dir / "results.csv"),
            "summary_csv": str(out_dir / "summary_by_model.csv"),
            "rows": len(results_df), "models": selected, "labels": label_set}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", default="runs/spec_classical")
    p.add_argument("--k-features", type=int, default=30)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of {logreg, svm_rbf, rf, mlp}.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Subset of label columns to evaluate.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    info = run(Path(args.train), Path(args.val), Path(args.test), Path(args.out),
               args.k_features, args.cv_folds, args.models, args.labels, args.seed)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
