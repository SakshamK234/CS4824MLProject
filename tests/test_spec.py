"""Smoke tests for the halo.spec workflow track."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from halo.spec.classical_models import evaluate_binary, make_pipeline, run
from halo.spec.feature_engineering import (FEATURE_COLUMNS, NUM_FEATURES,
                                           features_for_sequence)
from halo.data.labels import LABEL_COLUMNS

REPO = Path(__file__).resolve().parents[1]


def test_feature_engineering():
    f = features_for_sequence("MAAVKLGRCDE")
    assert set(f.keys()) == set(FEATURE_COLUMNS)
    assert NUM_FEATURES == 30
    assert f["length"] == 11
    aa_frac_cols = [c for c in FEATURE_COLUMNS
                    if c.startswith("frac_") and not c.startswith("frac_neighbour_")]
    assert abs(sum(f[c] for c in aa_frac_cols) - 1.0) < 1e-9


def test_feature_engineering_handles_unknown():
    f = features_for_sequence("XXXX")  # all non-canonical → length-stripped
    assert f["length"] == 0


def test_pipeline_construction():
    from sklearn.linear_model import LogisticRegression
    pipe = make_pipeline(LogisticRegression(max_iter=50), k_features=10)
    steps = [n for n, _ in pipe.steps]
    assert steps == ["impute", "scale", "var", "select", "clf"]


def test_evaluate_binary():
    import numpy as np
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = rng.uniform(size=200)
    m = evaluate_binary(y, s)
    for k in ("roc_auc", "pr_auc", "f1", "n_pos", "n"):
        assert k in m


def test_run_end_to_end_tiny(tmp_path):
    """Run on a 200-row subset of the real specData with one model + one label."""
    spec = REPO / "specData"
    if not (spec / "sequence_features_train.csv").exists():
        import pytest
        pytest.skip("specData feature CSVs not built")
    # Sub-sample to keep the test fast
    for split in ("train", "validation", "test"):
        df = pd.read_csv(spec / f"sequence_features_{split}.csv")
        # Keep at least one positive row of the chosen label
        pos = df[df["Extracellular"] == 1].head(80)
        neg = df[df["Extracellular"] == 0].head(80)
        pd.concat([pos, neg]).to_csv(tmp_path / f"{split}.csv", index=False)
    info = run(tmp_path / "train.csv", tmp_path / "validation.csv",
               tmp_path / "test.csv", tmp_path / "out", k_features=15,
               cv_folds=3, models=["logreg"], labels=["Extracellular"])
    assert info["rows"] == 1
    res = pd.read_csv(tmp_path / "out" / "results.csv")
    assert res.iloc[0]["model"] == "logreg"
    # The pipeline ran end-to-end; values should be finite
    assert pd.notna(res.iloc[0]["roc_auc"])
