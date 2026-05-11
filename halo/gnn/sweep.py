"""Stage 5d: random hyperparameter search over the GAT.

Samples 8 configs uniformly at random with a fixed seed across:
- layers in {2, 3}
- hidden in {64, 128}    (must be divisible by heads)
- heads in {4, 8}        (drop combos that violate hidden % heads != 0)
- dropout in {0.0, 0.2}
- lr in {1e-3, 3e-4}
Contact threshold is fixed at 8 A (rebuilding graphs three times is too
expensive; we document this in REPORT).

Each config trains on train, evaluates on val. Picks best by macro
val ROC-AUC. Writes per-config rows to <out>/configs.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd

from halo.gnn.train import TrainConfig, fit

log = logging.getLogger(__name__)


def _sample_configs(seed: int, n: int = 8) -> list[dict]:
    rng = random.Random(seed)
    space = {
        "layers": [2, 3],
        "hidden": [64, 128],
        "heads": [4, 8],
        "dropout": [0.0, 0.2],
        "lr": [1e-3, 3e-4],
    }
    seen, out = set(), []
    attempts = 0
    while len(out) < n and attempts < 200:
        attempts += 1
        c = {k: rng.choice(v) for k, v in space.items()}
        # GATv2Conv requires hidden to split evenly across heads.
        if c["hidden"] % c["heads"] != 0:
            continue
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", type=Path, default=Path("gnnData/graphs"))
    p.add_argument("--embedding-dir", type=Path, default=Path("gnnData/embeddings"))
    p.add_argument("--csvs-train", type=Path, default=Path("Data/deeploc_train.csv"))
    p.add_argument("--csvs-val", type=Path, default=Path("Data/deeploc_validation.csv"))
    p.add_argument("--csvs-test", type=Path, default=Path("Data/deeploc_test.csv"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-configs", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    configs = _sample_configs(args.seed, args.n_configs)
    log.info("sampled %d configs", len(configs))

    rows = []
    best_idx, best_val = -1, -1.0
    for i, c in enumerate(configs):
        cfg = TrainConfig(model="gat", edges="both", epochs=args.epochs,
                          batch_size=args.batch_size, lr=c["lr"],
                          hidden=c["hidden"], heads=c["heads"], layers=c["layers"],
                          dropout=c["dropout"], seed=args.seed)
        sub_out = args.out / f"cfg_{i:02d}"
        log.info("config %d: %s", i, c)
        summary = fit(cfg, args.graph_dir, args.embedding_dir, args.csvs_train,
                      args.csvs_val, args.csvs_test, sub_out,
                      device=args.device, amp=args.amp,
                      train_on_train_plus_val=False)
        row = {"config_id": i, **c, "val_roc_macro": summary["best_val_roc_macro"],
               "test_roc_macro": summary["test_roc_macro"],
               "test_pr_macro": summary["test_pr_macro"]}
        rows.append(row)
        if row["val_roc_macro"] > best_val:
            best_val = row["val_roc_macro"]; best_idx = i

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "configs.csv", index=False)
    chosen = configs[best_idx] if best_idx >= 0 else {}
    chosen["config_id"] = best_idx
    chosen["val_roc_macro"] = best_val
    (args.out / "chosen_config.json").write_text(json.dumps(chosen, indent=2))
    log.info("best config %d: %s (val_roc=%.4f)", best_idx, chosen, best_val)
    return 0


if __name__ == "__main__":
    sys.exit(main())
