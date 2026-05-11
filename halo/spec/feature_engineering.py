"""Per-protein sequence-derived features (CPU-only, instant).

For every protein we emit a fixed-length feature vector composed of:
  - length (1)
  - amino-acid composition: fraction of each of the 20 canonical amino acids (20)
  - mean hydrophobicity, charge, polarity, weight (4)
  - sequence entropy (1)
  - dipeptide-statistic summaries (4):
        fraction of hydrophobic-hydrophobic neighbours,
        fraction of charged-charged neighbours,
        fraction of polar-polar neighbours,
        fraction of identical neighbours.

Total: 30 features. This is the input the classical-model pipeline consumes.

Usage:
    python -m halo.spec.feature_engineering --csv Data/deeploc_train.csv \
        --out specData/sequence_features_train.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from halo.data.labels import AMINO_ACID_PROPERTIES, LABEL_COLUMNS

CANONICAL_AAS = "ACDEFGHIKLMNPQRSTVWY"          # 20 single-letter codes
THREE_LETTER = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
                "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]


def _per_aa_lookup(prop: str) -> dict[str, float]:
    return {one: AMINO_ACID_PROPERTIES[three][prop]
            for one, three in zip(CANONICAL_AAS, THREE_LETTER)}


HYDRO = _per_aa_lookup("hydrophobicity")
CHARGE = _per_aa_lookup("charge")
POLARITY = _per_aa_lookup("polarity")
WEIGHT = _per_aa_lookup("weight")


def features_for_sequence(seq: str) -> dict[str, float]:
    seq = seq.upper().replace("U", "C").replace("O", "K")  # selenocysteine / pyrrolysine
    # Drop any remaining non-canonical chars (e.g. X, B, Z, gaps).
    seq = "".join(c for c in seq if c in CANONICAL_AAS)
    n = len(seq)
    out: dict[str, float] = {"length": float(n)}
    if n == 0:
        for aa in CANONICAL_AAS:
            out[f"frac_{aa}"] = 0.0
        out.update({"mean_hydro": 0.0, "mean_charge": 0.0,
                    "mean_polarity": 0.0, "mean_weight": 0.0,
                    "seq_entropy": 0.0, "frac_neighbour_hydro": 0.0,
                    "frac_neighbour_charged": 0.0, "frac_neighbour_polar": 0.0,
                    "frac_neighbour_identical": 0.0})
        return out

    counts = {aa: 0 for aa in CANONICAL_AAS}
    for c in seq:
        counts[c] += 1
    fracs = {aa: counts[aa] / n for aa in CANONICAL_AAS}
    for aa in CANONICAL_AAS:
        out[f"frac_{aa}"] = fracs[aa]
    out["mean_hydro"] = sum(fracs[aa] * HYDRO[aa] for aa in CANONICAL_AAS)
    out["mean_charge"] = sum(fracs[aa] * CHARGE[aa] for aa in CANONICAL_AAS)
    out["mean_polarity"] = sum(fracs[aa] * POLARITY[aa] for aa in CANONICAL_AAS)
    out["mean_weight"] = sum(fracs[aa] * WEIGHT[aa] for aa in CANONICAL_AAS)
    out["seq_entropy"] = -sum(p * math.log(p) for p in fracs.values() if p > 0)

    # Dipeptide neighbour statistics
    is_hydro = {aa: HYDRO[aa] > 0 for aa in CANONICAL_AAS}
    is_charged = {aa: abs(CHARGE[aa]) > 0 for aa in CANONICAL_AAS}
    is_polar = {aa: POLARITY[aa] > 9 for aa in CANONICAL_AAS}
    if n < 2:
        out.update({"frac_neighbour_hydro": 0.0, "frac_neighbour_charged": 0.0,
                    "frac_neighbour_polar": 0.0, "frac_neighbour_identical": 0.0})
        return out
    pairs = list(zip(seq, seq[1:]))
    out["frac_neighbour_hydro"] = sum(is_hydro[a] and is_hydro[b] for a, b in pairs) / len(pairs)
    out["frac_neighbour_charged"] = sum(is_charged[a] and is_charged[b] for a, b in pairs) / len(pairs)
    out["frac_neighbour_polar"] = sum(is_polar[a] and is_polar[b] for a, b in pairs) / len(pairs)
    out["frac_neighbour_identical"] = sum(a == b for a, b in pairs) / len(pairs)
    return out


FEATURE_COLUMNS = (
    ["length"]
    + [f"frac_{aa}" for aa in CANONICAL_AAS]
    + ["mean_hydro", "mean_charge", "mean_polarity", "mean_weight", "seq_entropy",
       "frac_neighbour_hydro", "frac_neighbour_charged", "frac_neighbour_polar",
       "frac_neighbour_identical"]
)
NUM_FEATURES = len(FEATURE_COLUMNS)


def build_feature_frame(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna(subset=["Sequence"])
    feats = [features_for_sequence(s) for s in df["Sequence"].astype(str)]
    feat_df = pd.DataFrame(feats, columns=FEATURE_COLUMNS)
    keep = ["ACC"] + [c for c in ("Kingdom", "Partition", "Membrane") if c in df.columns]
    keep += LABEL_COLUMNS
    return pd.concat([df[keep].reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = build_feature_frame(args.csv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} rows × {len(out.columns)} cols → {args.out}")


if __name__ == "__main__":
    main()
