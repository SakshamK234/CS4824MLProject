# specData/

Pre-computed per-protein feature matrices for the class workflow (`SPEC.md`).
These let you run the full 7-step pipeline **on CPU in minutes**, with no GPU,
no AlphaFold downloads, and no ESM2 forward pass.

## Files

| File | Rows | Purpose |
|---|---|---|
| `sequence_features_train.csv`      | 17,145 | Training set |
| `sequence_features_validation.csv` |  5,696 | Validation set (used by GridSearchCV's K-fold) |
| `sequence_features_test.csv`       |  5,462 | Held-out test set (touched only at Step 7) |

Schema (44 columns):

```
ACC, Kingdom, Partition, Membrane,
Cytoplasm, Nucleus, Extracellular, Cell membrane, Mitochondrion, Plastid,
Endoplasmic reticulum, Lysosome/Vacuole, Golgi apparatus, Peroxisome,           # 10 multi-label targets
length,                                                                          # sequence length
frac_A, frac_C, ..., frac_Y,                                                     # 20 amino-acid composition fractions
mean_hydro, mean_charge, mean_polarity, mean_weight,                             # composition-weighted physicochemistry
seq_entropy,                                                                     # Shannon entropy of the AA distribution
frac_neighbour_hydro, frac_neighbour_charged,
frac_neighbour_polar, frac_neighbour_identical                                   # dipeptide-statistic summaries
```

The 30 numeric feature columns are listed in
[`halo/spec/feature_engineering.py:FEATURE_COLUMNS`](../halo/spec/feature_engineering.py).

## How they were generated

```bash
for split in train validation test; do
  python -m halo.spec.feature_engineering \
    --csv Data/deeploc_${split}.csv \
    --out specData/sequence_features_${split}.csv
done
```

Source data is `Data/deeploc_*.csv` (DeepLoc 2.0). The original sequences and
labels are unchanged — `specData/` adds engineered features alongside them, so
no `Data/` file is modified.

## Why these features instead of ESM embeddings

The class assignment requires CPU-runnable comparisons of multiple models. The
30 sequence-derived features are:
- **fast**: every CSV is built in ~10 seconds on a laptop;
- **reproducible**: deterministic, no model checkpoints or randomness;
- **leakage-free**: a column-wise function of the sequence — no per-sample
  cross-talk;
- **interpretable**: every feature has a direct biochemical meaning.

These match the *numerical* role of "molecular features" in the example
breast-cancer workflow. For the deep-learning route (HALO-GAT, ESM-with-
Attention, GraphTransformer, CrossAttentionFusion) you still need to run
`make embeddings` and `make graphs` once on a GPU. See `SPEC.md`, §"Model
panel" and §"Compute requirements".

## Optional add-ons (not auto-generated)

If the GPU/network steps below are run, these additional files can live here:

| File | Producer | Cost |
|---|---|---|
| `esm_pooled_train/val/test.parquet` | mean-pool of `cache/embeddings/*.npy` | GPU |
| `structural_summary_*.csv` (radius of gyration, contact density, mean pLDDT) | `cache/graphs/*.pt` | CPU once graphs exist |
| `motif_annotations.parquet` | `python -m halo.data.motif_annotations` | network |

Each is fully optional — `halo.spec` works with `sequence_features_*.csv` alone.
