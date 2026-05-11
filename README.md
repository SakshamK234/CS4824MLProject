# halo-class: Reproducing the Results

Multi-label subcellular-localization on DeepLoc 2.0. Two parallel tracks:

- **spec**: classical models on hand-engineered sequence features (CPU only, minutes).
- **gnn**: ESM2 + AlphaFold-contact GAT with GNNExplainer attribution (GPU recommended).

Every command below assumes you start from the repo root.

## 1. Environment

Python 3.10+ required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # core deps + the halo package
pip install -r requirements-gnn.txt    # only needed for the GNN track
```

For the GNN track on GPU, install a CUDA-matched PyTorch first, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Sanity check:

```bash
make test
```

## 2. Data

The DeepLoc 2.0 CSVs ship with the repo under [Data/](Data) (train / validation / test).
Pre-built engineered features for the spec track are in [specData/](specData).
No manual download is required for the spec track.

The GNN track additionally needs:

- AlphaFold predicted PDBs → `gnnData/structures/` (downloaded by `make gnn-download`).
- Cached ESM2-150M per-residue embeddings → `gnnData/embeddings/` (built by `make gnn-embed`).
- Per-protein torch_geometric graphs → `gnnData/graphs/` (built by `make gnn-graphs`).

## 3. Reproduce the spec (classical) results

```bash
make spec-features      # only if you want to rebuild specData/*.csv from raw CSVs
make spec-classical     # 4 models × 10 compartments, 5-fold CV grid search
make spec-figures       # bar charts, ROC-AUC heatmap, ROC/PR overlays
```

Outputs land in `runs/spec_classical/`:

- `results.csv`: per-(model, label) test ROC-AUC / PR-AUC / F1
- `summary_by_model.csv`: macro means across compartments
- `best_params.json`: chosen hyperparameters per model and label
- `curves.npz`: raw y_true / y_score arrays for plotting
- `figures/`: PNGs

Optional K-features sweep:

```bash
make spec-kfeatures-sweep MODELS="logreg rf" K_VALUES="10 20 30"
```

## 4. Reproduce the GNN results

Stages are split so each is resumable. The download and embed stages are the long ones; everything is idempotent.

```bash
make gnn-download       # AlphaFold PDBs → gnnData/structures/
make gnn-embed          # ESM2-150M per-residue embeddings → gnnData/embeddings/
make gnn-graphs         # torch_geometric Data objects → gnnData/graphs/
```

Ablation rows (Stage 5):

```bash
make gnn-ablate         # mlp_pool, gat_seq, gat_contact
```

Hyperparameter search (Stage 5d) and final headline run on the chosen config, refit on train+val:

```bash
make gnn-sweep          # writes runs/gnn/sweep/chosen_config.json
make gnn-headline       # writes runs/gnn/headline/{test_results.csv, checkpoint.pt, ...}
```

Post-hoc attribution and final figures:

```bash
cp runs/gnn/headline/checkpoint.pt runs/gnn/checkpoint.pt
make gnn-explain        # GNNExplainer + attention → runs/gnn/explanations/
make gnn-figures        # ablation bar, heatmap, curves, sweep, importance summary
```

End-to-end shortcut (skips ablations and sweep; runs `train` with defaults):

```bash
make gnn
```

## 5. Determinism

All training entry points accept `--seed` (default 0) and use it for `torch.manual_seed`, `numpy.random.seed`, and the K-fold splits. Exact reproducibility additionally requires the same PyTorch / CUDA / torch_geometric versions; results within ~0.005 ROC-AUC are expected across hardware.

## 6. Layout

```
halo/
  data/                # label columns, amino-acid properties
  spec/                # feature engineering + classical models + figures
  gnn/                 # AFDB download, ESM2 embed, graph build, GAT, explain
tests/                 # pytest smoke tests for both tracks
Data/                  # DeepLoc 2.0 CSVs (input)
specData/              # engineered features for the spec track
gnnData/               # generated AFDB / ESM2 / graph caches (gitignored)
runs/                  # all training outputs land here
Makefile               # every command shown above
Report/                # final write-up
```

## 7. Common issues

- `torch_geometric` import errors: install the version that matches your installed `torch` (see the torch_geometric install docs).
- AFDB 404s: some accessions have no AlphaFold model. Misses are recorded in `gnnData/structures/_missing.txt` and skipped by `gnn-graphs`.
- Length mismatches between PDB and embedding: logged to `gnnData/graphs/_length_mismatch.txt` and skipped.
- Out of GPU memory on `gnn-embed`: drop `--batch-size` or run on CPU with `--device cpu`.
