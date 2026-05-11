PY := python -m

# Build the engineered per-protein feature CSVs from the raw DeepLoc CSVs.
# specData/ ships with these pre-built; this target only re-creates them.
spec-features:
	for s in train validation test; do \
	  $(PY) halo.spec.feature_engineering \
	    --csv Data/deeploc_$$s.csv \
	    --out specData/sequence_features_$$s.csv ; \
	done

# Run the 4-model panel × 10 compartments with K-fold-CV hyperparameter tuning.
# Outputs: runs/spec_classical/{results.csv, summary_by_model.csv,
# best_params.json, curves.npz}.
spec-classical:
	$(PY) halo.spec.classical_models \
	  --train specData/sequence_features_train.csv \
	  --val   specData/sequence_features_validation.csv \
	  --test  specData/sequence_features_test.csv \
	  --out   runs/spec_classical \
	  --models logreg svm_rbf rf mlp \
	  --cv-folds 5 --k-features 30

# Bar charts, heatmap, ROC + PR overlays.
spec-figures:
	$(PY) halo.spec.figures \
	  --run-dir runs/spec_classical \
	  --out     runs/spec_classical/figures

spec: spec-features spec-classical spec-figures

# K-features sweep over SelectKBest top-K. Pass MODELS="logreg rf" to override.
MODELS ?= logreg rf
K_VALUES ?= 10 20 30
spec-kfeatures-sweep:
	$(PY) halo.spec.kfeatures_sweep \
	  --train specData/sequence_features_train.csv \
	  --val   specData/sequence_features_validation.csv \
	  --test  specData/sequence_features_test.csv \
	  --out   runs/spec_kfeatures_sweep \
	  --models $(MODELS) \
	  --k-values $(K_VALUES) \
	  --cv-folds 5

test:
	pytest -q tests

# ---------- GNN track (see PLAN_GNN.md) -----------------------------------
# Each target points at a stub module that will raise NotImplementedError
# until the corresponding stage is implemented.

gnn-download:
	$(PY) halo.gnn.afdb_download \
	  --csvs Data/deeploc_train.csv Data/deeploc_validation.csv Data/deeploc_test.csv \
	  --out  gnnData/structures \
	  --workers 16

gnn-embed:
	$(PY) halo.gnn.esm_embed \
	  --csvs Data/deeploc_train.csv Data/deeploc_validation.csv Data/deeploc_test.csv \
	  --out  gnnData/embeddings \
	  --batch-size 8 --window 512 --stride 448 \
	  --device cuda --dtype float16

gnn-graphs:
	$(PY) halo.gnn.build_graphs \
	  --csvs Data/deeploc_train.csv Data/deeploc_validation.csv Data/deeploc_test.csv \
	  --pdb-dir       gnnData/structures \
	  --embedding-dir gnnData/embeddings \
	  --out           gnnData/graphs \
	  --contact-threshold 8.0 \
	  --workers 8

gnn-train:
	$(PY) halo.gnn.train \
	  --graph-dir   gnnData/graphs \
	  --csvs-train  Data/deeploc_train.csv \
	  --csvs-val    Data/deeploc_validation.csv \
	  --csvs-test   Data/deeploc_test.csv \
	  --out         runs/gnn \
	  --epochs 20 --batch-size 16 --lr 1e-3 --weight-decay 1e-4 \
	  --device cuda --amp

# ---- Stage 5 ablation rows ------------------------------------------------
gnn-ablate-mlp-pool:
	$(PY) halo.gnn.train \
	  --graph-dir gnnData/graphs --embedding-dir gnnData/embeddings \
	  --out runs/gnn/ablation/mlp_pool \
	  --model mlp_pool --edges both --epochs 20 --batch-size 32 \
	  --lr 1e-3 --device cuda --amp --seed 0

gnn-ablate-gat-seq:
	$(PY) halo.gnn.train \
	  --graph-dir gnnData/graphs --embedding-dir gnnData/embeddings \
	  --out runs/gnn/ablation/gat_seq \
	  --model gat --edges sequence --epochs 20 --batch-size 16 \
	  --lr 1e-3 --device cuda --amp --seed 0

gnn-ablate-gat-contact:
	$(PY) halo.gnn.train \
	  --graph-dir gnnData/graphs --embedding-dir gnnData/embeddings \
	  --out runs/gnn/ablation/gat_contact \
	  --model gat --edges contact --epochs 20 --batch-size 16 \
	  --lr 1e-3 --device cuda --amp --seed 0

gnn-sweep:
	$(PY) halo.gnn.sweep \
	  --graph-dir gnnData/graphs --embedding-dir gnnData/embeddings \
	  --out runs/gnn/sweep \
	  --epochs 12 --batch-size 16 --device cuda --amp --seed 0

gnn-headline:
	$(PY) halo.gnn.train \
	  --graph-dir gnnData/graphs --embedding-dir gnnData/embeddings \
	  --out runs/gnn/headline \
	  --model gat --edges both --epochs 30 --batch-size 16 \
	  --config-json runs/gnn/sweep/chosen_config.json \
	  --train-plus-val --device cuda --amp --seed 0

gnn-ablate: gnn-ablate-mlp-pool gnn-ablate-gat-seq gnn-ablate-gat-contact

gnn-explain:
	$(PY) halo.gnn.explain \
	  --checkpoint runs/gnn/checkpoint.pt \
	  --graph-dir  gnnData/graphs \
	  --csv-test   Data/deeploc_test.csv \
	  --out        runs/gnn/explanations \
	  --proteins-per-class 50 \
	  --device cuda

gnn-figures:
	$(PY) halo.gnn.figures \
	  --classical-dir runs/spec_classical \
	  --gnn-dir       runs/gnn \
	  --out           runs/gnn/figures

gnn: gnn-download gnn-embed gnn-graphs gnn-train gnn-explain gnn-figures
