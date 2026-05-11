"""GNN + attention + GNNExplainer track for halo-class.

Parallel to halo.spec (classical baseline). Pipeline:

    Sequence -> ESM2-150M (frozen, fp16) -> per-residue embeddings
                                    \\
    UniProt ACC -> AlphaFold PDB -> Cα contact graph
                                    /
                  -> per-protein torch_geometric Data
                  -> GAT (multi-label, 10 sigmoid heads)
                  -> GNNExplainer post-hoc attribution

See PLAN_GNN.md for the execution plan and TODOs.
"""
