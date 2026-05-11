"""Build per-protein torch_geometric Data objects.

For each accession that has both an AlphaFold PDB (gnnData/structures/) and
a cached ESM2 embedding (gnnData/embeddings/), construct a Data with:

    x          : float16 [L, 640]  per-residue ESM2 embedding
    pos        : float32 [L, 3]    Calpha coordinates from PDB
    edge_index : long    [2, E]    union of sequence-adjacency + contact edges
    edge_type  : long    [E]       0 = sequence, 1 = contact
    y          : float32 [10]      multi-label vector from LABEL_COLUMNS
    acc        : str               UniProt accession
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

from halo.data.labels import LABEL_COLUMNS

log = logging.getLogger(__name__)


def _read_ca_coords(pdb_path: Path) -> np.ndarray | None:
    """Return [L_pdb, 3] float32 array of Calpha coords, or None on failure."""
    try:
        from Bio.PDB import PDBParser
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("p", str(pdb_path))
        coords: list[list[float]] = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        ca = residue["CA"]
                        coords.append(list(ca.get_coord()))
            break  # only first model
        if not coords:
            return None
        return np.asarray(coords, dtype=np.float32)
    except Exception as e:
        log.debug("pdb parse failed for %s: %s", pdb_path, e)
        return None


def build_one(acc: str, label_vec: np.ndarray, pdb_dir: Path,
              emb_dir: Path, out_dir: Path, contact_threshold: float = 8.0) -> str:
    """Returns 'ok' / 'skip' / 'no_pdb' / 'no_emb' / 'mismatch' / 'err'."""
    dst = out_dir / f"{acc}.pt"
    if dst.exists():
        return "skip"
    pdb_path = pdb_dir / f"{acc}.pdb"
    emb_path = emb_dir / f"{acc}.pt"
    if not pdb_path.exists() or pdb_path.stat().st_size == 0:
        return "no_pdb"
    if not emb_path.exists():
        return "no_emb"
    coords = _read_ca_coords(pdb_path)
    if coords is None:
        return "err"
    try:
        emb_obj = torch.load(emb_path, map_location="cpu", weights_only=False)
    except Exception:
        return "err"
    x = emb_obj["x"]  # [L_emb, 640]
    # Embedding length must match Calpha count, otherwise residue indices won't align.
    if x.shape[0] != coords.shape[0]:
        return "mismatch"
    L = x.shape[0]

    # Sequence edges (i, i+1) and (i+1, i)
    if L >= 2:
        seq_src = np.concatenate([np.arange(L - 1), np.arange(1, L)])
        seq_dst = np.concatenate([np.arange(1, L), np.arange(L - 1)])
    else:
        seq_src = np.array([], dtype=np.int64)
        seq_dst = np.array([], dtype=np.int64)

    # Contact edges via KDTree on Calpha coords
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=float(contact_threshold), output_type="ndarray")
    if pairs.size == 0:
        ct_src = np.array([], dtype=np.int64)
        ct_dst = np.array([], dtype=np.int64)
    else:
        # Drop pairs already covered by |i-j| <= 1
        keep = np.abs(pairs[:, 0] - pairs[:, 1]) > 1
        pairs = pairs[keep]
        ct_src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        ct_dst = np.concatenate([pairs[:, 1], pairs[:, 0]])

    src = np.concatenate([seq_src, ct_src]).astype(np.int64)
    dst_idx = np.concatenate([seq_dst, ct_dst]).astype(np.int64)
    etype = np.concatenate([np.zeros_like(seq_src), np.ones_like(ct_src)]).astype(np.int64)

    edge_index = torch.from_numpy(np.stack([src, dst_idx], axis=0))
    edge_type = torch.from_numpy(etype)
    pos = torch.from_numpy(coords)
    y = torch.from_numpy(label_vec.astype(np.float32))

    data = {
        "x": x,
        "pos": pos,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "y": y,
        "acc": acc,
        "n_residues": L,
    }
    tmp = dst.with_suffix(".pt.tmp")
    torch.save(data, tmp)
    tmp.replace(dst)
    return "ok"


def _worker(args):
    acc, lab, pdb_dir, emb_dir, out_dir, ct = args
    try:
        return acc, build_one(acc, lab, pdb_dir, emb_dir, out_dir, ct)
    except Exception as e:
        return acc, f"err:{e!r}"


def build_all(records: list[tuple[str, np.ndarray]], pdb_dir: Path, emb_dir: Path,
              out_dir: Path, contact_threshold: float = 8.0, workers: int = 8) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "skip": 0, "no_pdb": 0, "no_emb": 0, "mismatch": [], "err": 0}
    tasks = [(a, l, pdb_dir, emb_dir, out_dir, contact_threshold) for a, l in records]
    if workers <= 1:
        results = [_worker(t) for t in tqdm(tasks, desc="graphs")]
    else:
        with Pool(workers) as pool:
            results = list(tqdm(pool.imap_unordered(_worker, tasks, chunksize=8),
                                total=len(tasks), desc="graphs"))
    for acc, status in results:
        if status in stats and isinstance(stats[status], int):
            stats[status] += 1
        elif status == "mismatch":
            stats["mismatch"].append(acc)
        else:
            stats["err"] = stats.get("err", 0) + 1
    if stats["mismatch"]:
        (out_dir / "_length_mismatch.txt").write_text("\n".join(stats["mismatch"]))
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csvs", nargs="+", required=True, type=Path)
    p.add_argument("--pdb-dir", type=Path, required=True)
    p.add_argument("--embedding-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--contact-threshold", type=float, default=8.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    seen: dict[str, np.ndarray] = {}
    for c in args.csvs:
        df = pd.read_csv(c)
        col = "ACC" if "ACC" in df.columns else "AccessionID"
        for _, row in df.iterrows():
            acc = str(row[col]).strip()
            if not acc or acc.lower() == "nan":
                continue
            lab = np.array([float(row.get(L, 0.0) or 0.0) for L in LABEL_COLUMNS],
                           dtype=np.float32)
            if acc not in seen:
                seen[acc] = lab
    records = sorted(seen.items())
    if args.limit is not None:
        records = records[: args.limit]
    log.info("building %d graphs to %s", len(records), args.out)
    stats = build_all(records, args.pdb_dir, args.embedding_dir, args.out,
                      contact_threshold=args.contact_threshold, workers=args.workers)
    log.info("done: %s", {k: (len(v) if isinstance(v, list) else v) for k, v in stats.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
