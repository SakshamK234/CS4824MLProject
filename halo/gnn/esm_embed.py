"""Cache per-residue ESM2-150M embeddings, frozen and fp16, to disk.

For each protein in the DeepLoc CSVs, run the protein's sequence through
ESM2-150M (esm2_t30_150M_UR50D), collect the last-layer per-residue hidden
states, downcast to fp16, and write one tensor per accession to
gnnData/embeddings/<ACC>.pt.

Sliding-window inference for sequences longer than 512 tokens:
- window 512, stride 448 (overlap 64)
- per-residue embeddings averaged across windows that cover that residue
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)

ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
HIDDEN_DIM = 640


def _clean_seq(seq: str) -> str:
    """Restrict to the 20 canonical amino acids; replace everything else with X.

    ESM2 tokenizer has its own X/UNK token so non-canonical chars are tolerated.
    """
    if not isinstance(seq, str):
        return ""
    s = seq.strip().upper()
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(c if c in allowed else "X" for c in s)


def embed_one(seq: str, model, tokenizer, device: torch.device,
              window: int = 512, stride: int = 448) -> torch.Tensor:
    """Return per-residue embedding tensor of shape [L, 640] in fp16 on CPU.

    Sliding window with averaging in overlap region.
    """
    L = len(seq)
    if L == 0:
        return torch.zeros(0, HIDDEN_DIM, dtype=torch.float16)

    out = torch.zeros(L, HIDDEN_DIM, dtype=torch.float32)
    counts = torch.zeros(L, dtype=torch.float32)

    starts = list(range(0, max(1, L), stride))
    # Ensure the final window covers the C-terminus even if stride didn't reach it.
    if starts[-1] + window < L:
        starts.append(max(0, L - window))
    starts = sorted(set(starts))

    for start in starts:
        end = min(start + window, L)
        sub = seq[start:end]
        toks = tokenizer(sub, return_tensors="pt", add_special_tokens=True)
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                hs = model(**toks).last_hidden_state  # [1, T, 640]
        # ESM2 prepends <cls> and appends <eos>; strip both.
        hs = hs[0, 1 : 1 + (end - start), :].float().cpu()
        out[start:end] += hs
        counts[start:end] += 1.0
        if end == L:
            break

    counts = counts.clamp_min(1.0)
    out = out / counts.unsqueeze(-1)
    return out.to(torch.float16)


def _load_model(device: torch.device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = AutoModel.from_pretrained(ESM_MODEL_NAME)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def embed_all(records: list[tuple[str, str]], out_dir: Path, device: torch.device,
              window: int = 512, stride: int = 448) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = _load_model(device)
    stats = {"ok": 0, "skip": 0, "err": 0}
    for acc, seq in tqdm(records, desc="esm2"):
        dst = out_dir / f"{acc}.pt"
        if dst.exists():
            try:
                obj = torch.load(dst, map_location="cpu", weights_only=False)
                if obj.get("x") is not None and obj["x"].shape[0] == len(seq):
                    stats["skip"] += 1
                    continue
            except Exception:
                pass
        try:
            x = embed_one(seq, model, tok, device, window=window, stride=stride)
        except Exception as e:
            log.warning("embed failed for %s: %s", acc, e)
            stats["err"] += 1
            continue
        tmp = dst.with_suffix(".pt.tmp")
        torch.save({"acc": acc, "x": x, "seq": seq}, tmp)
        tmp.replace(dst)
        stats["ok"] += 1
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csvs", nargs="+", required=True, type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)  # accepted for CLI compat
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--stride", type=int, default=448)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.info("device: %s", device)

    seen: dict[str, str] = {}
    for c in args.csvs:
        df = pd.read_csv(c)
        col = "ACC" if "ACC" in df.columns else "AccessionID"
        for _, row in df.iterrows():
            acc = str(row[col]).strip()
            seq = _clean_seq(row.get("Sequence", ""))
            if not acc or acc.lower() == "nan" or not seq:
                continue
            if acc not in seen:
                seen[acc] = seq
    records = sorted(seen.items())
    if args.limit is not None:
        records = records[: args.limit]
    log.info("embedding %d unique accessions to %s", len(records), args.out)
    stats = embed_all(records, args.out, device, window=args.window, stride=args.stride)
    log.info("done: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
