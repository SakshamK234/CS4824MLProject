"""Bulk-download AlphaFold predicted structures by UniProt accession.

Reads ACC columns from the DeepLoc CSVs in Data/, dedupes, and pulls each
protein's predicted structure from the AlphaFold Database static URL pattern:

    https://alphafold.ebi.ac.uk/files/AF-{ACC}-F1-model_v4.pdb

Outputs one PDB per accession into gnnData/structures/. Idempotent and
resumable: existing non-empty files are skipped. Misses are logged to
gnnData/structures/_missing.txt for downstream filtering.
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

AFDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb"
AFDB_FALLBACK_URLS = [
    "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v5.pdb",
    "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb",
]


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def download_one(acc: str, out_dir: Path, session: requests.Session) -> str:
    """Returns one of: 'ok', 'skip', 'miss', 'err'."""
    dst = out_dir / f"{acc}.pdb"
    if dst.exists() and dst.stat().st_size > 0:
        return "skip"
    # Try the newest AFDB model version first, then fall back to older versions.
    urls = [AFDB_URL.format(acc=acc)] + [u.format(acc=acc) for u in AFDB_FALLBACK_URLS]
    r = None
    last_status = None
    for url in urls:
        try:
            r = session.get(url, timeout=30)
        except Exception as e:
            log.debug("network error for %s: %s", acc, e)
            continue
        last_status = r.status_code
        if r.status_code == 200 and r.content:
            break
    if r is None:
        return "err"
    if last_status == 404:
        return "miss"
    if last_status != 200 or not r.content:
        return "err"
    tmp = dst.with_suffix(".pdb.tmp")
    tmp.write_bytes(r.content)
    tmp.replace(dst)
    return "ok"


def download_all(accessions: list[str], out_dir: Path, workers: int = 16) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _make_session()
    results = {"ok": 0, "skip": 0, "miss": [], "err": []}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(download_one, a, out_dir, session): a for a in accessions}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="afdb"):
            acc = futs[fut]
            try:
                status = fut.result()
            except Exception as e:
                log.warning("worker error for %s: %s", acc, e)
                status = "err"
            if status == "ok":
                results["ok"] += 1
            elif status == "skip":
                results["skip"] += 1
            elif status == "miss":
                results["miss"].append(acc)
            else:
                results["err"].append(acc)
    (out_dir / "_missing.txt").write_text("\n".join(sorted(results["miss"])))
    if results["err"]:
        (out_dir / "_errors.txt").write_text("\n".join(sorted(results["err"])))
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csvs", nargs="+", required=True, type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap for smoke tests.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    accs: set[str] = set()
    for c in args.csvs:
        df = pd.read_csv(c)
        col = "ACC" if "ACC" in df.columns else "AccessionID"
        accs.update(df[col].astype(str).str.strip().tolist())
    accs = sorted(a for a in accs if a and a.lower() != "nan")
    if args.limit is not None:
        accs = accs[: args.limit]
    log.info("downloading %d unique accessions to %s", len(accs), args.out)
    res = download_all(accs, args.out, workers=args.workers)
    n_total = len(accs)
    n_have = res["ok"] + res["skip"]
    log.info("done: ok=%d skip=%d miss=%d err=%d coverage=%.1f%%",
             res["ok"], res["skip"], len(res["miss"]), len(res["err"]),
             100.0 * n_have / max(1, n_total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
