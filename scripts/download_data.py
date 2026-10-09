"""Step 0 — download raw datasets.

  * SIFT1M  : ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
              (sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs, sift_learn.fvecs)
  * Quora/NQ: CohereLabs/beir-embed-english-v3 (corpus parquet shards + test queries + test qrels)
  * MS MARCO (extra credit): same source, 8.8M passages (~18.8 GB), dev queries (6,980)

Usage:
    python scripts/download_data.py [--only sift quora nq msmarco]
"""
import argparse
import socket
import sys
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ann.data import RAW, SPLIT  # noqa: E402
SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
HF_REPO = "CohereLabs/beir-embed-english-v3"


socket.setdefaulttimeout(30)          # fail fast (instead of hanging) on machines without internet


def download_sift():
    out_dir = RAW / "sift"
    if (out_dir / "sift_base.fvecs").exists():
        print("[sift] already present")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    tgz = RAW / "sift.tar.gz"
    if not tgz.exists():
        print(f"[sift] downloading {SIFT_URL}")
        urllib.request.urlretrieve(SIFT_URL, tgz)
    print("[sift] extracting")
    with tarfile.open(tgz) as tf:
        tf.extractall(RAW)          # creates RAW/sift/
    tgz.unlink()
    print("[sift] done:", sorted(p.name for p in out_dir.iterdir()))


def download_beir(name):
    # Only the evaluation split of queries/qrels is needed (test; dev for MS MARCO).
    sp = SPLIT[name]
    patterns = [f"{name}/corpus/*.parquet", f"{name}/queries/{sp}.parquet", f"{name}/qrels/{sp}.parquet"]
    from huggingface_hub import snapshot_download        # only needed when online
    print(f"[{name}] downloading {patterns}")
    snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=patterns,
                      local_dir=RAW / "beir", max_workers=4)
    print(f"[{name}] done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["sift", "quora", "nq"])
    args = ap.parse_args()
    for d in args.only:
        download_sift() if d == "sift" else download_beir(d)
