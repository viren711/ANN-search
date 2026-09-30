"""Step 1a — convert SIFT1M (fvecs/ivecs) to .npy and create the SIFT100K development set.

SIFT1M  : 1,000,000 base x 128, 10,000 queries, supplied exact top-100 ground truth (L2).
SIFT100K: first 100,000 SIFT1M base vectors + the same 10,000 queries; its ground truth is
          computed with FAISS exact search in scripts/compute_gt.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from ann.data import RAW, ds_dir, read_fvecs, read_ivecs, save_meta

SUB = 100_000

base = read_fvecs(RAW / "sift" / "sift_base.fvecs")
query = read_fvecs(RAW / "sift" / "sift_query.fvecs")
gt = read_ivecs(RAW / "sift" / "sift_groundtruth.ivecs")
print("SIFT1M base", base.shape, "query", query.shape, "gt", gt.shape)

for name, b in [("sift1m", base), ("sift100k", base[:SUB])]:
    d = ds_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "base.npy", np.ascontiguousarray(b, dtype=np.float32))
    np.save(d / "query.npy", query)
    save_meta(name, {"name": name, "corpus_size": int(b.shape[0]), "num_queries": int(query.shape[0]),
                     "original_dim": 128, "search_dim": 128, "metric": "L2",
                     "source": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
                               + ("" if name == "sift1m" else f" (first {SUB} base vectors)")})
np.save(ds_dir("sift1m") / "gt_supplied.npy", gt)
print("saved to", ds_dir("sift1m"), "and", ds_dir("sift100k"))
