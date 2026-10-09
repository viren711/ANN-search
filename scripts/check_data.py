"""Verify prepared datasets (after transfer): shapes, dtype, finite values, unit norms for the
PCA datasets, query count vs metadata. Exits non-zero on any problem.

Usage: python scripts/check_data.py sift1m sift100k quora nq [msmarco]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from ann.data import ds_dir, load_meta

ok = True
for name in sys.argv[1:]:
    d = ds_dir(name)
    try:
        m = load_meta(name)
        xb = np.load(d / "base.npy", mmap_mode="r")
        xq = np.load(d / "query.npy")
        assert xb.dtype == np.float32 and xq.dtype == np.float32, "dtype must be float32"
        assert xb.ndim == 2 and xb.shape[1] == 128 and xq.shape[1] == 128, f"bad shapes {xb.shape} {xq.shape}"
        assert xb.shape[0] == m["corpus_size"], f"corpus {xb.shape[0]} != meta {m['corpus_size']}"
        assert xq.shape[0] == m["num_queries"], f"queries {xq.shape[0]} != meta {m['num_queries']}"
        probe = np.asarray(xb[np.linspace(0, xb.shape[0] - 1, 2000).astype(int)])
        assert np.isfinite(probe).all() and np.isfinite(xq).all(), "non-finite values"
        if "pca_variance_explained" in m:
            nb = np.linalg.norm(probe, axis=1)
            assert np.allclose(nb, 1, atol=1e-3), "corpus vectors not L2-normalised"
        gt = "gt_ids.npy present" if (d / "gt_ids.npy").exists() else "gt_ids.npy missing (computed by gt stage)"
        print(f"[ok] {name:9s} base {xb.shape} query {xq.shape} | {gt}")
    except Exception as e:                               # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {e!r}")
sys.exit(0 if ok else 1)
