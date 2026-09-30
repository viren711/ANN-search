"""Dataset paths and I/O helpers (fvecs/ivecs readers, processed .npy loaders)."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "raw"
PROCESSED = ROOT / "dataset" / "processed"
INDICES = ROOT / "indices"
RESULTS = ROOT / "results"

DATASETS = ["sift1m", "sift100k", "quora", "nq"]


# ---------------------------------------------------------------- TEXMEX formats
def read_fvecs(path):
    """Read a .fvecs file -> contiguous float32 array [n, d]."""
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:].view(np.float32))


def read_ivecs(path):
    """Read a .ivecs file -> contiguous int32 array [n, d]."""
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return np.ascontiguousarray(a.reshape(-1, d + 1)[:, 1:])


# ---------------------------------------------------------------- processed data
def ds_dir(name):
    return PROCESSED / name


def load_base(name, mmap=False):
    """Corpus vectors [N, 128] float32. mmap=False loads into RAM (needed for fast search)."""
    x = np.load(ds_dir(name) / "base.npy", mmap_mode="r" if mmap else None)
    return x if mmap else np.ascontiguousarray(x, dtype=np.float32)


def load_query(name):
    return np.ascontiguousarray(np.load(ds_dir(name) / "query.npy"), dtype=np.float32)


def load_gt(name):
    """Exact top-100 neighbor IDs [nq, 100] int32 (reference answers for recall)."""
    return np.load(ds_dir(name) / "gt_ids.npy")


def load_meta(name):
    p = ds_dir(name) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_meta(name, meta):
    p = ds_dir(name) / "meta.json"
    old = load_meta(name)
    old.update(meta)
    p.write_text(json.dumps(old, indent=2))


def timing_subset(nq, size=1000, seed=0):
    """Fixed random subset of query indices used for *timing only* (same for every method)."""
    if nq <= size:
        return np.arange(nq)
    return np.sort(np.random.default_rng(seed).choice(nq, size, replace=False))
