"""Step 1b — Quora / NQ: Cohere 1024-d embeddings -> PCA-128 -> float32 -> L2-normalised.

Procedure (per dataset, independently):
  1. Draw a uniform random sample (without replacement) of corpus rows, seed = --seed.
  2. Fit PCA(128) on the sampled *corpus* embeddings only (queries never used for fitting).
  3. Apply the same fitted PCA to every corpus row (streamed, 10K-row parquet row groups)
     and to the test queries.
  4. Cast to float32 and L2-normalise every projected vector.
The 1024-d corpus is never fully loaded (NQ would be ~11 GB); only the sample is in RAM.

Usage: python scripts/prepare_beir.py --dataset quora [--sample 200000 --seed 42]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA

from ann.data import RAW, ds_dir, save_meta

DIM_IN, DIM_OUT = 1024, 128


def emb_batches(files):
    """Yield (global_offset, float32 [n, 1024]) for each parquet row group."""
    off = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            col = pf.read_row_group(rg, columns=["emb"]).column("emb").combine_chunks()
            x = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            x = x.reshape(-1, DIM_IN)
            yield off, x
            off += x.shape[0]


def project(x, mean, comps):
    y = (x - mean) @ comps.T                             # float32 GEMM
    y = y.astype(np.float32, copy=False)
    y /= np.linalg.norm(y, axis=1, keepdims=True).clip(min=1e-12)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["quora", "nq", "msmarco"])
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = RAW / "beir" / args.dataset
    files = sorted((src / "corpus").glob("*.parquet"))
    N = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    out = ds_dir(args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{args.dataset}] corpus rows = {N} in {len(files)} shard(s)")

    # ---- 1. uniform random sample of corpus rows
    n_s = min(args.sample, N)
    idx = np.sort(np.random.default_rng(args.seed).choice(N, n_s, replace=False))
    S = np.empty((n_s, DIM_IN), np.float32)
    t0 = time.perf_counter()
    filled = 0
    for off, x in emb_batches(files):
        lo, hi = np.searchsorted(idx, [off, off + x.shape[0]])
        S[filled:filled + hi - lo] = x[idx[lo:hi] - off]
        filled += hi - lo
    assert filled == n_s
    print(f"  sampled {n_s} rows in {time.perf_counter() - t0:.1f}s")

    # ---- 2. PCA fit on corpus sample only
    t0 = time.perf_counter()
    pca = PCA(n_components=DIM_OUT, svd_solver="covariance_eigh", random_state=args.seed).fit(S)
    fit_s = time.perf_counter() - t0
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA fit {fit_s:.1f}s, variance explained by {DIM_OUT} comps = {evr:.4f}")
    del S
    mean = pca.mean_.astype(np.float32)
    comps = pca.components_.astype(np.float32)          # [128, 1024]
    np.savez(out / "pca.npz", mean=mean, components=comps,
             explained_variance_ratio=pca.explained_variance_ratio_, sample_idx=idx)

    # ---- 3/4. project + normalise corpus (streamed into a memmap)
    t0 = time.perf_counter()
    base = np.lib.format.open_memmap(out / "base.npy", mode="w+", dtype=np.float32, shape=(N, DIM_OUT))
    for off, x in emb_batches(files):
        base[off:off + x.shape[0]] = project(x, mean, comps)
    base.flush()
    del base
    print(f"  projected corpus in {time.perf_counter() - t0:.1f}s")

    # ---- queries (test split)
    qt = pq.read_table(src / "queries" / "test.parquet", columns=["_id", "emb"])
    q = np.asarray(qt.column("emb").combine_chunks().flatten().to_numpy(zero_copy_only=False),
                   np.float32).reshape(-1, DIM_IN)
    np.save(out / "query.npy", project(q, mean, comps))
    (out / "query_ids.txt").write_text("\n".join(qt.column("_id").to_pylist()))
    qrels_q = set(pq.read_table(src / "qrels" / "test.parquet").column(0).to_pylist())
    print(f"  queries: {q.shape[0]} (distinct query ids in test qrels: {len(qrels_q)})")

    save_meta(args.dataset, {
        "name": args.dataset, "corpus_size": int(N), "num_queries": int(q.shape[0]),
        "original_dim": DIM_IN, "search_dim": DIM_OUT, "metric": "L2 on L2-normalised vectors",
        "source": f"CohereLabs/beir-embed-english-v3 ({args.dataset}, test queries)",
        "pca_impl": f"sklearn.decomposition.PCA(n_components=128, svd_solver='covariance_eigh')",
        "pca_fit_sample_size": int(n_s), "pca_seed": args.seed,
        "pca_variance_explained": evr, "pca_fit_time_s": fit_s,
    })
    print(json.dumps({"variance_explained": evr, "N": N, "nq": int(q.shape[0])}))


if __name__ == "__main__":
    main()
