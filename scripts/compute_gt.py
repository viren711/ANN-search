"""Step 2 — exact top-100 neighbours with FAISS IndexFlatL2 (computed once, saved, reused).

  * sift1m  : FAISS exact result is compared with the supplied ground truth; the *supplied*
              ground truth is used as gt_ids.npy (as required).
  * sift100k, quora, nq: FAISS exact search over the 128-d vectors defines gt_ids.npy.

Usage: python scripts/compute_gt.py --dataset sift1m sift100k quora nq [msmarco]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np

from ann.data import ds_dir, load_base, load_query, save_meta
from ann.metrics import k_recall_at_k

K = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["sift1m", "sift100k", "quora", "nq"])
    args = ap.parse_args()
    for name in args.dataset:
        xb, xq = load_base(name, mmap=True), load_query(name)
        t0 = time.perf_counter()
        index = faiss.IndexFlatL2(xb.shape[1])
        for s in range(0, xb.shape[0], 1 << 20):          # chunked add: avoids 2 copies in RAM
            index.add(np.ascontiguousarray(xb[s:s + (1 << 20)]))
        D, I = index.search(xq, K)
        dt = time.perf_counter() - t0
        I = I.astype(np.int32)
        np.save(ds_dir(name) / "gt_faiss_ids.npy", I)
        np.save(ds_dir(name) / "gt_faiss_dist.npy", D)
        info = {"gt_k": K, "gt_time_s": dt}
        if name == "sift1m":
            sup = np.load(ds_dir(name) / "gt_supplied.npy")
            agree = {"top1_exact_match": float((I[:, 0] == sup[:, 0]).mean()),
                     "10-recall@10_vs_supplied": k_recall_at_k(I, sup, 10),
                     "100-recall@100_vs_supplied": k_recall_at_k(I, sup, 100)}
            print(f"[sift1m] FAISS exact vs supplied GT: {agree}")
            info["faiss_vs_supplied_gt"] = agree
            np.save(ds_dir(name) / "gt_ids.npy", sup)
        else:
            np.save(ds_dir(name) / "gt_ids.npy", I)
        save_meta(name, info)
        print(f"[{name}] exact top-{K} for {len(xq)} queries over {len(xb)} vectors in {dt:.1f}s")


if __name__ == "__main__":
    main()
