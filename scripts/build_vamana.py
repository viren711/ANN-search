"""Part 2 — build a Vamana index with our own implementation and record build statistics.

Records: R, L_build, alpha, total build time, avg/max out-degree, serialized size, peak RSS.
Asserts that no vertex has out-degree > R.

Usage: python scripts/build_vamana.py --dataset sift100k [--R 64 --L 100 --alpha 1.2 --threads 12]
       [--subset 250000]   (build on the first n base vectors — for scaling measurements)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from ann.bench import RSSSampler, peak_rss_mib, set_threads, system_info
from ann.data import INDICES, load_base
from ann.vamana import VamanaIndex


def index_dir(dataset, R, L, alpha, subset=None):
    tag = f"{dataset}{'_n' + str(subset) if subset else ''}_R{R}_L{L}_a{alpha}"
    return INDICES / "vamana" / tag


def build(dataset, R, L, alpha, seed=0, threads=12, mode="batched", subset=None, save=True):
    set_threads(threads)
    xb = load_base(dataset)
    if subset:
        xb = np.ascontiguousarray(xb[:subset])
    VamanaIndex.build(xb[:2000], R=R, L=L, alpha=alpha, seed=seed, mode=mode, verbose=False)  # JIT warm-up
    rss_before = peak_rss_mib()
    with RSSSampler() as mem:
        t0 = time.perf_counter()
        idx = VamanaIndex.build(xb, R=R, L=L, alpha=alpha, seed=seed, mode=mode)
        wall = time.perf_counter() - t0
    st = idx.graph_stats()
    assert st["max_out_degree"] <= R
    stats = dict(dataset=dataset, N=int(xb.shape[0]), R=R, L_build=L, alpha=alpha, seed=seed, mode=mode,
                 build_threads=threads, build_time_s=wall, pass_times_s=idx.params["pass_times_s"],
                 peak_build_rss_mib=mem.peak_mib, rss_before_build_mib=rss_before,
                 vectors_mib=xb.nbytes / 2**20, **st, cpu=system_info(threads, "-")["cpu_model"])
    if save:
        d = index_dir(dataset, R, L, alpha, subset)
        stats["serialized_graph_bytes"] = idx.save(d)
        stats["index_bytes_with_vectors"] = stats["serialized_graph_bytes"] + xb.nbytes
        (d / "build_stats.json").write_text(json.dumps(stats, indent=2))
        stats["path"] = str(d)
    print(json.dumps(stats, indent=2))
    return idx, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--mode", default="batched", choices=["batched", "serial"])
    ap.add_argument("--subset", type=int, default=None)
    a = ap.parse_args()
    build(a.dataset, a.R, a.L, a.alpha, a.seed, a.threads, a.mode, a.subset)
