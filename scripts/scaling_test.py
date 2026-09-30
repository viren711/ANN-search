"""Part 2 development procedure — build time on growing subsets (100K / 250K / 500K ...).

Fits build_time ~ c * N^b on the measurements and extrapolates to the full corpus so we
know whether a full build is feasible before launching it.

Usage: python scripts/scaling_test.py --dataset sift1m --sizes 100000 250000 500000
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from ann.data import RESULTS, load_meta
from build_vamana import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift1m")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000, 250_000, 500_000])
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()

    rows = []
    for n in a.sizes:
        _, st = build(a.dataset, a.R, a.L, a.alpha, threads=a.threads, subset=n, save=False)
        rows.append({"N": n, "build_time_s": st["build_time_s"], "peak_rss_mib": st["peak_build_rss_mib"]})
    N = np.array([r["N"] for r in rows], float)
    T = np.array([r["build_time_s"] for r in rows], float)
    b, logc = np.polyfit(np.log(N), np.log(T), 1)
    full = load_meta(a.dataset)["corpus_size"]
    est = float(np.exp(logc) * full ** b)
    res = {"dataset": a.dataset, "R": a.R, "L_build": a.L, "alpha": a.alpha, "threads": a.threads,
           "measurements": rows, "fit_exponent": float(b), "full_N": full, "estimated_full_build_s": est}
    out = RESULTS / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / "scaling.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
