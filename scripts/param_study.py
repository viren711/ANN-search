"""Part 3a — Vamana parameter study on SIFT100K.

Builds (or reuses) graphs for the reference configuration and variants in R and alpha, sweeps
L_search on each, writes one table and the two required plots:
  (i)  10-recall@10 vs L_search      (ii) 10-recall@10 vs mean latency

Usage: python scripts/param_study.py [--configs 64,100,1.2 32,100,1.2 96,100,1.2 64,100,1.0 64,100,1.4]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from ann.data import RESULTS, load_base, load_gt, load_query
from ann.plots import plot_recall_latency, plot_xy
from ann.vamana import VamanaIndex
from build_vamana import build, index_dir
from search_vamana import sweep

DS = "sift100k"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["64,100,1.2", "32,100,1.2", "96,100,1.2", "64,100,1.0", "64,100,1.4"])
    ap.add_argument("--Ls", type=int, nargs="+", default=[10, 20, 30, 50, 75, 100, 150, 200])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--build-threads", type=int, default=12)
    a = ap.parse_args()

    xb, xq, gt = load_base(DS), load_query(DS), load_gt(DS)
    (RESULTS / DS).mkdir(parents=True, exist_ok=True)
    frames, series = [], {}
    for c in a.configs:
        R, L, alpha = int(c.split(",")[0]), int(c.split(",")[1]), float(c.split(",")[2])
        d = index_dir(DS, R, L, alpha)
        build(DS, R, L, alpha, threads=a.build_threads, skip_existing=True)
        bstats = json.loads((d / "build_stats.json").read_text())
        idx = VamanaIndex.load(d, xb)
        idx.params["build_time_s"] = bstats["build_time_s"]
        df = sweep(idx, xq, gt, a.Ls, threads=a.threads,
                   perq_path=RESULTS / DS / f"param_R{R}_L{L}_a{alpha}_perquery.npz")
        df["avg_out_degree"] = bstats["avg_out_degree"]
        frames.append(df)
        series[f"R={R}, α={alpha}, L_b={L}"] = df
    out = RESULTS / DS
    out.mkdir(parents=True, exist_ok=True)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(out / "param_study.csv", index=False)
    plot_xy(series, out / "param_recall_vs_Lsearch.png", "SIFT100K: 10-recall@10 vs L_search",
            "L_search", "10-recall@10", logx=True)
    plot_recall_latency(series, out / "param_recall_vs_latency.png", "SIFT100K: 10-recall@10 vs latency")
    plot_xy(series, out / "param_recall_vs_expanded.png", "SIFT100K: recall vs vertices expanded",
            "expanded_per_query", "10-recall@10", logx=True)
    cols = ["R", "alpha", "L_build", "L_search", "1-recall@1", "10-recall@10", "mean_latency_ms",
            "qps", "dist_evals_per_query", "expanded_per_query"]
    print(table[cols].to_string(index=False))
