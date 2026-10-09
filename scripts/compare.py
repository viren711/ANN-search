"""Part 3b — Vamana vs FAISS HNSW vs FAISS exact on one full dataset.

Reads results/<ds>/faiss.csv and results/<ds>/vamana_R*_L*_a*.csv (produced by run_faiss.py and
search_vamana.py), draws the combined recall-latency plot and a compact table of
comparable-recall operating points (fastest setting of each method reaching each target).

Usage: python scripts/compare.py --dataset sift1m [--targets 0.9 0.95 0.99 0.995 0.999]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from ann.data import RESULTS
from ann.plots import plot_recall_latency, plot_xy

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--vamana-csv", default=None, help="default: first results/<ds>/vamana_*.csv")
ap.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.99, 0.995, 0.999])
a = ap.parse_args()

out = RESULTS / a.dataset
fa = pd.read_csv(out / "faiss.csv")
va = pd.read_csv(a.vamana_csv or sorted(out.glob("vamana_R*.csv"))[0])
exact, hnsw = fa[fa.method == "faiss-exact"], fa[fa.method == "faiss-hnsw"]
cnt = out / "hnsw_counts.csv"                     # optional, from scripts/hnsw_counts.py
if cnt.exists() and "efSearch" in hnsw.columns:
    c = pd.read_csv(cnt)[["efSearch", "dist_evals_per_query", "hops_per_query"]]
    hnsw = hnsw.drop(columns=[x for x in ("dist_evals_per_query", "hops_per_query") if x in hnsw.columns])
    hnsw = hnsw.merge(c, on="efSearch", how="left").rename(columns={"hops_per_query": "expanded_per_query"})
    plot_xy({"FAISS HNSW": hnsw, "Vamana (ours)": va}, out / "compare_recall_vs_dist_evals.png",
            f"{a.dataset}: recall vs distance evaluations / query", "dist_evals_per_query", "10-recall@10",
            logx=True)
plot_recall_latency({"FAISS HNSW": hnsw, "Vamana (ours)": va}, out / "compare_recall_latency.png",
                    f"{a.dataset}: Vamana vs HNSW", exact=exact)

rows = [exact.assign(target="reference")]
for t in a.targets:
    for df in (hnsw, va):
        ok = df[df["10-recall@10"] >= t]
        if len(ok):
            rows.append(ok.nsmallest(1, "mean_latency_ms").assign(target=t))
cols = ["target", "method", "param", "1-recall@1", "10-recall@10", "mean_latency_ms", "qps",
        "build_time_s", "index_bytes", "dist_evals_per_query", "expanded_per_query"]
tab = pd.concat(rows, ignore_index=True).reindex(columns=cols)
tab.to_csv(out / "compare_table.csv", index=False)
print(tab.to_string(index=False))
