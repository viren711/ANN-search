"""Regenerate every figure from the saved CSVs (no experiments are re-run).

Usage: python scripts/replot.py [--root server_results/results]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from ann.data import RESULTS
from ann.plots import plot_recall_latency, plot_xy

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(RESULTS))
root = Path(ap.parse_args().root)

for d in sorted(p for p in root.iterdir() if p.is_dir()):
    ds, f = d.name, d / "faiss.csv"
    if f.exists():
        fa = pd.read_csv(f)
        exact, hnsw = fa[fa.method == "faiss-exact"], fa[fa.method == "faiss-hnsw"]
        plot_recall_latency({"FAISS HNSW": hnsw}, d / "hnsw_recall_latency.png", f"{ds}: HNSW", exact=exact)
        v = sorted(d.glob("vamana_R64_L100_a1.2.csv"))
        if v and ds != "sift100k":
            va = pd.read_csv(v[0])
            plot_recall_latency({"FAISS HNSW": hnsw, "Vamana (ours)": va}, d / "compare_recall_latency.png",
                                f"{ds}: Vamana vs HNSW", exact=exact)
            c = d / "hnsw_counts.csv"
            if c.exists():
                h = pd.read_csv(c)
                plot_xy({"FAISS HNSW": h, "Vamana (ours)": va}, d / "compare_recall_vs_dist_evals.png",
                        f"{ds}: recall vs distance evaluations / query", "dist_evals_per_query",
                        "10-recall@10", logx=True)
    ps = d / "param_study.csv"
    if ps.exists():
        t = pd.read_csv(ps)
        series = {f"R={int(R)}, α={a}, L_b={int(Lb)}": g for (R, a, Lb), g in t.groupby(["R", "alpha", "L_build"])}
        plot_xy(series, d / "param_recall_vs_Lsearch.png", "SIFT100K: 10-recall@10 vs L_search",
                "L_search", "10-recall@10", logx=True)
        plot_recall_latency(series, d / "param_recall_vs_latency.png", "SIFT100K: 10-recall@10 vs latency")
        plot_xy(series, d / "param_recall_vs_expanded.png", "SIFT100K: recall vs vertices expanded",
                "expanded_per_query", "10-recall@10", logx=True)
ec = root / "extra_credit"
for f in sorted(ec.glob("ec*.csv")) if ec.exists() else []:
    df = pd.read_csv(f)
    series = {f"{dv}-{c}": g for (dv, c), g in df.groupby(["device", "cache"])}
    plot_recall_latency(series, ec / f"{f.stem}_recall_latency.png", f"{f.stem}: disk-resident search")
print("replotted", root)
