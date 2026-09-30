"""Part 2/3 — sweep L_search on a built Vamana graph.

Per operating point: 1-recall@1, 10-recall@10, mean latency, QPS, distance evals/query,
vertices expanded/query. Same timing protocol as run_faiss.py (threads, mode, subset).

Usage: python scripts/search_vamana.py --dataset sift1m --R 64 --L 100 --alpha 1.2
       [--Ls 10 20 50 100 200 --threads 1]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from ann.bench import set_threads, system_info, time_queries
from ann.data import RESULTS, load_base, load_gt, load_query, timing_subset
from ann.metrics import recall_report
from ann.vamana import VamanaIndex
from build_vamana import index_dir

DEFAULT_LS = [10, 20, 30, 50, 75, 100, 150, 200]


def sweep(idx, xq, gt, Ls, threads=1, mode="batch", timing_queries=1000, repeats=5, k=10):
    set_threads(threads)
    xq_t = xq[timing_subset(len(xq), timing_queries)]
    p, rows = idx.params, []
    for L in Ls:
        ids, _, st = idx.search(xq, k, L, return_stats=True)
        row = dict(method="vamana", R=p["R"], alpha=p["alpha"], L_build=p["L_build"], L_search=L,
                   param=f"R={p['R']},a={p['alpha']},Lb={p['L_build']},Ls={L}",
                   build_time_s=p.get("build_time_s"),
                   index_bytes=idx.graph.nbytes + idx.degrees.nbytes + idx.data.nbytes,
                   dist_evals_per_query=float(st["dist_evals"].mean()),
                   expanded_per_query=float(st["expanded"].mean()), **recall_report(ids, gt))
        row.update(time_queries(lambda Q: idx.search(Q, k, L), xq_t, repeats=repeats, mode=mode))
        rows.append(row)
        print({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in row.items()}, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--Ls", type=int, nargs="+", default=DEFAULT_LS)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mode", default="batch", choices=["batch", "single"])
    ap.add_argument("--timing-queries", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    xb, xq, gt = load_base(a.dataset), load_query(a.dataset), load_gt(a.dataset)
    d = index_dir(a.dataset, a.R, a.L, a.alpha)
    idx = VamanaIndex.load(d, xb)
    idx.params["build_time_s"] = json.loads((d / "build_stats.json").read_text())["build_time_s"]
    df = sweep(idx, xq, gt, a.Ls, a.threads, a.mode, a.timing_queries, a.repeats)
    out = RESULTS / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"vamana_R{a.R}_L{a.L}_a{a.alpha}.csv", index=False)
    (out / "vamana_system.json").write_text(json.dumps(system_info(a.threads, a.mode), indent=2))
    print(df.to_string())
