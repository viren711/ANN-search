"""Follow-up — HNSW work counters (no timing): distance computations and graph hops per query.

Uses FAISS's global hnsw_stats (ndis, nhops) with 1 thread over the full query set, so HNSW can be
compared with Vamana on "distance evaluations per query" at equal recall (hardware-independent).

Usage: python scripts/hnsw_counts.py --dataset nq [--M 32 --efc 200 --ef 16 32 64 ...]
Output: results/<ds>/hnsw_counts.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import pandas as pd

from ann.data import INDICES, RESULTS, load_gt, load_query
from ann.metrics import recall_report

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--M", type=int, default=32)
ap.add_argument("--efc", type=int, default=200)
ap.add_argument("--ef", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
a = ap.parse_args()

faiss.omp_set_num_threads(1)
xq, gt = load_query(a.dataset), load_gt(a.dataset)
idx = faiss.read_index(str(INDICES / "faiss" / f"{a.dataset}_hnsw_M{a.M}_efc{a.efc}.index"))
st = faiss.cvar.hnsw_stats
rows = []
for ef in a.ef:
    idx.hnsw.efSearch = ef
    st.reset()
    _, I = idx.search(xq, 10)
    rows.append(dict(method="faiss-hnsw", efSearch=ef, dist_evals_per_query=st.ndis / len(xq),
                     hops_per_query=st.nhops / len(xq), **recall_report(I, gt)))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv(RESULTS / a.dataset / "hnsw_counts.csv", index=False)
print(df.to_string(index=False))
