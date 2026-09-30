"""Part 2 deliverable — single-query sanity check on SIFT100K.

Reports exact top-10 IDs, Vamana top-10 IDs, 10-recall@10, vertices expanded and distance
evaluations for one query.

Usage: python scripts/sanity_query.py [--qid 0 --L-search 50]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ann.data import RESULTS, load_base, load_gt, load_query
from ann.metrics import k_recall_at_k
from ann.vamana import VamanaIndex
from build_vamana import index_dir

ap = argparse.ArgumentParser()
ap.add_argument("--qid", type=int, default=0)
ap.add_argument("--L-search", type=int, default=50)
ap.add_argument("--R", type=int, default=64)
ap.add_argument("--L", type=int, default=100)
ap.add_argument("--alpha", type=float, default=1.2)
a = ap.parse_args()

xb, xq, gt = load_base("sift100k"), load_query("sift100k"), load_gt("sift100k")
idx = VamanaIndex.load(index_dir("sift100k", a.R, a.L, a.alpha), xb)
q = xq[a.qid:a.qid + 1]
ids, _, st = idx.search(q, 10, a.L_search, return_stats=True)
res = {"query_id": a.qid, "R": a.R, "L_build": a.L, "alpha": a.alpha, "L_search": a.L_search,
       "exact_top10": gt[a.qid, :10].tolist(), "vamana_top10": ids[0].tolist(),
       "10-recall@10": k_recall_at_k(ids, gt[a.qid:a.qid + 1], 10),
       "vertices_expanded": int(st["expanded"][0]), "distance_evaluations": int(st["dist_evals"][0])}
(RESULTS / "sift100k").mkdir(parents=True, exist_ok=True)
(RESULTS / "sift100k" / "sanity_query.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
