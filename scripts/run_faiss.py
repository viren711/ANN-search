"""Part 1 — FAISS exact search (IndexFlatL2) and HNSW (IndexHNSWFlat) baselines.

Recall is computed over the full query set; timing uses a fixed random subset
(--timing-queries, same subset for every method) after a warm-up, median of --repeats runs.
Indices are persisted under indices/faiss/ and reused.

Usage: python scripts/run_faiss.py --dataset sift1m [--M 32 --efc 200 --ef 16 32 64 128 256 512]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
import pandas as pd

from ann.bench import set_threads, system_info, time_queries
from ann.data import INDICES, RESULTS, load_base, load_gt, load_query, timing_subset
from ann.metrics import recall_report
from ann.plots import plot_recall_latency


def evaluate(index, xq, gt, xq_t, args, k=10):
    set_threads(args.threads)
    _, I = index.search(xq, k)
    row = recall_report(I, gt)
    row.update(time_queries(lambda Q: index.search(Q, k), xq_t, repeats=args.repeats, mode=args.mode))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--ef", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--threads", type=int, default=1, help="search threads (same for all methods)")
    ap.add_argument("--build-threads", type=int, default=12)
    ap.add_argument("--mode", default="batch", choices=["batch", "single"])
    ap.add_argument("--timing-queries", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--skip-exact", action="store_true")
    args = ap.parse_args()

    xb, xq, gt = load_base(args.dataset), load_query(args.dataset), load_gt(args.dataset)
    xq_t = xq[timing_subset(len(xq), args.timing_queries)]
    out_dir = RESULTS / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (INDICES / "faiss").mkdir(parents=True, exist_ok=True)
    rows = []

    # ---- 1a exact
    if not args.skip_exact:
        t0 = time.perf_counter()
        flat = faiss.IndexFlatL2(xb.shape[1])
        flat.add(xb)
        build = time.perf_counter() - t0
        size = faiss.serialize_index(flat).nbytes
        r = evaluate(flat, xq, gt, xq_t, args)
        # Exact search amortises the corpus scan over a query batch (FAISS switches to a BLAS GEMM
        # path for large batches), so batch latency understates per-query cost. Also record the
        # latency when queries are issued one at a time (1 repeat; reference only).
        single = time_queries(lambda Q: flat.search(Q, 10), xq_t[:min(len(xq_t), 300)],
                              warmup=10, repeats=1, mode="single")
        r["latency_ms_single_query_mode"] = single["mean_latency_ms"]
        rows.append(dict(method="faiss-exact", param="-", build_time_s=build, index_bytes=size, **r))
        print(rows[-1], flush=True)
        del flat

    # ---- 1b HNSW
    path = INDICES / "faiss" / f"{args.dataset}_hnsw_M{args.M}_efc{args.efc}.index"
    if path.exists():
        hnsw = faiss.read_index(str(path))
        build = json.loads(path.with_suffix(".json").read_text())["build_time_s"]
    else:
        set_threads(args.build_threads)
        hnsw = faiss.IndexHNSWFlat(xb.shape[1], args.M)
        hnsw.hnsw.efConstruction = args.efc
        t0 = time.perf_counter()
        hnsw.add(xb)
        build = time.perf_counter() - t0
        faiss.write_index(hnsw, str(path))
        path.with_suffix(".json").write_text(json.dumps({"build_time_s": build, "build_threads": args.build_threads}))
    size = path.stat().st_size
    for ef in args.ef:
        hnsw.hnsw.efSearch = ef
        r = evaluate(hnsw, xq, gt, xq_t, args)
        rows.append(dict(method="faiss-hnsw", param=f"M={args.M},efC={args.efc},efS={ef}", efSearch=ef,
                         build_time_s=build, index_bytes=size, **r))
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    old = out_dir / "faiss.csv"
    if old.exists() and args.skip_exact:                 # keep earlier exact row
        prev = pd.read_csv(old)
        df = pd.concat([prev[prev.method == "faiss-exact"], df], ignore_index=True)
    df.to_csv(old, index=False)
    (out_dir / "faiss_system.json").write_text(json.dumps(
        system_info(args.threads, args.mode) | {"build_threads": args.build_threads}, indent=2))
    plot_recall_latency({"FAISS HNSW": df[df.method == "faiss-hnsw"]},
                        out_dir / "hnsw_recall_latency.png", f"{args.dataset}: HNSW",
                        exact=df[df.method == "faiss-exact"])
    print(df.to_string())


if __name__ == "__main__":
    main()
