"""Extra credit — one timed disk-resident search workload (run as a separate process).

The RAM limit is imposed with setrlimit(RLIMIT_AS) *before* anything is loaded, so the process
cannot hold the full graph + vectors. Per-query latencies are measured individually
(queries issued one at a time, 1 search thread + W I/O threads).

Prints one JSON line with recall, latency (mean/p50/p95), QPS, reads/query, bytes/query,
rounds/query, expansions/query, kernel-reported storage bytes (/proc/self/io), peak RSS.

Usage: python scripts/ec_disk_search.py --index-dir DIR --dataset msmarco --L 50 --W 4
"""
import argparse
import resource
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--index-dir", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--L", type=int, default=50)
ap.add_argument("--W", type=int, default=4)
ap.add_argument("--io", default="fadvise", choices=["threads", "fadvise", "serial"])
ap.add_argument("--num-queries", type=int, default=1000)
ap.add_argument("--ram-limit-gib", type=float, default=8.0, help="0 = no limit")
ap.add_argument("--label", default="")
a = ap.parse_args()

if a.ram_limit_gib > 0:
    lim = int(a.ram_limit_gib * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from ann.data import load_gt, load_query, timing_subset  # noqa: E402
from ann.diskann import DiskIndex  # noqa: E402
from ann.metrics import recall_report  # noqa: E402


def proc_read_bytes():
    try:
        for line in open("/proc/self/io"):
            if line.startswith("read_bytes"):
                return int(line.split()[1])
    except OSError:
        pass
    return -1


sel = timing_subset(len(load_query(a.dataset)), a.num_queries)
xq = load_query(a.dataset)[sel]
gt = load_gt(a.dataset)[sel]
idx = DiskIndex(a.index_dir, W=a.W, io=a.io)
idx.warmup_jit()
idx.reset_counters()

ids = np.empty((len(xq), 10), np.int32)
lat = np.empty(len(xq))
rb0 = proc_read_bytes()
t_all = time.perf_counter()
for i in range(len(xq)):
    t0 = time.perf_counter()
    ids[i] = idx.search(xq[i], 10, a.L)
    lat[i] = time.perf_counter() - t0
wall = time.perf_counter() - t_all
rb1 = proc_read_bytes()
nq = len(xq)
c = idx.c
res = dict(label=a.label, L_search=a.L, W=a.W, io=a.io, num_queries=nq, **recall_report(ids, gt),
           mean_latency_ms=1e3 * lat.mean(), p50_latency_ms=1e3 * np.percentile(lat, 50),
           p95_latency_ms=1e3 * np.percentile(lat, 95), p99_latency_ms=1e3 * np.percentile(lat, 99),
           qps=nq / wall, reads_per_query=c["reads"] / nq, bytes_per_query=c["bytes"] / nq,
           rounds_per_query=c["rounds"] / nq, expansions_per_query=c["expanded"] / nq,
           pq_dists_per_query=c["pq_dists"] / nq,
           storage_read_bytes_per_query=(rb1 - rb0) / nq if rb0 >= 0 else None,
           peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           resident_index_ram_mib=idx.ram_bytes() / 2**20, ram_limit_gib=a.ram_limit_gib,
           ram_limit_mechanism="setrlimit(RLIMIT_AS)" if a.ram_limit_gib > 0 else "none")
idx.close()
print("RESULT " + json.dumps(res), flush=True)
