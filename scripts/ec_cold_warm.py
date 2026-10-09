"""Extra credit 1 & 2 — cold- and warm-cache disk-search experiments.

For every device copy and every L_search:
  cold : (index files closed) fincore -> fadvise(DONTNEED) per file -> fincore -> start a new
         search process and run the timed workload.
  warm : run the workload once to exercise the index (untimed), fincore, then a new process
         runs the timed workload.

EC1 (single device):   --copies ssd=indices/disk/msmarco_R64_L100_a1.2_pq32
EC2 (HDD vs SSD):      --copies hdd=/mnt/hdd/ann_idx ssd=/mnt/ssd/ann_idx --src <built index dir>
                       (the identical index is copied to each device if not already there)

Usage: python scripts/ec_cold_warm.py --dataset msmarco --src DIR --copies ssd=DIR [hdd=DIR]
       [--Ls 50 100 200 --W 4 --num-queries 1000 --out-name ec1]
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from ann.data import RESULTS
from ann.pagecache import device_info, evict, residency
from ann.plots import plot_recall_latency

FILES = ["index.bin", "pq_codes.npy", "pq_centroids.npy", "meta.json"]
HERE = Path(__file__).resolve().parent


def run_worker(d, a, L, nq, label):
    cmd = [sys.executable, str(HERE / "ec_disk_search.py"), "--index-dir", str(d), "--dataset", a.dataset,
           "--L", str(L), "--W", str(a.W), "--io", a.io, "--num-queries", str(nq),
           "--ram-limit-gib", str(a.ram_limit_gib), "--label", label]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"worker failed ({label}):\n{p.stderr[-3000:]}")
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def total_res(r):
    return {"resident_bytes": sum(v["resident_bytes"] for v in r.values()),
            "resident_frac_index_bin": next(v["resident_frac"] for k, v in r.items() if k.endswith("index.bin")),
            "tool": next(iter(r.values()))["tool"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--src", default=None, help="built index dir to copy to each device (EC2)")
    ap.add_argument("--copies", nargs="+", required=True, help="label=dir pairs")
    ap.add_argument("--Ls", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--io", default="fadvise", choices=["threads", "fadvise", "serial"])
    ap.add_argument("--num-queries", type=int, default=1000)
    ap.add_argument("--num-queries-hdd", type=int, default=200, help="smaller workload for rotational disks")
    ap.add_argument("--ram-limit-gib", type=float, default=8.0)
    ap.add_argument("--out-name", default="ec1")
    a = ap.parse_args()

    out = RESULTS / "extra_credit"
    out.mkdir(parents=True, exist_ok=True)
    rows, raw = [], {"args": vars(a), "devices": {}, "runs": []}
    for spec in a.copies:
        label, d = spec.split("=", 1)
        d = Path(d)
        if a.src and Path(a.src).resolve() != d.resolve():
            d.mkdir(parents=True, exist_ok=True)
            for f in FILES:
                if not (d / f).exists():
                    print(f"copying {f} -> {d}", flush=True)
                    shutil.copy2(Path(a.src) / f, d / f)
        files = [d / f for f in FILES[:3]]
        dev = device_info(d)
        raw["devices"][label] = dev
        nq = a.num_queries_hdd if dev.get("rota") == "1" else a.num_queries
        print(f"[{label}] {dev}", flush=True)
        for L in a.Ls:
            # ---------------- cold
            before = residency(files)
            evict(files)
            after = residency(files)
            cold = run_worker(d, a, L, nq, f"{label}-cold")
            cold.update(device=label, cache="cold",
                        fincore_before_evict=total_res(before), fincore_before_run=total_res(after))
            # ---------------- warm
            run_worker(d, a, L, nq, f"{label}-exercise")
            warm_res = residency(files)
            warm = run_worker(d, a, L, nq, f"{label}-warm")
            warm.update(device=label, cache="warm", fincore_before_run=total_res(warm_res))
            for r in (cold, warm):
                raw["runs"].append(r)
                rows.append({k: v for k, v in r.items() if not isinstance(v, dict)}
                            | {"resident_frac_before_run": r["fincore_before_run"]["resident_frac_index_bin"],
                               "resident_bytes_before_run": r["fincore_before_run"]["resident_bytes"]})
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{a.out_name}.csv", index=False)
    (out / f"{a.out_name}.json").write_text(json.dumps(raw, indent=2))
    series = {f"{dv}-{c}": g for (dv, c), g in df.groupby(["device", "cache"])}
    plot_recall_latency(series, out / f"{a.out_name}_recall_latency.png", f"{a.dataset}: disk-resident search")
    cols = ["device", "cache", "L_search", "1-recall@1", "10-recall@10", "mean_latency_ms", "p50_latency_ms",
            "p95_latency_ms", "qps", "reads_per_query", "bytes_per_query", "rounds_per_query",
            "expansions_per_query", "storage_read_bytes_per_query", "peak_rss_mib", "resident_frac_before_run"]
    print(df[cols].to_string(index=False))
