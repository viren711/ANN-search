"""Extra credit 1 — turn a built Vamana graph into a disk-resident DiskANN-style index.

Inputs : dataset/processed/<ds>/base.npy and indices/vamana/<ds>_R*_L*_a*/ (from build_vamana.py)
Outputs: indices/disk/<ds>_R*_L*_a*_pq<M>/{index.bin, pq_codes.npy, pq_centroids.npy, meta.json}

PQ: FAISS ProductQuantizer (M sub-quantizers x 8 bits), trained on a uniform random sample.

Usage: python scripts/ec_build_disk_index.py --dataset msmarco [--R 64 --L 100 --alpha 1.2 --pq-m 32]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np

from ann.bench import set_threads
from ann.data import INDICES, load_base
from ann.diskann import SECTOR, layout, write_disk_index
from build_vamana import index_dir


def disk_dir(dataset, R, L, alpha, pq_m):
    return INDICES / "disk" / f"{dataset}_R{R}_L{L}_a{alpha}_pq{pq_m}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--L", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--pq-m", type=int, default=32, help="PQ sub-quantizers (bytes/vector)")
    ap.add_argument("--pq-train", type=int, default=250_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    out = disk_dir(a.dataset, a.R, a.L, a.alpha, a.pq_m)
    if a.skip_existing and (out / "meta.json").exists():
        print(f"[skip] {out} exists")
        sys.exit(0)
    out.mkdir(parents=True, exist_ok=True)
    set_threads(a.threads)
    gdir = index_dir(a.dataset, a.R, a.L, a.alpha)
    gmeta = json.loads((gdir / "meta.json").read_text())
    base = load_base(a.dataset, mmap=True)
    graph = np.load(gdir / "graph.npy", mmap_mode="r")
    degrees = np.load(gdir / "degrees.npy")
    N, D = base.shape

    # ---- PQ codebooks + codes
    t0 = time.perf_counter()
    rng = np.random.default_rng(a.seed)
    tr = np.sort(rng.choice(N, min(a.pq_train, N), replace=False))
    pq = faiss.ProductQuantizer(D, a.pq_m, 8)
    pq.train(np.ascontiguousarray(base[tr]))
    codes = np.empty((N, a.pq_m), np.uint8)
    for s in range(0, N, 1 << 20):
        codes[s:s + (1 << 20)] = pq.compute_codes(np.ascontiguousarray(base[s:s + (1 << 20)]))
    cents = faiss.vector_to_array(pq.centroids).reshape(a.pq_m, 256, D // a.pq_m).astype(np.float32)
    np.save(out / "pq_codes.npy", codes)
    np.save(out / "pq_centroids.npy", cents)
    t_pq = time.perf_counter() - t0
    print(f"PQ trained+encoded in {t_pq:.1f}s")

    # ---- sector-packed graph + full-precision vectors
    t0 = time.perf_counter()
    rec, nps = write_disk_index(out / "index.bin", base, graph, degrees)
    t_w = time.perf_counter() - t0
    meta = dict(dataset=a.dataset, N=N, D=D, R=a.R, L_build=a.L, alpha=a.alpha, start=gmeta["start"],
                rec_size=rec, nodes_per_sector=nps, sector_bytes=SECTOR,
                record_layout=f"[{D} x float32 vector | int32 degree | {a.R} x int32 neighbours], "
                              f"{rec} B, {nps} records per 4 KiB sector, no record crosses a sector",
                pq_m=a.pq_m, pq_nbits=8, pq_train_sample=int(len(tr)), pq_seed=a.seed,
                pq_time_s=t_pq, write_time_s=t_w,
                index_bin_bytes=(out / "index.bin").stat().st_size,
                pq_codes_bytes=(out / "pq_codes.npy").stat().st_size,
                pq_centroids_bytes=(out / "pq_centroids.npy").stat().st_size)
    meta["total_on_disk_bytes"] = meta["index_bin_bytes"] + meta["pq_codes_bytes"] + meta["pq_centroids_bytes"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
