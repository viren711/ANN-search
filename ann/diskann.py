"""Disk-resident DiskANN-style search (extra credit 1) — own implementation.

On-disk layout (index.bin), DiskANN-style sector packing:
  record(v) = [ D x float32 full-precision vector | int32 degree | R x int32 neighbour ids ]
  rec_size  = 4*D + 4 + 4*R            (D=128, R=64 -> 772 B)
  nodes_per_sector = 4096 // rec_size   (-> 5); a record never crosses a 4 KiB sector boundary
  offset(v) = (v // nps) * 4096 + (v % nps) * rec_size     -> direct lookup by vertex id

Permanently in RAM: PQ codes (N x M uint8, FAISS-trained codebooks), PQ centroids, entry
point id, a per-search visited-tag array (int32 N) and small per-query buffers. The graph and
the full-precision vectors are only read on demand from index.bin with pread().

Search (beam search, as in DiskANN): candidate list of size L ordered by PQ (asymmetric)
distance; each round takes the W closest unexpanded candidates, fetches their sectors together
(W reads in flight), computes the exact distance of every fetched vertex from its on-disk
full-precision vector, and pushes unseen neighbours into the list using PQ distances.
Final top-k = the expanded vertices ranked by exact distance (full-precision re-ranking).

I/O modes for the W sector reads of one round:
  fadvise (default): posix_fadvise(WILLNEED) on all W sectors first (the kernel submits them
                     asynchronously, so they are in flight together), then pread() each one.
                     Lowest overhead in our tests (SIFT100K, NVMe: warm 0.77 ms, cold 1.48 ms).
  threads          : W preads on a thread pool (higher per-round overhead: 1.93 / 2.44 ms).
  serial           : plain sequential preads, no overlap (0.72 / 2.02 ms).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

from ann.vamana import _insert, _l2sq

SECTOR = 4096


def layout(D, R):
    rec = 4 * D + 4 + 4 * R
    nps = SECTOR // rec
    assert nps >= 1, "record larger than a sector"
    return rec, nps


# ============================================================================ build
def write_disk_index(path, base, graph, degrees, chunk_sectors=1 << 16):
    """Stream vectors + adjacency into the sector-packed file. Inputs may be np.memmap."""
    N, D = base.shape
    R = graph.shape[1]
    rec, nps = layout(D, R)
    chunk = chunk_sectors * nps
    with open(path, "wb") as f:
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            n = e - s
            n_pad = -(-n // nps) * nps
            buf = np.zeros((n_pad, rec), np.uint8)
            buf[:n, :4 * D] = np.ascontiguousarray(base[s:e], dtype=np.float32).view(np.uint8).reshape(n, 4 * D)
            buf[:n, 4 * D:4 * D + 4] = np.ascontiguousarray(degrees[s:e], dtype="<i4").view(np.uint8).reshape(n, 4)
            buf[:n, 4 * D + 4:] = np.ascontiguousarray(graph[s:e], dtype="<i4").view(np.uint8).reshape(n, 4 * R)
            sec = np.zeros((n_pad // nps, SECTOR), np.uint8)
            sec[:, :nps * rec] = buf.reshape(-1, nps * rec)
            f.write(sec.tobytes())
        f.flush()
        os.fsync(f.fileno())
    return rec, nps


# ============================================================================ kernels
@njit(cache=True)
def _pq_dist_one(codes, table, v):
    s = np.float32(0.0)
    c = codes[v]
    for m in range(table.shape[0]):
        s += table[m, c[m]]
    return s


@njit(cache=True)
def _pick_frontier(cand_ids, cand_exp, n, W, out):
    """Take up to W closest unexpanded candidates, mark them expanded."""
    k = 0
    for i in range(n):
        if not cand_exp[i]:
            cand_exp[i] = True
            out[k] = cand_ids[i]
            k += 1
            if k == W:
                break
    return k


@njit(cache=True, fastmath=True)
def _process_frontier(buf, row_of, off_of, frontier, nf, D, q, table, codes, tags, epoch,
                      cand_ids, cand_d, cand_exp, n, L, res_ids, res_d, n_res):
    """Exact distance for each fetched vertex; PQ distances for its unseen neighbours."""
    n_pq = 0
    for i in range(nf):
        row = buf[row_of[i]]
        o = off_of[i]
        vec = row[o:o + 4 * D].view(np.float32)
        res_ids[n_res] = frontier[i]
        res_d[n_res] = _l2sq(vec, q)
        n_res += 1
        deg = row[o + 4 * D:o + 4 * D + 4].view(np.int32)[0]
        nbrs = row[o + 4 * D + 4:o + 4 * D + 4 + 4 * deg].view(np.int32)
        for t in range(deg):
            v = nbrs[t]
            if tags[v] == epoch:
                continue
            tags[v] = epoch
            n_pq += 1
            _, n = _insert(cand_ids, cand_d, cand_exp, n, L, v, _pq_dist_one(codes, table, v))
    return n, n_res, n_pq


# ============================================================================ index
class DiskIndex:
    def __init__(self, index_dir, W=4, io="fadvise"):
        d = Path(index_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        m = self.meta
        self.N, self.D, self.R = m["N"], m["D"], m["R"]
        self.rec, self.nps, self.start = m["rec_size"], m["nodes_per_sector"], m["start"]
        self.codes = np.load(d / "pq_codes.npy")                     # [N, M] uint8  (RAM)
        self.centroids = np.load(d / "pq_centroids.npy")             # [M, 256, dsub] (RAM)
        self.M, _, self.dsub = self.centroids.shape
        self.tags = np.zeros(self.N, np.int32)
        self.epoch = 0
        self.W, self.io = W, io
        self.fd = os.open(d / "index.bin", os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)       # no readahead on random reads
        self.pool = ThreadPoolExecutor(max_workers=W) if io == "threads" else None
        self.reset_counters()

    def close(self):
        if self.pool:
            self.pool.shutdown()
        os.close(self.fd)

    def reset_counters(self):
        self.c = {"reads": 0, "bytes": 0, "rounds": 0, "expanded": 0, "pq_dists": 0}

    def ram_bytes(self):
        return int(self.codes.nbytes + self.centroids.nbytes + self.tags.nbytes)

    # ------------------------------------------------------------------ I/O
    def _pread(self, s):
        return os.pread(self.fd, SECTOR, s * SECTOR)

    def _read_sectors(self, sectors):
        if self.io == "threads" and len(sectors) > 1:
            chunks = list(self.pool.map(self._pread, sectors.tolist()))
        else:
            if self.io == "fadvise":                   # issue all reads asynchronously first
                for s in sectors.tolist():
                    os.posix_fadvise(self.fd, s * SECTOR, SECTOR, os.POSIX_FADV_WILLNEED)
            chunks = [self._pread(s) for s in sectors.tolist()]
        self.c["reads"] += len(chunks)
        self.c["bytes"] += SECTOR * len(chunks)
        return np.frombuffer(b"".join(chunks), np.uint8).reshape(len(chunks), SECTOR)

    # ------------------------------------------------------------------ search
    def search(self, q, k=10, L=50):
        q = np.ascontiguousarray(q, np.float32)
        L = max(L, k)
        W = self.W
        table = ((q.reshape(self.M, 1, self.dsub) - self.centroids) ** 2).sum(-1).astype(np.float32)
        self.epoch += 1
        ep = self.epoch
        cand_ids = np.empty(L, np.int32)
        cand_d = np.empty(L, np.float32)
        cand_exp = np.zeros(L, np.bool_)
        cap = 4 * L + 4 * W
        res_ids = np.empty(cap, np.int32)
        res_d = np.empty(cap, np.float32)
        frontier = np.empty(W, np.int32)
        s = self.start
        self.tags[s] = ep
        cand_ids[0], cand_d[0], n, n_res = s, _pq_dist_one(self.codes, table, s), 1, 0
        while True:
            nf = _pick_frontier(cand_ids, cand_exp, n, W, frontier)
            if nf == 0:
                break
            if n_res + nf > cap:                          # grow result buffers (rare)
                cap *= 2
                res_ids = np.resize(res_ids, cap)
                res_d = np.resize(res_d, cap)
            fr = frontier[:nf]
            sectors, row_of = np.unique(fr // self.nps, return_inverse=True)
            buf = self._read_sectors(sectors)
            off_of = (fr % self.nps) * self.rec
            n, n_res, n_pq = _process_frontier(buf, row_of.astype(np.int64), off_of.astype(np.int64), fr, nf,
                                               self.D, q, table, self.codes, self.tags, ep,
                                               cand_ids, cand_d, cand_exp, n, L, res_ids, res_d, n_res)
            self.c["rounds"] += 1
            self.c["expanded"] += nf
            self.c["pq_dists"] += n_pq
        kk = min(k, n_res)
        top = np.argsort(res_d[:n_res])[:kk]
        out = np.full(k, -1, np.int32)
        out[:kk] = res_ids[:n_res][top]
        return out

    def warmup_jit(self):
        """Compile kernels without touching the on-disk index (keeps cold runs cold)."""
        D, R = self.D, self.R
        fake = np.zeros((1, SECTOR), np.uint8)
        tags = np.zeros(4, np.int32)
        codes = np.zeros((4, self.M), np.uint8)
        table = np.zeros((self.M, 256), np.float32)
        ids, ds, ex = np.empty(4, np.int32), np.empty(4, np.float32), np.zeros(4, np.bool_)
        fr = np.zeros(1, np.int32)
        _pick_frontier(ids, ex, 0, 1, fr)
        _process_frontier(fake, np.zeros(1, np.int64), np.zeros(1, np.int64), fr, 1, D,
                          np.zeros(D, np.float32), table, codes, tags, 1, ids, ds, ex, 0, 4,
                          np.empty(4, np.int32), np.empty(4, np.float32), 0)
        _pq_dist_one(codes, table, 0)
