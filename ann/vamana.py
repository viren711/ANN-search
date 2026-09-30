"""Vamana graph index (DiskANN, Subramanya et al., NeurIPS 2019) — own implementation.

GreedySearch, RobustPrune and the Vamana build loop are written here from scratch; Numba JIT
compiles the performance-critical kernels.

Representation
  * vectors  : contiguous float32 [N, D] (row-major)
  * graph    : fixed-width int32 adjacency matrix [N, R] (unused slots = -1) + int32 degrees [N]
  * distances: squared L2 internally. The RobustPrune rule  alpha * d(p*, p') <= d(p, p')
               uses true Euclidean d, which with squared distances is
               alpha^2 * d2(p*, p') <= d2(p, p').
  * visitation: per-thread int32 "tag" arrays of length N with an epoch counter, so a search
               never clears an O(N) visited set (buffers are reused across searches).

Build modes
  * "serial" : literally the pseudocode (one point inserted at a time).
  * "batched": parallel batch insertion (as in DiskANN / ParlayANN). Points in a batch search
               the current graph and RobustPrune concurrently (graph read-only); results are
               committed; then reverse edges are grouped by target vertex and each target is
               updated by exactly one thread (race-free, deterministic). Batch sizes grow by
               prefix doubling 1, 2, 4, ... up to max_batch_frac * N.
"""
import json
import time
from pathlib import Path

import numba as nb
import numpy as np
from numba import njit, prange


# ============================================================================ kernels
@njit(inline="always", fastmath=True)
def _l2sq(a, b):
    s = np.float32(0.0)
    for i in range(a.shape[0]):
        t = a[i] - b[i]
        s += t * t
    return s


@njit(inline="always")
def _bump(epochs, tid):
    epochs[tid] += 1
    return epochs[tid]


@njit(inline="always")
def _insert(ids, ds, exp, n, cap, nid, nd):
    """Insert (nid, nd) into the ascending list ids/ds of length n (capacity cap).

    Returns (position, new_n); position = -1 if the list is full and nd is not better than
    the current worst.  When full, the worst element is evicted (keeps only the L best).
    """
    if n == cap and nd >= ds[n - 1]:
        return -1, n
    i = n if n < cap else cap - 1
    while i > 0 and ds[i - 1] > nd:
        ids[i] = ids[i - 1]
        ds[i] = ds[i - 1]
        exp[i] = exp[i - 1]
        i -= 1
    ids[i] = nid
    ds[i] = nd
    exp[i] = False
    if n < cap:
        n += 1
    return i, n


@njit(fastmath=True, cache=True)
def _greedy_search(data, graph, degrees, start, q, L, tags, epoch, record):
    """GreedySearch(G, s, q, k, L).

    Candidate set C is a sorted array of the L best points seen so far; `cur` points at the
    closest unexpanded candidate (everything before it is already expanded).  A vertex is
    "seen" once its distance is computed (tags[v] == epoch); it is never evaluated twice.

    Returns (n, cand_ids, cand_d, n_visited, vis_ids, vis_d, n_expanded, n_dist_evals).
    vis_* (the expanded set V with distances to q) is only filled when record=True (build).
    """
    cand_ids = np.empty(L, np.int32)
    cand_d = np.empty(L, np.float32)
    cand_exp = np.zeros(L, np.bool_)
    vcap = 2 * L + 16 if record else 1
    vis_ids = np.empty(vcap, np.int32)
    vis_d = np.empty(vcap, np.float32)
    nv = 0

    tags[start] = epoch
    cand_ids[0] = start
    cand_d[0] = _l2sq(data[start], q)
    n = 1
    n_dist = 1
    n_exp = 0
    cur = 0
    while cur < n:
        p = cand_ids[cur]
        cand_exp[cur] = True
        n_exp += 1
        if record:
            if nv == vcap:                               # grow visited buffer (rare)
                vcap *= 2
                t_ids = np.empty(vcap, np.int32)
                t_d = np.empty(vcap, np.float32)
                t_ids[:nv] = vis_ids[:nv]
                t_d[:nv] = vis_d[:nv]
                vis_ids, vis_d = t_ids, t_d
            vis_ids[nv] = p
            vis_d[nv] = cand_d[cur]
            nv += 1
        min_ins = L
        for t in range(degrees[p]):
            v = graph[p, t]
            if tags[v] == epoch:
                continue
            tags[v] = epoch
            d = _l2sq(data[v], q)
            n_dist += 1
            pos, n = _insert(cand_ids, cand_d, cand_exp, n, L, v, d)
            if pos >= 0 and pos < min_ins:
                min_ins = pos
        if min_ins < cur:
            cur = min_ins
        while cur < n and cand_exp[cur]:
            cur += 1
    return n, cand_ids, cand_d, nv, vis_ids, vis_d, n_exp, n_dist


@njit(fastmath=True, cache=True)
def _robust_prune(data, graph, degrees, p, cand_ids, cand_d, n_cand, alpha2, R, tags, epoch, out):
    """RobustPrune(G, p, V, alpha, R).  cand_d[i] = squared distance(p, cand_ids[i]).

    Pool = (V ∪ OutNeighbors(p)) \\ {p} (deduplicated), processed in increasing distance
    from p.  Writes the new out-neighbours into `out` and returns their count (<= R).
    The caller is responsible for writing `out` into the graph.
    """
    deg = degrees[p]
    m_cap = n_cand + deg
    pool_ids = np.empty(m_cap, np.int32)
    pool_d = np.empty(m_cap, np.float32)
    m = 0
    tags[p] = epoch                                      # excludes p itself
    for i in range(n_cand):
        c = cand_ids[i]
        if tags[c] != epoch:
            tags[c] = epoch
            pool_ids[m] = c
            pool_d[m] = cand_d[i]
            m += 1
    xp = data[p]
    for i in range(deg):
        c = graph[p, i]
        if tags[c] != epoch:
            tags[c] = epoch
            pool_ids[m] = c
            pool_d[m] = _l2sq(data[c], xp)
            m += 1
    order = np.argsort(pool_d[:m])
    ids = pool_ids[order]
    ds = pool_d[order]
    alive = np.ones(m, np.bool_)
    n_out = 0
    for i in range(m):
        if not alive[i]:
            continue
        ps = ids[i]
        out[n_out] = ps
        n_out += 1
        if n_out == R:
            break
        xs = data[ps]
        for j in range(i + 1, m):
            if alive[j] and alpha2 * _l2sq(xs, data[ids[j]]) <= ds[j]:
                alive[j] = False
    return n_out


@njit(fastmath=True, cache=True)
def _add_back_edges(data, graph, degrees, j, srcs, n_src, alpha2, R, tags, epochs, tid, out):
    """For each p in srcs: C <- OutNeighbors(j) ∪ {p}; if |C| > R RobustPrune(j, C) else set."""
    e = _bump(epochs, tid)
    deg = degrees[j]
    tags[j] = e
    for t in range(deg):
        tags[graph[j, t]] = e
    new = np.empty(n_src, np.int32)
    k = 0
    for t in range(n_src):
        s = srcs[t]
        if tags[s] != e:
            tags[s] = e
            new[k] = s
            k += 1
    if k == 0:
        return
    if deg + k <= R:
        graph[j, deg:deg + k] = new[:k]
        degrees[j] = deg + k
        return
    xj = data[j]
    nd = np.empty(k, np.float32)
    for t in range(k):
        nd[t] = _l2sq(data[new[t]], xj)
    e = _bump(epochs, tid)
    n_out = _robust_prune(data, graph, degrees, j, new, nd, k, alpha2, R, tags, e, out)
    graph[j, :n_out] = out[:n_out]
    graph[j, n_out:] = -1
    degrees[j] = n_out


@njit(fastmath=True, cache=True)
def _pass_serial(data, graph, degrees, start, order, L, alpha2, R, tags, epochs):
    """One Vamana pass, exactly as in the pseudocode (single-threaded)."""
    out = np.empty(R, np.int32)
    scratch = np.empty(R, np.int32)
    one = np.empty(1, np.int32)
    t0 = tags[0]
    for idx in range(order.shape[0]):
        p = order[idx]
        e = _bump(epochs, 0)
        res = _greedy_search(data, graph, degrees, start, data[p], L, t0, e, True)
        e = _bump(epochs, 0)
        n_out = _robust_prune(data, graph, degrees, p, res[4], res[5], res[3], alpha2, R, t0, e, out)
        graph[p, :n_out] = out[:n_out]
        graph[p, n_out:] = -1
        degrees[p] = n_out
        one[0] = p
        for t in range(n_out):
            _add_back_edges(data, graph, degrees, out[t], one, 1, alpha2, R, t0, epochs, 0, scratch)


@njit(parallel=True, fastmath=True, cache=True)
def _batch_search_prune(data, graph, degrees, start, batch, L, alpha2, R, tags, epochs, new_out, new_deg):
    """Phase 1 of a batch: GreedySearch + RobustPrune for every batch point (graph read-only)."""
    for i in prange(batch.shape[0]):
        tid = nb.get_thread_id()
        p = batch[i]
        e = _bump(epochs, tid)
        res = _greedy_search(data, graph, degrees, start, data[p], L, tags[tid], e, True)
        e = _bump(epochs, tid)
        new_deg[i] = _robust_prune(data, graph, degrees, p, res[4], res[5], res[3],
                                   alpha2, R, tags[tid], e, new_out[i])


@njit(parallel=True, cache=True)
def _batch_commit(graph, degrees, batch, new_out, new_deg):
    for i in prange(batch.shape[0]):
        p = batch[i]
        d = new_deg[i]
        graph[p, :d] = new_out[i, :d]
        graph[p, d:] = -1
        degrees[p] = d


@njit(parallel=True, fastmath=True, cache=True)
def _batch_back_edges(data, graph, degrees, batch, new_out, new_deg, alpha2, R, tags, epochs):
    """Phase 2: group reverse edges (j <- p) by target j; each j is updated by one thread."""
    nb_ = batch.shape[0]
    offs = np.zeros(nb_ + 1, np.int64)
    for i in range(nb_):
        offs[i + 1] = offs[i] + new_deg[i]
    total = offs[nb_]
    tgt = np.empty(total, np.int32)
    src = np.empty(total, np.int32)
    for i in prange(nb_):
        o = offs[i]
        for t in range(new_deg[i]):
            tgt[o + t] = new_out[i, t]
            src[o + t] = batch[i]
    order = np.argsort(tgt, kind="mergesort")
    tgt_s = tgt[order]
    src_s = src[order]
    # group boundaries
    n_groups = 0
    for t in range(total):
        if t == 0 or tgt_s[t] != tgt_s[t - 1]:
            n_groups += 1
    gstart = np.empty(n_groups + 1, np.int64)
    g = 0
    for t in range(total):
        if t == 0 or tgt_s[t] != tgt_s[t - 1]:
            gstart[g] = t
            g += 1
    gstart[n_groups] = total
    for gi in prange(n_groups):
        tid = nb.get_thread_id()
        a = gstart[gi]
        b = gstart[gi + 1]
        out = np.empty(R, np.int32)
        _add_back_edges(data, graph, degrees, tgt_s[a], src_s[a:b], b - a, alpha2, R,
                        tags[tid], epochs, tid, out)


@njit(parallel=True, fastmath=True, cache=True)
def _search_batch(data, graph, degrees, start, queries, k, L, tags, epochs,
                  out_ids, out_d, n_exp, n_dist):
    for qi in prange(queries.shape[0]):
        tid = nb.get_thread_id()
        e = _bump(epochs, tid)
        res = _greedy_search(data, graph, degrees, start, queries[qi], L, tags[tid], e, False)
        n = res[0]
        kk = min(k, n)
        for t in range(kk):
            out_ids[qi, t] = res[1][t]
            out_d[qi, t] = res[2][t]
        for t in range(kk, k):
            out_ids[qi, t] = -1
            out_d[qi, t] = np.inf
        n_exp[qi] = res[6]
        n_dist[qi] = res[7]


# ============================================================================ helpers
def random_regular_graph(N, R, seed):
    """Random directed graph with out-degree min(R, N-1), no self loops, no duplicate edges."""
    rng = np.random.default_rng(seed)
    deg = min(R, N - 1)
    g = np.full((N, R), -1, np.int32)
    chunk = 1 << 16
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        rows = np.arange(s, e)[:, None]
        c = rng.integers(0, N - 1, size=(e - s, deg))
        c += c >= rows                                   # skip self
        srt = np.sort(c, axis=1)
        bad = np.nonzero((srt[:, 1:] == srt[:, :-1]).any(axis=1))[0]
        for b in bad:                                    # resample rows with duplicates
            r = rng.choice(N - 1, deg, replace=False)
            c[b] = r + (r >= s + b)
        g[s:e, :deg] = c
    return g, np.full(N, deg, np.int32)


def approx_medoid(data, chunk=1 << 18):
    """Database point nearest to the dataset mean (practical medoid approximation)."""
    mean = data.mean(axis=0, dtype=np.float64).astype(np.float32)
    best, best_d = -1, np.inf
    for s in range(0, data.shape[0], chunk):
        d = ((data[s:s + chunk] - mean) ** 2).sum(axis=1)
        i = int(np.argmin(d))
        if d[i] < best_d:
            best, best_d = s + i, float(d[i])
    return best


def _new_tags(N):
    T = nb.config.NUMBA_NUM_THREADS
    return np.zeros((T, N), np.int32), np.zeros(T, np.int32)


# ============================================================================ index class
class VamanaIndex:
    def __init__(self, data, graph, degrees, start, params=None):
        self.data = data
        self.graph = graph
        self.degrees = degrees
        self.start = int(start)
        self.params = params or {}
        self._tags, self._epochs = _new_tags(data.shape[0])

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, data, R=64, L=100, alpha=1.2, seed=0, mode="batched",
              max_batch_frac=0.02, verbose=True):
        assert data.dtype == np.float32 and data.flags.c_contiguous
        N = data.shape[0]
        L = max(L, 1)
        t_all = time.perf_counter()
        graph, degrees = random_regular_graph(N, R, seed)
        start = approx_medoid(data)
        tags, epochs = _new_tags(N)
        rng = np.random.default_rng(seed + 1)
        max_batch = max(1, int(max_batch_frac * N))
        new_out = np.empty((max_batch, R), np.int32)
        new_deg = np.empty(max_batch, np.int32)
        pass_times = []
        for alpha_pass in (1.0, float(alpha)):
            t0 = time.perf_counter()
            a2 = np.float32(alpha_pass * alpha_pass)
            order = rng.permutation(N).astype(np.int32)
            if mode == "serial":
                _pass_serial(data, graph, degrees, start, order, L, a2, R, tags, epochs)
            else:
                pos, bs = 0, 1
                while pos < N:
                    b = order[pos:pos + bs]
                    nb_ = len(b)
                    _batch_search_prune(data, graph, degrees, start, b, L, a2, R, tags, epochs,
                                        new_out[:nb_], new_deg[:nb_])
                    _batch_commit(graph, degrees, b, new_out[:nb_], new_deg[:nb_])
                    _batch_back_edges(data, graph, degrees, b, new_out[:nb_], new_deg[:nb_],
                                      a2, R, tags, epochs)
                    pos += nb_
                    bs = min(bs * 2, max_batch)
            pass_times.append(time.perf_counter() - t0)
            if verbose:
                print(f"  pass alpha={alpha_pass}: {pass_times[-1]:.1f}s  "
                      f"avg deg {degrees.mean():.2f}  max deg {degrees.max()}", flush=True)
        params = dict(R=R, L_build=L, alpha=float(alpha), seed=seed, mode=mode,
                      max_batch_frac=max_batch_frac, pass_times_s=pass_times,
                      build_time_s=time.perf_counter() - t_all, entry_point=int(start),
                      entry_point_procedure="database point nearest to the dataset mean")
        return cls(data, graph, degrees, start, params)

    # ------------------------------------------------------------------ search
    def search(self, queries, k=10, L=50, return_stats=False):
        """Batch GreedySearch from the entry point with query-time list size L (L >= k)."""
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        nq = queries.shape[0]
        L = max(int(L), k)
        ids = np.empty((nq, k), np.int32)
        dists = np.empty((nq, k), np.float32)
        n_exp = np.empty(nq, np.int32)
        n_dist = np.empty(nq, np.int32)
        _search_batch(self.data, self.graph, self.degrees, self.start, queries, k, L,
                      self._tags, self._epochs, ids, dists, n_exp, n_dist)
        if return_stats:
            return ids, dists, {"expanded": n_exp, "dist_evals": n_dist}
        return ids, dists

    # ------------------------------------------------------------------ stats / io
    def graph_stats(self):
        R = self.graph.shape[1]
        assert int(self.degrees.max()) <= R, "out-degree exceeds R!"
        return {"avg_out_degree": float(self.degrees.mean()),
                "max_out_degree": int(self.degrees.max()),
                "min_out_degree": int(self.degrees.min()),
                "graph_bytes": int(self.graph.nbytes + self.degrees.nbytes)}

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "graph.npy", self.graph)
        np.save(path / "degrees.npy", self.degrees)
        meta = dict(self.params, start=self.start, N=int(self.data.shape[0]),
                    D=int(self.data.shape[1]), **self.graph_stats())
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        return sum(f.stat().st_size for f in path.iterdir())

    @classmethod
    def load(cls, path, data):
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        return cls(data, np.load(path / "graph.npy"), np.load(path / "degrees.npy"),
                   meta["start"], meta)
