"""Benchmark utilities: thread control, system info, query timing, peak-memory tracking."""
import os
import platform
import resource
import statistics
import threading
import time

import psutil


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def set_threads(n):
    """Use the same thread count for FAISS (OpenMP) and our Numba kernels."""
    import faiss
    import numba
    faiss.omp_set_num_threads(n)
    numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))


def system_info(search_threads, query_mode):
    return {"cpu_model": cpu_model(), "logical_cpus": os.cpu_count(),
            "ram_gib": round(psutil.virtual_memory().total / 2**30, 1),
            "search_threads": search_threads, "query_mode": query_mode}


def time_queries(search_fn, queries, warmup=200, repeats=5, mode="batch"):
    """Time `search_fn` over `queries`.

    mode="batch"  : one call with all queries; latency = wall / nq.
    mode="single" : one call per query (queries issued individually).
    Returns median-of-repeats mean latency (ms) and QPS. Index loading is excluded (done by caller).
    """
    nq = len(queries)
    search_fn(queries[:min(warmup, nq)])                     # warm-up
    walls = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        if mode == "batch":
            search_fn(queries)
        else:
            for i in range(nq):
                search_fn(queries[i:i + 1])
        walls.append(time.perf_counter() - t0)
    w = statistics.median(walls)
    return {"mean_latency_ms": 1e3 * w / nq, "qps": nq / w,
            "timing_queries": nq, "timing_repeats": repeats,
            "timing_rel_spread": (max(walls) - min(walls)) / w}


def peak_rss_mib():
    """Peak resident set size of this process so far (Linux ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class RSSSampler:
    """Samples current RSS in a background thread; gives peak RSS during a code block."""

    def __init__(self, interval=0.2):
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self._proc = psutil.Process()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self._proc.memory_info().rss
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)

    @property
    def peak_mib(self):
        return self.peak / 2**20
