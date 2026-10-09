# ANN-search — CS6101 PA2: Vamana / DiskANN vs FAISS HNSW

Our own Vamana implementation (GreedySearch, RobustPrune, two-pass build; Python + Numba) is
compared with FAISS exact search and FAISS HNSW on SIFT1M, BEIR-Quora and BEIR-NQ (Cohere v3
embeddings, PCA-128). Extra credit: disk-resident DiskANN-style search over 8.8M MS MARCO
passages with verified cold and warm page-cache runs.

**Full write-up of the problem, the design and every result: [`assignment.txt`](assignment.txt).**
Step-by-step plan and status: [`project.txt`](project.txt). Commands for the server: [`COMMANDS.txt`](COMMANDS.txt).

## Status

| part | status | where |
|---|---|---|
| Data prep (SIFT1M, SIFT100K, Quora, NQ, MS MARCO; PCA-128; exact GT) | done | `server_results/results/dataset_summary.csv` |
| Part 1: FAISS exact + HNSW (7 efSearch points) | done | `server_results/results/<ds>/faiss.csv` |
| Part 2: Vamana builds, scaling test, sanity query | done | `server_results/indices/vamana/*/build_stats.json` |
| Part 3a: SIFT100K R / α / L_search study | done | `server_results/results/sift100k/param_study.csv` |
| Part 3b: Vamana vs HNSW vs exact | done | `server_results/results/<ds>/compare_*` |
| Extra credit 1: disk-resident search, cold and warm | done | `server_results/results/extra_credit/ec1_msmarco.*` |
| Extra credit 2: HDD vs SSD | **not possible**: no HDD on the server or the laptop | `assignment.txt` section 9 |
| Follow-up experiments (optional, ~30-40 min) | ready, not yet run | `./run_all.sh followup` |

## Headline results (server: 2x Xeon Gold 5220R, 96 threads for builds, 1 thread for queries)

Fastest setting reaching the 10-recall@10 target. Latency in ms per query; exact search is
batch / one-at-a-time.

| dataset | target | FAISS HNSW (M=32, efC=200) | Vamana (R=64, L=100, α=1.2) | exact |
|---|---|---|---|---|
| SIFT1M (1M) | 0.99 | efS=128: 0.9951, 0.547 ms | L=75: 0.9931, **0.430 ms** | 3.37 / 41.0 ms |
| Quora (523K) | 0.99 | efS=32: 0.9963, 0.114 ms | L=20: 0.9968, 0.116 ms | 1.76 / 21.2 ms |
| NQ (2.68M) | 0.98 | efS=128: 0.9913, 0.670 ms | L=100: 0.9847, **0.549 ms** | 9.01 / 109.3 ms |
| NQ (2.68M) | 0.99 | efS=128: 0.9913, **0.670 ms** | L=200: 0.9924, 0.982 ms | |

| dataset | Vamana build | HNSW build | Vamana graph+vectors | HNSW index |
|---|---|---|---|---|
| SIFT1M | 52.8 s | 22.5 s | 772 MB | 784 MB |
| Quora | 23.0 s | 11.4 s | 404 MB | 410 MB |
| NQ | 160.9 s | 75.5 s | 2,070 MB | 2,103 MB |
| MS MARCO (8.84M) | 459 s | – | 6,826 MB | – |

- Vamana matches HNSW on SIFT1M and Quora up to ~0.995 recall and is faster on NQ up to
  ~0.985. HNSW is 1.5-2.2x faster for NQ at >= 0.99, where a few hard queries need long
  walks from Vamana's single entry point.
- Vamana builds 2.0-2.4x slower (two passes); index sizes are the same.
- Disk-resident MS MARCO, L=100, W=4: 10-recall@10 = 0.970. Cold 7.35 ms, warm 1.68 ms
  per query. Peak RAM 460 MiB under an 8 GiB RLIMIT_AS cap. The index was 0% resident
  (fincore) before every cold run.

Figures: `server_results/results/<ds>/compare_recall_latency.png`,
`server_results/results/sift100k/param_recall_vs_*.png`,
`server_results/results/extra_credit/ec1_msmarco_recall_latency.png`. Redraw all of them from the
CSVs with `python scripts/replot.py --root server_results/results`.

## Running on the (offline) server

**Exact commands: [`COMMANDS.txt`](COMMANDS.txt).** In short:

```bash
unzip -q ANN-search_offline.zip -d ANN-search && cd ANN-search
tar xf ~/ANN-search_data_compulsory.tar          # prepared 128-d datasets
tar xf ~/ANN-search_data_msmarco.tar             # extra credit only
./setup_env.sh                                   # offline: bundled CPython 3.12 + wheels
./run_all.sh smoke && ./run_all.sh && ./run_all.sh ec
./run_all.sh followup                            # optional follow-up experiments
```

- The server needs no internet. `offline/` carries a portable CPython 3.12
  (python-build-standalone) and every dependency as a manylinux (glibc >= 2.28) x86_64 wheel.
  The datasets travel already PCA-reduced, so the `data` stage only verifies them.
- Every stage logs to `logs/<stage>.log`. Finished stages are skipped on re-run (resume after a
  crash), and `FORCE=1` redoes them. Stages: `sysinfo data gt summary part1 scaling part2
  part3a part3b ec_prep ec1 ec2 followup report collect`.
- All parameters live in `config.env`, and any of them can be overridden from the shell,
  e.g. `BUILD_THREADS=32 ./run_all.sh part2`.
- At the end, `collect_results.sh` writes `results_bundle_<host>_<date>.tar.gz` (a few MB),
  the only file to copy back. Unpack it into `server_results/`.

### Transfer files (built on a machine with internet)

| command | output | when |
|---|---|---|
| `./make_bundle.sh` | `dist/ANN-search_offline.zip` (code + Python + wheels, 289 MB), `dist/ANN-search_data_compulsory.tar` (2.2 GB), `dist/ANN-search_data_msmarco.tar` (4.3 GB), `dist/SHA256SUMS` | first install |
| `./make_bundle.sh patch` | `dist/ANN-search_code_patch.zip` (code only, < 100 KB) | code updates on an already-installed server |

### Hardware notes

- **GPUs are not used, on purpose.** The assignment compares HNSW and Vamana by *CPU* query
  latency under identical thread settings, and the Vamana code must be our own.
- Builds use `BUILD_THREADS` (default: all logical CPUs). Query timing always uses
  `SEARCH_THREADS=1` for every method, and `results/system/` records the hardware.
- Each timing row has `timing_rel_spread` ((max-min)/median over 5 repeats). Above ~0.1,
  re-run that stage with `FORCE=1`.
- Disk: ~15 GB compulsory and ~30 GB extra credit. RAM: the MS MARCO graph build peaked at
  10.1 GiB with 96 threads; the disk-resident search uses ~0.5 GB.

## Layout

```
ann/             library: data.py (paths, I/O), vamana.py (our Vamana), diskann.py (disk search),
                 pagecache.py (fincore / fadvise), bench.py (timing), metrics.py, plots.py
scripts/         one script per step (project.txt lists them in order)
server_results/  the results bundle from the server run (tables, plots, logs, metadata)
assignment.txt   problem, solution and analysis      project.txt  step plan / status
run_all.sh       pipeline driver   config.env  parameters   setup_env.sh  offline install
COMMANDS.txt     commands for the server               make_bundle.sh  transfer files
```

Local `results/` and `logs/` only hold laptop dev and smoke runs and are not committed.
The authoritative numbers are in `server_results/`.
