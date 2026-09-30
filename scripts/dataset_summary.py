"""Dataset-summary deliverable table (corpus size, #queries, dims, raw 128-d float32 memory, PCA info)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from ann.data import DATASETS, RESULTS, load_meta

rows = []
for name in DATASETS:
    m = load_meta(name)
    if not m:
        continue
    rows.append({"dataset": name, "corpus_size": m["corpus_size"], "num_queries": m["num_queries"],
                 "original_dim": m["original_dim"], "search_dim": m["search_dim"],
                 "raw_128d_float32_MiB": round(m["corpus_size"] * m["search_dim"] * 4 / 2**20, 1),
                 "pca_impl": m.get("pca_impl", "-"), "pca_sample": m.get("pca_fit_sample_size", "-"),
                 "pca_seed": m.get("pca_seed", "-"),
                 "pca_var_explained": round(m["pca_variance_explained"], 4) if "pca_variance_explained" in m else "-"})
df = pd.DataFrame(rows)
RESULTS.mkdir(exist_ok=True)
df.to_csv(RESULTS / "dataset_summary.csv", index=False)
try:
    txt = df.to_markdown(index=False)
except ImportError:                                      # tabulate not installed
    txt = df.to_string(index=False)
(RESULTS / "dataset_summary.md").write_text(txt)
print(df.to_string(index=False))
