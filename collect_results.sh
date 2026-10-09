#!/usr/bin/env bash
# Package everything needed for analysis/report into one small archive (no datasets, no indices).
set -euo pipefail
cd "$(dirname "$0")"
OUT=results_bundle_$(hostname -s)_$(date +%Y%m%d_%H%M).tar.gz
FILES=(results logs)
for f in dataset/processed/*/meta.json indices/vamana/*/build_stats.json indices/vamana/*/meta.json \
         indices/disk/*/meta.json indices/faiss/*.json; do
  [[ -e $f ]] && FILES+=("$f")
done
tar czf "$OUT" "${FILES[@]}"
echo "Created $OUT ($(du -h "$OUT" | cut -f1)) — copy this file back for analysis."
