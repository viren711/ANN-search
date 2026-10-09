#!/usr/bin/env bash
# Build the transfer files for an OFFLINE server (run on a machine with internet + prepared data).
#
#   ./make_bundle.sh            -> code + compulsory data (+ msmarco data if prepared)
#   ./make_bundle.sh code       -> only the code/runtime zip (after code changes)
#   ./make_bundle.sh patch      -> code-only zip (<100 KB) for a server that is already set up:
#                                  on the server:  unzip -o ANN-search_code_patch.zip -d <project dir>
#
# dist/ANN-search_offline.zip          code + portable CPython 3.12 + all wheels (~300 MB)
# dist/ANN-search_data_compulsory.tar  dataset/processed/{sift1m,sift100k,quora,nq}   (~2.2 GB)
# dist/ANN-search_data_msmarco.tar     dataset/processed/msmarco (extra credit)       (~4.6 GB)
# dist/SHA256SUMS                      checksums to verify the transfer on the server
set -euo pipefail
cd "$(dirname "$0")"
WHAT=${1:-all}
mkdir -p dist

if [[ $WHAT == patch ]]; then
  rm -f dist/ANN-search_code_patch.zip
  zip -qr dist/ANN-search_code_patch.zip ann scripts run_all.sh setup_env.sh collect_results.sh config.env \
      requirements.txt README.md project.txt COMMANDS.txt assignment.txt -x '*__pycache__*' '*.pyc'
  echo "Created dist/ANN-search_code_patch.zip ($(du -h dist/ANN-search_code_patch.zip | cut -f1))"
  (cd dist && sha256sum ANN-search_code_patch.zip)
  exit 0
fi

if [[ $WHAT == all || $WHAT == code ]]; then
  [[ -f offline/cpython-3.12-x86_64-linux-gnu.tar.gz && -d offline/wheels ]] \
    || { echo "offline/ runtime missing (see project.txt section 6c)"; exit 1; }
  rm -f dist/ANN-search_offline.zip
  zip -qr dist/ANN-search_offline.zip ann scripts run_all.sh setup_env.sh collect_results.sh config.env \
      requirements.txt README.md project.txt COMMANDS.txt assignment.txt \
      offline/cpython-3.12-x86_64-linux-gnu.tar.gz offline/wheels -x '*__pycache__*' '*.pyc'
  echo "Created dist/ANN-search_offline.zip ($(du -h dist/ANN-search_offline.zip | cut -f1))"
fi

if [[ $WHAT == all || $WHAT == data ]]; then
  tar cf dist/ANN-search_data_compulsory.tar dataset/processed/{sift1m,sift100k,quora,nq}
  echo "Created dist/ANN-search_data_compulsory.tar ($(du -h dist/ANN-search_data_compulsory.tar | cut -f1))"
  if [[ -f dataset/processed/msmarco/base.npy ]]; then
    tar cf dist/ANN-search_data_msmarco.tar \
        dataset/processed/msmarco/{base.npy,query.npy,query_ids.txt,meta.json,pca.npz}
    echo "Created dist/ANN-search_data_msmarco.tar ($(du -h dist/ANN-search_data_msmarco.tar | cut -f1))"
  else
    echo "msmarco not prepared -> no msmarco data archive (extra credit needs it)"
  fi
fi

(cd dist && sha256sum ANN-search_*.zip ANN-search_*.tar > SHA256SUMS 2>/dev/null; cat SHA256SUMS)
