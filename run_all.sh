#!/usr/bin/env bash
# =============================================================================
# CS6101 PA2 — complete experiment pipeline.
#
#   ./run_all.sh smoke            quick end-to-end check on SIFT100K (~5 min)  <- run first
#   ./run_all.sh                  all compulsory stages (Parts 1-3)
#   ./run_all.sh ec               extra credit (MS MARCO disk-resident search, HDD vs SSD)
#   ./run_all.sh all              compulsory + extra credit
#   ./run_all.sh followup         follow-up experiments after the first full run
#   ./run_all.sh part1 part3b     any list of individual stages (sysinfo data gt summary part1 scaling
#                                 part2 part3a part3b ec_prep ec1 ec2 report collect)
#
# Each stage writes logs/<stage>.log and logs/stages/<stage>.done; finished stages are
# skipped on re-run (resume after interruption). FORCE=1 re-runs them.
# Settings: config.env (override any of them from the shell).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./config.env
PY=${PY:-.venv/bin/python}
export NUMBA_NUM_THREADS=$BUILD_THREADS PYTHONUNBUFFERED=1 BUILD_THREADS SEARCH_THREADS
mkdir -p logs/stages results

COMPULSORY="sysinfo data gt summary part1 scaling part2 part3a part3b report collect"
EXTRA="sysinfo ec_prep ec1 ec2 report collect"

T=(--threads "$SEARCH_THREADS" --mode "$QUERY_MODE" --timing-queries "$TIMING_QUERIES" --repeats "$REPEATS")
VAMANA=(--R "$VAMANA_R" --L "$VAMANA_L" --alpha "$VAMANA_ALPHA")

# ----------------------------------------------------------------------------- stages
stage_sysinfo()  { $PY scripts/sysinfo.py; }
# Data: use dataset/processed/<ds> if present (offline transfer); else build it from
# dataset/raw; download only if raw data is missing too (needs internet).
ensure_data() {
  local ds=$1
  if [[ -f dataset/processed/$ds/base.npy && -f dataset/processed/$ds/query.npy ]]; then
    echo "[$ds] prepared data present"; return
  fi
  echo "[$ds] prepared data NOT found in $PWD/dataset/processed/$ds"
  echo "      (offline server? extract ANN-search_data_*.tar inside $PWD, then re-run)"
  case $ds in
    sift1m|sift100k)
      [[ -f dataset/raw/sift/sift_base.fvecs ]] || $PY scripts/download_data.py --only sift
      $PY scripts/prepare_sift.py ;;
    *)
      [[ -d dataset/raw/beir/$ds/corpus ]] || $PY scripts/download_data.py --only "$ds"
      $PY scripts/prepare_beir.py --dataset "$ds" --sample "$PCA_SAMPLE" --seed "$PCA_SEED" ;;
  esac
}
stage_data()     {
  for ds in sift1m sift100k quora nq; do ensure_data $ds; done
  $PY scripts/check_data.py sift1m sift100k quora nq
}
stage_gt()       { $PY scripts/compute_gt.py --dataset sift1m sift100k quora nq; }
stage_summary()  { $PY scripts/dataset_summary.py; }
stage_part1()    {
  for ds in sift100k $MAIN_DATASETS; do
    $PY scripts/run_faiss.py --dataset $ds --M "$HNSW_M" --efc "$HNSW_EFC" --ef $HNSW_EF \
        --build-threads "$BUILD_THREADS" "${T[@]}"
  done
}
stage_scaling()  {
  for ds in $SCALING_DATASETS; do
    $PY scripts/scaling_test.py --dataset $ds --sizes $SCALING_SIZES "${VAMANA[@]}" --threads "$BUILD_THREADS"
  done
}
stage_part2()    {
  for ds in sift100k $MAIN_DATASETS; do
    $PY scripts/build_vamana.py --dataset $ds "${VAMANA[@]}" --threads "$BUILD_THREADS" --skip-existing
  done
  $PY scripts/sanity_query.py "${VAMANA[@]}"
}
stage_part3a()   { $PY scripts/param_study.py --configs $PARAM_CONFIGS --Ls $PARAM_LS \
                       --threads "$SEARCH_THREADS" --build-threads "$BUILD_THREADS"; }
stage_part3b()   {
  for ds in $MAIN_DATASETS; do
    $PY scripts/search_vamana.py --dataset $ds "${VAMANA[@]}" --Ls $VAMANA_LS "${T[@]}"
    $PY scripts/compare.py --dataset $ds --vamana-csv "results/$ds/vamana_R${VAMANA_R}_L${VAMANA_L}_a${VAMANA_ALPHA}.csv"
  done
}
EC_IDX="indices/disk/${EC_DATASET}_R${EC_R}_L${EC_L}_a${EC_ALPHA}_pq${EC_PQ_M}"
EC_ARGS=(--dataset "$EC_DATASET" --Ls $EC_LS --W "$EC_W" --io "$EC_IO" --num-queries "$EC_NUM_QUERIES"
         --num-queries-hdd "$EC_NUM_QUERIES_HDD" --ram-limit-gib "$EC_RAM_LIMIT_GIB")
stage_ec_prep()  {
  ensure_data "$EC_DATASET"
  $PY scripts/check_data.py "$EC_DATASET"
  $PY scripts/compute_gt.py --dataset "$EC_DATASET"
}
stage_ec1()      {
  $PY scripts/build_vamana.py --dataset "$EC_DATASET" --R "$EC_R" --L "$EC_L" --alpha "$EC_ALPHA" \
      --threads "$BUILD_THREADS" --skip-existing
  $PY scripts/ec_build_disk_index.py --dataset "$EC_DATASET" --R "$EC_R" --L "$EC_L" --alpha "$EC_ALPHA" \
      --pq-m "$EC_PQ_M" --threads "$BUILD_THREADS" --skip-existing
  local dst=${EC_SSD_DIR:-$EC_IDX}
  $PY scripts/ec_cold_warm.py "${EC_ARGS[@]}" --src "$EC_IDX" --copies "ssd=$dst" --out-name ec1_${EC_DATASET}
}
stage_ec2()      {
  if [[ -z "$EC_HDD_DIR" || -z "$EC_SSD_DIR" ]]; then
    echo "EC2 skipped: set EC_HDD_DIR and EC_SSD_DIR (two different physical drives) in config.env"
    touch logs/stages/ec2.nomark; return 0
  fi
  $PY scripts/ec_cold_warm.py "${EC_ARGS[@]}" --src "$EC_IDX" \
      --copies "hdd=$EC_HDD_DIR" "ssd=$EC_SSD_DIR" --out-name ec2_${EC_DATASET}_hdd_vs_ssd
}
# Follow-up experiments after the first full run (see assignment.txt, "Further experiments"):
#  (1) HNSW distance/hop counters, (2) MS MARCO in-memory Vamana reference for the disk index,
#  (3) disk search at higher L, (4) beam-width ablation W in {1,2,8} (W=4 is the main run),
#  (5) one denser NQ graph (R=96, L_build=200) as a diagnostic (FOLLOWUP_NQ_VARIANT=0 skips it).
stage_followup() {
  for ds in sift100k $MAIN_DATASETS; do
    $PY scripts/hnsw_counts.py --dataset $ds --M "$HNSW_M" --efc "$HNSW_EFC" --ef $HNSW_EF
  done
  for ds in $MAIN_DATASETS; do
    $PY scripts/compare.py --dataset $ds --vamana-csv "results/$ds/vamana_R${VAMANA_R}_L${VAMANA_L}_a${VAMANA_ALPHA}.csv"
  done
  $PY scripts/search_vamana.py --dataset "$EC_DATASET" --R "$EC_R" --L "$EC_L" --alpha "$EC_ALPHA" \
      --Ls 50 100 200 300 500 "${T[@]}"
  local base=(--dataset "$EC_DATASET" --src "$EC_IDX" --copies "ssd=${EC_SSD_DIR:-$EC_IDX}" --io "$EC_IO"
              --num-queries "$EC_NUM_QUERIES" --ram-limit-gib "$EC_RAM_LIMIT_GIB")
  $PY scripts/ec_cold_warm.py "${base[@]}" --Ls 300 500 --W "$EC_W" --out-name ec1_${EC_DATASET}_highL
  for w in 1 2 8; do
    $PY scripts/ec_cold_warm.py "${base[@]}" --Ls 100 --W $w --out-name ec1_${EC_DATASET}_W$w
  done
  # (5) NQ diagnostic: ONE denser graph, to test whether Vamana's high-recall gap to HNSW on NQ
  #     (>=0.99 10-recall@10) is a graph-quality effect. Not a parameter sweep.
  if [[ -n "${FOLLOWUP_NQ_VARIANT:-1}" && "${FOLLOWUP_NQ_VARIANT:-1}" != 0 ]]; then
    $PY scripts/build_vamana.py --dataset nq --R 96 --L 200 --alpha 1.2 --threads "$BUILD_THREADS" --skip-existing
    $PY scripts/search_vamana.py --dataset nq --R 96 --L 200 --alpha 1.2 --Ls $VAMANA_LS "${T[@]}"
  fi
}
stage_report()   { $PY scripts/make_report_tables.py; }
stage_collect()  { bash collect_results.sh; }

# Fast end-to-end check of every code path on SIFT100K (no markers written).
stage_smoke()    {
  $PY scripts/sysinfo.py
  ensure_data sift100k
  $PY scripts/check_data.py sift100k
  $PY scripts/compute_gt.py --dataset sift100k
  $PY scripts/run_faiss.py --dataset sift100k --ef 16 64 256 --build-threads "$BUILD_THREADS" \
      --threads "$SEARCH_THREADS" --mode "$QUERY_MODE" --timing-queries 1000 --repeats 2
  $PY scripts/build_vamana.py --dataset sift100k "${VAMANA[@]}" --threads "$BUILD_THREADS" --skip-existing
  $PY scripts/search_vamana.py --dataset sift100k "${VAMANA[@]}" --Ls 10 50 100 --threads "$SEARCH_THREADS" \
      --timing-queries 1000 --repeats 2
  $PY scripts/sanity_query.py "${VAMANA[@]}"
  $PY scripts/ec_build_disk_index.py --dataset sift100k "${VAMANA[@]}" --pq-m "$EC_PQ_M" --threads "$BUILD_THREADS" --skip-existing
  $PY scripts/ec_cold_warm.py --dataset sift100k --copies "ssd=indices/disk/sift100k_R${VAMANA_R}_L${VAMANA_L}_a${VAMANA_ALPHA}_pq${EC_PQ_M}" \
      --Ls 50 --num-queries 300 --W "$EC_W" --io "$EC_IO" --out-name smoke_sift100k
  $PY scripts/make_report_tables.py
  echo "SMOKE TEST PASSED"
}

# ----------------------------------------------------------------------------- driver
ALWAYS_RUN=" sysinfo report collect smoke "      # cheap stages: never skipped, no marker
run_stage() {
  local s=$1 marker=logs/stages/$1.done rc
  declare -F "stage_$s" >/dev/null || { echo "unknown stage '$s'"; exit 1; }
  if [[ $ALWAYS_RUN != *" $s "* && -f $marker && -z "${FORCE:-}" ]]; then
    echo "=== [$s] already done (FORCE=1 to redo)"; return
  fi
  echo "=== [$s] start $(date '+%F %T')  (log: logs/$s.log)"
  local t0=$SECONDS
  # run in a subshell with -e active (bash ignores -e inside if/&&/|| contexts)
  set +e
  ( set -e; "stage_$s" ) 2>&1 | tee "logs/$s.log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "!!! [$s] FAILED (exit $rc) after $(( (SECONDS - t0) / 60 )) min — see logs/$s.log"; exit 1
  fi
  if [[ -f logs/stages/$s.nomark ]]; then rm -f "logs/stages/$s.nomark"      # stage skipped itself
  elif [[ $ALWAYS_RUN != *" $s "* ]]; then date '+%F %T' > "$marker"; fi
  echo "=== [$s] done in $(( (SECONDS - t0) / 60 )) min"
}

[[ -x $PY ]] || { echo "Python env not found ($PY). Run ./setup_env.sh first (or set PY=...)."; exit 1; }
case "${1:-compulsory}" in
  compulsory) STAGES=$COMPULSORY ;;
  ec)         STAGES=$EXTRA ;;
  followup)   STAGES="sysinfo followup report collect" ;;
  all)        STAGES="$COMPULSORY ec_prep ec1 ec2 report collect" ;;
  *)          STAGES="$*" ;;
esac
echo "Stages: $STAGES"
echo "BUILD_THREADS=$BUILD_THREADS SEARCH_THREADS=$SEARCH_THREADS QUERY_MODE=$QUERY_MODE PY=$PY"
for s in $STAGES; do run_stage "$s"; done
echo "Pipeline finished: $(date '+%F %T')"
