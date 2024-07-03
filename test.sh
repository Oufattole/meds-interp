#!/usr/bin/env bash
# bash test.sh
set -e

meds-tab-describe MEDS_cohort_dir="/tmp/pytest-of-leander/pytest-23/test_knn_tuning0"

meds-tab-tabularize-time-series --multirun \
   worker="range(0,$N_PARALLEL_WORKERS)" \
   hydra/launcher=joblib \
   MEDS_cohort_dir="/tmp/pytest-of-leander/pytest-23/test_knn_tuning0" \
   tabularization.min_code_inclusion_frequency=10 \
   do_overwrite=False \
   tabularization.window_sizes="(1d 30d 365d full)" \
   tabularization.aggs="[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"

meds-tab-cache-task MEDS_cohort_dir="/tmp/pytest-of-leander/pytest-23/test_knn_tuning0" \
   task_name="$TASK" \
   tabularization.min_code_inclusion_frequency=10 \
   do_overwrite=False \
   tabularization.window_sizes="(1d 30d 365d full)" \
   tabularization.aggs="[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"

meds-tab-xgboost --multirun \
   MEDS_cohort_dir="/tmp/pytest-of-leander/pytest-23/test_knn_tuning0" \
   task_name="$TASK" \
   output_dir="" \
   tabularization.min_code_inclusion_frequency=10 \
   tabularization.window_sizes="$(generate-permutations [1d, 30d, 365d, full])" \
   do_overwrite=False \
   "tabularization.aggs=$(generate-permutations [value/count,value/sum,value/sum_sqd,value/min,value/max])"
