#!/bin/sh

SIMDIR=/media/adam/Data/PhD/Data/upright/heins-ral25/simulations
NUMCHECK="3"

set -x  # echo commands

# ./process_sim_runs.py "$SIMDIR/robust_h60_2025-01-20_18-28-33" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/robust_h50_2025-01-21_07-55-47" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/robust_h40_2025-01-21_12-08-58" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/robust_h30_2025-01-21_15-45-50" --check-constraints "$NUMCHECK"
#
# ./process_sim_runs.py "$SIMDIR/center_h60_2025-01-20_21-25-56" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/center_h50_2025-01-21_10-48-32" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/center_h40_2025-01-21_14-41-13" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/center_h30_2025-01-21_21-27-16" --check-constraints "$NUMCHECK"

# ./process_sim_runs.py "$SIMDIR/top_h60_2025-01-20_19-43-56" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/top_h50_2025-01-21_09-04-33" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/top_h40_2025-01-21_13-25-35" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/top_h30_2025-01-21_19-47-05" --check-constraints "$NUMCHECK"


### SQP 20 ###

# ./process_sim_runs.py "$SIMDIR/sqp20/center_h30_2025-01-31_06-07-32" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/sqp20/center_h40_2025-01-31_05-04-35" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/sqp20/center_h50_2025-01-31_04-01-36" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/sqp20/center_h60_2025-01-31_02-59-55" --check-constraints "$NUMCHECK"
#
# ./process_sim_runs.py "$SIMDIR/sqp20/top_h30_2025-01-31_01-50-32" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/sqp20/top_h40_2025-01-31_00-42-41" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/sqp20/top_h50_2025-01-30_23-35-46" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/sqp20/top_h60_2025-01-30_22-29-15" --check-constraints "$NUMCHECK"

### SQP 10 ###

# ./process_sim_runs.py "$SIMDIR/sqp10/robust_h30_2025-01-31_20-08-51" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/sqp10/robust_h40_2025-01-31_18-59-46" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/sqp10/robust_h50_2025-01-31_17-51-16" --check-constraints "$NUMCHECK"
# ./process_sim_runs.py "$SIMDIR/sqp10/robust_h60_2025-01-30_10-52-16" --check-constraints "$NUMCHECK"

./process_sim_runs.py "$SIMDIR/sqp10/center_h30_2025-01-30_21-06-37" --check-constraints "$NUMCHECK"
./process_sim_runs.py "$SIMDIR/sqp10/center_h40_2025-01-30_20-03-33" --check-constraints "$NUMCHECK"
./process_sim_runs.py "$SIMDIR/sqp10/center_h50_2025-01-30_19-00-58" --check-constraints "$NUMCHECK"
./process_sim_runs.py "$SIMDIR/sqp10/center_h60_2025-01-30_17-59-23" --check-constraints "$NUMCHECK"

./process_sim_runs.py "$SIMDIR/sqp10/top_h30_2025-01-30_16-50-24" --check-constraints "$NUMCHECK"
./process_sim_runs.py "$SIMDIR/sqp10/top_h40_2025-01-30_15-43-15" --check-constraints "$NUMCHECK"
./process_sim_runs.py "$SIMDIR/sqp10/top_h50_2025-01-30_14-36-24" --check-constraints "$NUMCHECK"
./process_sim_runs.py "$SIMDIR/sqp10/top_h60_2025-01-30_13-29-51" --check-constraints "$NUMCHECK"
