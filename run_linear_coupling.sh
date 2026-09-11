#!/usr/bin/env bash
# Runs run_linear_coupling.py once per (dim_y, trial) pair, in parallel.
# Each invocation runs exactly one trial with no internal parallelism.
#
# To run only some methods, set METHODS to a comma-separated subset of:
#   URA-GD, URA-RMSprop, URA-L-BFGS, URA-CMA-ES, CMA-ES (black-box), L-BFGS (white-box)
# e.g. METHODS="URA-GD,URA-CMA-ES" ./run_linear_coupling.sh
set -euo pipefail
cd "$(dirname "$0")"

DIM_Y_LIST=(10 30 100 300 1000)
N_TRIALS=20
MAX_JOBS="${MAX_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"
export METHODS="${METHODS:-}"

for dim_y in "${DIM_Y_LIST[@]}"; do
    for ((trial = 0; trial < N_TRIALS; trial++)); do
        echo "$dim_y $trial"
    done
done | xargs -P "$MAX_JOBS" -L 1 sh -c \
    'python run_linear_coupling.py --dim-y "$1" --trial "$2" ${METHODS:+--methods "$METHODS"}' _

echo "all trials complete"
