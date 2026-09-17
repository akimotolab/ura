#!/usr/bin/env bash
# Runs run_linear_coupling.py once per (dim_y, trial) pair, in parallel.
# Each invocation runs exactly one trial with no internal parallelism.
#
# To run only some methods, set METHODS to a comma-separated subset of:
#   URA-GD, URA-RMSprop, URA-L-BFGS, URA-CMA-ES, CMA-ES (black-box), L-BFGS (white-box)
# e.g. METHODS="URA-GD,URA-CMA-ES" ./run_linear_coupling.sh
#
# To ablate URA-L-BFGS's warm-starting / early-stopping, set ABLATION to one
# of: full (default), no-es, no-ws, no-es-no-ws. Each non-"full" mode writes
# its own distinctly-labeled output (e.g. "URA-L-BFGS (no-es)"), so it never
# collides with a normal run or another mode -- run once per mode to compare:
# e.g. ABLATION=no-es ./run_linear_coupling.sh
#
# To zero out C_mat (the matrix multiplying x in y - Cx), decoupling the
# lower-level optimum from x, set ZERO_COUPLING=1. This writes
# "qdecoupled_cond1"/"qdecoupled_cond1e4" instead of "qcoupling_cond1"/
# "qcoupling_cond1e4", so running once with and once without this gives 4
# total benchmark function types (coupled/decoupled x the two condition
# numbers) without collision:
# e.g. ZERO_COUPLING=1 ./run_linear_coupling.sh
set -euo pipefail
cd "$(dirname "$0")"

DIM_Y_LIST=(10 30 100 300 1000)
N_TRIALS=20
MAX_JOBS="${MAX_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"
export METHODS="${METHODS:-}"
export ABLATION="${ABLATION:-full}"
export ZERO_COUPLING_FLAG=""
if [ "${ZERO_COUPLING:-0}" = "1" ]; then
    ZERO_COUPLING_FLAG="--zero-coupling"
fi

for dim_y in "${DIM_Y_LIST[@]}"; do
    for ((trial = 0; trial < N_TRIALS; trial++)); do
        echo "$dim_y $trial"
    done
done | xargs -P "$MAX_JOBS" -L 1 sh -c \
    'python run_linear_coupling.py --dim-y "$1" --trial "$2" ${METHODS:+--methods "$METHODS"} --ablation "$ABLATION" $ZERO_COUPLING_FLAG' _

echo "all trials complete"
