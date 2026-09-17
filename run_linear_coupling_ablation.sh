#!/usr/bin/env bash
# Runs run_linear_coupling.py once per (dim_y, trial) pair, in parallel, for
# each of the 3 ablated URA-L-BFGS variants (no-es, no-ws, no-es-no-ws) --
# see run_linear_coupling.py's --ablation. The normal ("full") URA-L-BFGS and
# the other methods (URA-GD, URA-RMSprop, URA-CMA-ES, CMA-ES (black-box),
# L-BFGS (white-box)) are unaffected by these modes, so this script never runs
# them -- use run_linear_coupling.sh for those.
#
# Each mode writes its own distinctly-labeled output (e.g. "URA-L-BFGS
# (no-es)"), so running all 3 here never collides with each other or with a
# normal run_linear_coupling.sh run.
set -euo pipefail
cd "$(dirname "$0")"

DIM_Y_LIST=(10 30 100 300 1000)
N_TRIALS=20
MAX_JOBS="${MAX_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"
ABLATED_MODES=(no-es no-ws no-es-no-ws)

for mode in "${ABLATED_MODES[@]}"; do
    label=$(python -c "
from run_linear_coupling import ABLATION_LBFGS_LABEL
print(ABLATION_LBFGS_LABEL['$mode'])
")
    echo "=== ablation=$mode (methods=$label) ==="
    export ABLATION="$mode"
    export METHODS="$label"

    for dim_y in "${DIM_Y_LIST[@]}"; do
        for ((trial = 0; trial < N_TRIALS; trial++)); do
            echo "$dim_y $trial"
        done
    done | xargs -P "$MAX_JOBS" -L 1 sh -c \
        'python run_linear_coupling.py --dim-y "$1" --trial "$2" --methods "$METHODS" --ablation "$ABLATION"' _
done

echo "all ablation trials complete"
