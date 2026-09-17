"""Regenerate convergence plots from saved CSV data in results_data/."""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scienceplots  # noqa: F401

plt.style.use(["science", "bright", "grid"])
# plt.rcParams["text.latex.preamble"] += r"\usepackage[T1]{fontenc}"

DIM_X = 10
DIM_Y_LIST = [10, 30, 100, 300, 1000]
DATA_DIR = "results_data"

plot_order = ["qcoupling_cond1", "qcoupling_cond1e4"]
# run_linear_coupling.py --zero-coupling writes these under a "qdecoupled_*"
# prefix instead of "qcoupling_*" (see its --zero-coupling help), so they
# never collide with the coupled data above.
plot_order_decoupled = ["qdecoupled_cond1", "qdecoupled_cond1e4"]

titles = {
    "qcoupling_cond1":   r"$\kappa(A)=1$",
    "qcoupling_cond1e4": r"$\kappa(A)=10^4$",
    # Same titles as the coupled case -- condition number of A has the same
    # meaning whether or not C_mat is zeroed.
    "qdecoupled_cond1":   r"$\kappa(A)=1$",
    "qdecoupled_cond1e4": r"$\kappa(A)=10^4$",
}
colors = {
    "URA-GD":                     "C0",
    "URA-RMSprop":                "C1",
    "URA-L-BFGS":                 "C2",
    "URA-CMA-ES":                 "C3",
    "CMA-ES (black-box)":         "C4",
    "L-BFGS (white-box)":         "C5",
    "URA-L-BFGS (no-es)":         "C6",
    "URA-L-BFGS (no-ws)":         "C7",
    "URA-L-BFGS (no-es-no-ws)":   "C8",
}

SAFE_TO_LABEL = {
    "URA-GD":                     "URA-GD",
    "URA-RMSprop":                "URA-RMSprop",
    "URA-L-BFGS":                 "URA-L-BFGS",
    "URA-CMA-ES":                 "URA-CMA-ES",
    "CMA-ES_black-box":           "CMA-ES (black-box)",
    "L-BFGS_white-box":           "L-BFGS (white-box)",
    "URA-L-BFGS_no-es":           "URA-L-BFGS (no-es)",
    "URA-L-BFGS_no-ws":           "URA-L-BFGS (no-ws)",
    "URA-L-BFGS_no-es-no-ws":     "URA-L-BFGS (no-es-no-ws)",
}

# Methods produced only by an opt-in run (run_linear_coupling.py
# --ablation no-es/no-ws/no-es-no-ws). Missing files for these never block
# plotting the rest.
OPTIONAL_METHODS = {
    "URA-L-BFGS (no-es)", "URA-L-BFGS (no-ws)", "URA-L-BFGS (no-es-no-ws)",
}

# Methods that call a gradient oracle; their grad-call counts get an extra
# dashed overlay on the f-calls plot. Black-box methods (URA-CMA-ES,
# CMA-ES (black-box)) never call df_lower, so they're excluded.
GRAD_METHODS = {
    "URA-GD", "URA-RMSprop", "URA-L-BFGS", "L-BFGS (white-box)",
} | OPTIONAL_METHODS


def aggregate_trials(
    traces_list: list,
    x_idx: int,
    log_x: bool = False,
    n_points: int = 500,
) -> tuple:
    """Interpolate each trial onto a common x-grid; return median and IQR.

    traces_list : list of (iters, total_calls, fvals, upper_calls, lower_calls)

    Trials that terminate early are included: np.interp with right=fvals[-1]
    holds the last recorded value constant for x beyond their termination point.
    Returns (x_grid, median, q25, q75).
    """
    all_xs = [t[x_idx] for t in traces_list]
    all_fvals = [np.maximum(t[2], 1e-6) for t in traces_list]

    if log_x:
        all_xs = [xs + 1 for xs in all_xs]

    x_min = min(xs[0] for xs in all_xs)
    x_max = max(xs[-1] for xs in all_xs)

    if log_x:
        x_min = max(x_min, 1)
        x_grid = np.logspace(np.log10(x_min), np.log10(x_max), n_points)
    else:
        x_grid = np.linspace(x_min, x_max, n_points)

    interp = np.array([
        np.interp(x_grid, xs, fvals, left=fvals[0], right=fvals[-1])
        for xs, fvals in zip(all_xs, all_fvals)
    ])

    return (
        x_grid,
        np.median(interp, axis=0),
        np.percentile(interp, 25, axis=0),
        np.percentile(interp, 75, axis=0),
    )


def make_plot(
    trial_results: dict,
    dim_x: int,
    dim_y: int,
    x_idx: int,
    xlabel: str,
    filename: str,
    log_x: bool = False,
    grad_idx: int | None = None,
    plot_order: list = plot_order,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    handles, labels = None, None
    for ax, prob_label in zip(axes.flatten(), plot_order):
        for method in colors:
            traces_list = trial_results.get(prob_label, {}).get(method, [])
            if not traces_list:
                continue
            x_grid, median, q25, q75 = aggregate_trials(traces_list, x_idx, log_x)
            color = colors[method]
            ax.plot(x_grid, median, label=method, color=color)
            ax.fill_between(x_grid, q25, q75, color=color, alpha=0.2, linewidth=0)
            if grad_idx is not None and method in GRAD_METHODS:
                g_grid, g_median, _, _ = aggregate_trials(traces_list, grad_idx, log_x)
                ax.plot(g_grid, g_median, color=color, linestyle="--")
        ax.axhline(1e-6, color="grey", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.set_title(titles[prob_label])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Objective value")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    # fig.suptitle(
    #     r"$f(x,y)=\tfrac{1}{2}\|x\|^2+\tfrac{1}{2}(y-Cx)^\top A(y-Cx)$,"
    #     rf" $n_x={dim_x}$, $n_y={dim_y}$",
    #     fontsize=9,
    # )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=(len(labels) + 1) // 2,
        framealpha=0.7,
        bbox_to_anchor=(0.5, 0),
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {filename}")


def generate_plots(dim_y: int, order: list, prefix: str) -> None:
    """Build trial_results for `order`'s problems and render its two plots
    (iters, fcalls) exactly as make_plot always has, just under `prefix`."""
    trial_results: dict[str, dict[str, list]] = {}
    missing = []

    for prob_label in order:
        trial_results[prob_label] = {}
        for safe_method, label in SAFE_TO_LABEL.items():
            # Current runs write one file per trial (run_linear_coupling.sh
            # launches one process per trial); fall back to the older
            # single combined-file layout if no per-trial files exist.
            pattern = os.path.join(
                DATA_DIR, f"x{DIM_X}_y{dim_y}_{prob_label}_{safe_method}_trial*.csv"
            )
            paths = sorted(glob.glob(pattern))
            if not paths:
                legacy_path = os.path.join(
                    DATA_DIR, f"x{DIM_X}_y{dim_y}_{prob_label}_{safe_method}.csv"
                )
                if os.path.exists(legacy_path):
                    paths = [legacy_path]
            if not paths:
                if label not in OPTIONAL_METHODS:
                    missing.append(pattern)
                continue

            df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)

            # Group by trial; each group becomes one entry in traces_list.
            # CSVs from single-trial runs have no "trial" column — treat as trial 0.
            if "trial" not in df.columns:
                df["trial"] = 0

            # Older CSVs predate grad_calls; treat missing as no gradient calls.
            if "grad_calls" not in df.columns:
                df["grad_calls"] = 0

            traces_list = []
            for _, grp in df.groupby("trial", sort=True):
                traces_list.append((
                    grp["iteration"].to_numpy(),
                    grp["total_calls"].to_numpy(),
                    grp["f_upper_min"].to_numpy(),
                    grp["upper_calls"].to_numpy(),
                    grp["lower_calls"].to_numpy(),
                    grp["grad_calls"].to_numpy(),
                ))
            trial_results[prob_label][label] = traces_list

    if missing:
        print(f"DIM_Y={dim_y} ({prefix}): skipping — missing files:")
        for p in missing:
            print(f"  {p}")
        return

    make_plot(trial_results, DIM_X, dim_y, 0, "Iterations",
              f"{prefix}_iters_x{DIM_X}_y{dim_y}.pdf", plot_order=order)
    make_plot(trial_results, DIM_X, dim_y, 1, r"Total $f$-calls",
              f"{prefix}_fcalls_x{DIM_X}_y{dim_y}.pdf", log_x=True, grad_idx=5,
              plot_order=order)


for DIM_Y in DIM_Y_LIST:
    generate_plots(DIM_Y, plot_order, "qcoupling")
    generate_plots(DIM_Y, plot_order_decoupled, "qdecoupled")
