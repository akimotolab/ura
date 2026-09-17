"""Plot success rate and total function evaluations vs $d_y$ for the linear
coupling experiment.

Reads per-trial results from a `run_linear_coupling.py` output directory
(`results_data/` by default) and plots, per problem condition, the success
rate and the number of total function evaluations (upper + lower calls)
required to reach f_upper_min <= threshold. Runs that did not converge are
marked with an open marker at the budget cap.

Usage:
    uv run python plot_vary_dy.py [results_data_dir] [--threshold 1e-6] [--out plot.pdf]
        [--budget-cap 1000000]

--budget-cap must match the f-call budget the sweep was actually run with
(run_linear_coupling.py uses max_f_calls=1e6 for the bilevel/CMA-ES methods);
it is not read from the result files.
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from run_linear_coupling import ABLATION_LBFGS_LABEL, ABLATION_MODES


plt.style.use(["science", "grid", "bright"])

THRESHOLD_DEFAULT = 1e-6
BUDGET_CAP_DEFAULT = 1_000_000
DIM_X = 10
DIM_Y_LIST = [10, 30, 100, 300, 1000]
DATA_DIR_DEFAULT = "results_data"

titles = {
    "qcoupling_cond1": r"$\kappa(A)=1$",
    "qcoupling_cond1e4": r"$\kappa(A)=10^4$",
    # Same titles for the decoupled (--zero-coupling) case -- condition
    # number of A has the same meaning whether or not C_mat is zeroed.
    "qdecoupled_cond1": r"$\kappa(A)=1$",
    "qdecoupled_cond1e4": r"$\kappa(A)=10^4$",
}

# Marker/color per method label, keyed the same way regardless of --ablation
# (the URA-L-BFGS *slot* always uses these; build_method_style below swaps
# in whichever ablation-mode label actually goes in that slot).
_METHOD_MARKER_COLOR = {
    "URA-GD": dict(color="C0", marker="v"),
    "URA-RMSprop": dict(color="C1", marker="o"),
    "URA-L-BFGS": dict(color="C2", marker="^"),
    "URA-CMA-ES": dict(color="C3", marker="s"),
    "CMA-ES (black-box)": dict(color="C4", marker="D"),
    "L-BFGS (white-box)": dict(color="C5", marker="P"),
}

# Methods whose "calls to converge" also get a dotted gradient-calls overlay
# in the eval panel (see plot()). Restricted to the two L-BFGS-family solvers.
GRAD_OVERLAY_METHODS = {"URA-L-BFGS", "L-BFGS (white-box)"}


def safe_name(label: str) -> str:
    """Mirror run_linear_coupling.save_csv's filename sanitization."""
    return label.replace(" ", "_").replace("(", "").replace(")", "")


def build_plot_order(zero_coupling: bool) -> list[str]:
    prefix = "qdecoupled" if zero_coupling else "qcoupling"
    return [f"{prefix}_cond1", f"{prefix}_cond1e4"]


def build_method_style(ablation: str) -> dict:
    """METHOD_STYLE, with the URA-L-BFGS slot's label/safe-name swapped to
    whichever ablation-mode variant is selected (see run_linear_coupling.py's
    --ablation / ABLATION_LBFGS_LABEL). marker/color stay tied to the "slot"
    (URA-L-BFGS's marker/color), not the specific label, so switching
    --ablation doesn't change how the URA-L-BFGS series looks.
    """
    style = {}
    for label, mc in _METHOD_MARKER_COLOR.items():
        actual_label = ABLATION_LBFGS_LABEL[ablation] if label == "URA-L-BFGS" else label
        style[actual_label] = dict(**mc, safe=safe_name(actual_label))
    return style


def load_results(
    base_dir: Path,
    threshold: float,
    budget_cap: float,
    plot_order: list[str],
    method_style: dict,
) -> pd.DataFrame:
    records = []
    for dim_y in DIM_Y_LIST:
        for prob_label in plot_order:
            for label, style in method_style.items():
                safe_method = style["safe"]
                # Current runs write one file per trial (run_linear_coupling.sh
                # launches one process per trial); fall back to the older
                # single combined-file layout if no per-trial files exist.
                pattern = str(
                    base_dir / f"x{DIM_X}_y{dim_y}_{prob_label}_{safe_method}_trial*.csv"
                )
                paths = sorted(Path(p) for p in glob.glob(pattern))
                if not paths:
                    legacy_path = base_dir / f"x{DIM_X}_y{dim_y}_{prob_label}_{safe_method}.csv"
                    if legacy_path.exists():
                        paths = [legacy_path]
                if not paths:
                    continue

                df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
                if "trial" not in df.columns:
                    df["trial"] = 0
                # Older CSVs predate grad_calls; treat missing as no gradient calls.
                if "grad_calls" not in df.columns:
                    df["grad_calls"] = 0

                for trial, grp in df.groupby("trial", sort=True):
                    grp = grp.sort_values("iteration")
                    f_final = grp["f_upper_min"].iloc[-1]
                    total_calls = grp["total_calls"].iloc[-1]
                    grad_calls = grp["grad_calls"].iloc[-1]
                    converged = f_final <= threshold

                    records.append(
                        dict(
                            dim_y=dim_y,
                            prob_label=prob_label,
                            method=label,
                            trial=trial,
                            f_final=f_final,
                            total_calls=total_calls,
                            grad_calls=grad_calls,
                            converged=converged,
                            # Use budget cap for failed runs (for plotting)
                            total_calls_plot=total_calls if converged else budget_cap,
                            grad_calls_plot=grad_calls if converged else budget_cap,
                        )
                    )

    return pd.DataFrame(records)


def plot(
    df: pd.DataFrame,
    threshold: float,
    out_path: str,
    budget_cap: float,
    plot_order: list[str],
    method_style: dict,
    grad_overlay_labels: set[str],
) -> None:
    methods = [m for m in method_style if m in df["method"].unique()]
    dy_values = sorted(df["dim_y"].unique())

    # Match plot_vary_k.py's per-panel size: figsize=(8, 5.5) for its 3 columns.
    fig, axes = plt.subplots(
        2, len(plot_order), figsize=(8 * len(plot_order) / 3, 5.5), sharey="row"
    )

    for col, prob_label in enumerate(plot_order):
        ax_rate = axes[0, col]
        ax_eval = axes[1, col]
        sub = df[df["prob_label"] == prob_label]

        for method in methods:
            style = method_style[method]
            m_sub = sub[sub["method"] == method]
            if m_sub.empty:
                continue

            agg = (
                m_sub.groupby("dim_y")["total_calls_plot"]
                .agg(
                    median="median",
                    q25=lambda x: x.quantile(0.25),
                    q75=lambda x: x.quantile(0.75),
                )
                .reindex(dy_values)
            )
            conv = (
                m_sub.groupby("dim_y")["converged"]
                .agg(n_converged="sum", n_total="count")
                .reindex(dy_values)
            )

            dys = agg.index.values
            median = agg["median"].values
            q25 = agg["q25"].values
            q75 = agg["q75"].values
            success_rate = (conv["n_converged"] / conv["n_total"]).values
            all_converged = (conv["n_converged"] == conv["n_total"]).values
            any_failed = ~all_converged

            ax_rate.plot(
                dys,
                success_rate,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                label=method,
                markersize=8,
            )

            ax_eval.semilogy(
                dys,
                median,
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                label=method,
                markersize=8,
            )
            ax_eval.fill_between(dys, q25, q75, color=style["color"], alpha=0.15)
            # Open markers where not all seeds converged
            if any_failed.any():
                ax_eval.semilogy(
                    dys[any_failed],
                    median[any_failed],
                    marker=style["marker"],
                    color=style["color"],
                    linestyle="none",
                    markersize=8,
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                )

            # Dotted overlay: gradient calls (instead of total f-calls) to
            # converge, for the L-BFGS-family solvers only. Same color, no
            # markers/band, no separate legend entry.
            if method in grad_overlay_labels:
                grad_agg = (
                    m_sub.groupby("dim_y")["grad_calls_plot"]
                    .median()
                    .reindex(dy_values)
                    .values
                )
                ax_eval.semilogy(
                    dys,
                    grad_agg,
                    color=style["color"],
                    linestyle=":",
                    linewidth=1.8,
                )

        ax_rate.set_title(titles.get(prob_label, prob_label))
        ax_rate.set_xscale("log")
        ax_rate.set_ylim(-0.05, 1.05)
        ax_rate.set_xticks(dy_values)
        ax_rate.set_xticklabels([str(v) for v in dy_values])
        ax_rate.tick_params(axis="x", which="minor", bottom=False)
        ax_rate.grid(True, which="both", alpha=0.3)

        ax_eval.set_xlabel(r"Lower-level dimension $d_y$")
        ax_eval.set_xscale("log")
        ax_eval.set_xticks(dy_values)
        ax_eval.set_xticklabels([str(v) for v in dy_values])
        ax_eval.tick_params(axis="x", which="minor", bottom=False)
        ax_eval.axhline(budget_cap, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_eval.grid(True, which="both", alpha=0.3)

    axes[0, 0].set_ylabel("Success rate")
    axes[1, 0].set_ylabel("Total function evaluations")

    # Custom legend: filled marker + unfilled marker per method, no line
    legend_handles = []
    legend_labels = []
    for method in methods:
        style = method_style[method]
        filled = Line2D(
            [0],
            [0],
            linestyle="none",
            marker=style["marker"],
            color=style["color"],
            markerfacecolor=style["color"],
            markersize=8,
        )
        unfilled = Line2D(
            [0],
            [0],
            linestyle="none",
            marker=style["marker"],
            color=style["color"],
            markerfacecolor="white",
            markeredgewidth=1.5,
            markersize=8,
        )
        legend_handles.append((filled, unfilled))
        legend_labels.append(method)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.legend(
        legend_handles,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot success rate / total FEs vs d_y for the linear coupling experiment"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=DATA_DIR_DEFAULT,
        help=f"Path to linear-coupling results directory (default: {DATA_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_DEFAULT,
        help=f"Convergence threshold (default: {THRESHOLD_DEFAULT})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output file (default: plot_vary_dy.pdf, with '_decoupled' and/or "
            "'_<ablation>' appended when --zero-coupling / --ablation are set)"
        ),
    )
    parser.add_argument(
        "--budget-cap",
        type=float,
        default=BUDGET_CAP_DEFAULT,
        help=(
            "Function evaluation budget the sweep was run with (default: "
            f"{BUDGET_CAP_DEFAULT:.0f}). Must match max_f_calls used by run_linear_coupling.py; "
            "non-converged runs are plotted at this value."
        ),
    )
    parser.add_argument(
        "--zero-coupling",
        action="store_true",
        help=(
            "Plot the decoupled benchmark (run_linear_coupling.py --zero-coupling's "
            "'qdecoupled_cond1'/'qdecoupled_cond1e4' output) instead of the normal "
            "coupled one ('qcoupling_cond1'/'qcoupling_cond1e4')."
        ),
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=ABLATION_MODES,
        help=(
            "Which URA-L-BFGS variant to plot in the URA-L-BFGS slot (default: "
            "full, i.e. normal URA-L-BFGS); see run_linear_coupling.py --ablation."
        ),
    )
    args = parser.parse_args()

    plot_order = build_plot_order(args.zero_coupling)
    method_style = build_method_style(args.ablation)
    grad_overlay_labels = {
        ABLATION_LBFGS_LABEL[args.ablation] if m == "URA-L-BFGS" else m
        for m in GRAD_OVERLAY_METHODS
    }

    out_path = args.out
    if out_path is None:
        out_path = "plot_vary_dy"
        if args.zero_coupling:
            out_path += "_decoupled"
        if args.ablation != "full":
            out_path += f"_{args.ablation}"
        out_path += ".pdf"

    base_dir = Path(args.results_dir)
    df = load_results(base_dir, args.threshold, args.budget_cap, plot_order, method_style)

    if df.empty:
        print("No results found.")
        return

    print(f"Loaded {len(df)} trial records.")
    print(
        df.groupby(["method", "dim_y", "prob_label"])["converged"]
        .agg(["sum", "count"])
        .to_string()
    )

    plot(
        df,
        args.threshold,
        out_path,
        args.budget_cap,
        plot_order,
        method_style,
        grad_overlay_labels,
    )


if __name__ == "__main__":
    main()
