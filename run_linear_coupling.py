import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilevel_cma import BilevelCMAGrad, BilevelCMALBFGS, MyDdCma, mirror
from scipy.optimize import minimize as scipy_minimize

import scienceplots  # noqa: F401

# plt.style.use(["science", "retro", "grid"])
# plt.rcParams["text.latex.preamble"] += r"\usepackage[T1]{fontenc}"

DIM_X = 10
DIM_Y_LIST = [10, 30, 100, 300, 1000]
N_TRIALS = 20
N_WORKERS = None
BOUND = 5.0

GAP_TOL = 1e-6
VXMIN = 1e-8
CXMAX = 1e6

DATA_DIR = "results_data"
os.makedirs(DATA_DIR, exist_ok=True)


def make_quadratic_coupling(
    condition: float,
    dim_y: int,
    C_mat: np.ndarray,
    R_inner: np.ndarray,
):
    """
    f(x, y) = 1/2 ||x||^2 + 1/2 (y - Cx)^T A (y - Cx)
    where A = R^T diag(w) R with w logspaced from 1 to condition.

    Inner Hessian w.r.t. y = A, condition number = condition.
    Lower-level optimum: y*(x) = Cx.
    After substitution: f(x, y*(x)) = 1/2 ||x||^2, minimised at x*=0, y*=0.
    """
    w = np.logspace(0, np.log10(condition), dim_y)

    def _Av(r):
        return R_inner.T @ (w * (R_inner @ r))

    def f_upper(x, y):
        r = y - C_mat @ x
        return 0.5 * np.dot(x, x) + 0.5 * np.dot(r, _Av(r))

    def f_batch(Z):
        """Vectorized f_upper for a population Z of shape (lam, N)."""
        X = Z[:, :DIM_X]
        R = Z[:, DIM_X:] - X @ C_mat.T
        AR = ((R @ R_inner.T) * w) @ R_inner
        return 0.5 * np.sum(X**2, axis=1) + 0.5 * np.sum(R * AR, axis=1)

    def df_lower(x, y):
        return _Av(y - C_mat @ x)

    def df_full(x, y):
        Ar = _Av(y - C_mat @ x)
        return np.concatenate([x - C_mat.T @ Ar, Ar])

    return f_upper, f_batch, df_lower, df_full


configs = [
    {"label": "URA-GD", "cls": BilevelCMAGrad, "extra": {"use_rmsprop": False}},
    {"label": "URA-RMSprop", "cls": BilevelCMAGrad, "extra": {"use_rmsprop": True}},
    {"label": "URA-L-BFGS", "cls": BilevelCMALBFGS, "extra": {}, "cmax": None},
]

titles = {
    "qcoupling_cond1": r"$\kappa(A)=1$",
    "qcoupling_cond1e4": r"$\kappa(A)=10^4$",
}
colors = {
    "URA-GD": "C0",
    "URA-RMSprop": "C1",
    "URA-L-BFGS": "C2",
    "CMA-ES (black-box)": "C3",
    "BFGS (white-box)": "C4",
}

plot_order = ["qcoupling_cond1", "qcoupling_cond1e4"]


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

    # np.interp: for x beyond the last recorded point, hold the final value.
    # Since fvals are the running best they are already monotone non-increasing,
    # so linear interpolation between recorded points is exact.
    interp = np.array(
        [
            np.interp(x_grid, xs, fvals, left=fvals[0], right=fvals[-1])
            for xs, fvals in zip(all_xs, all_fvals)
        ]
    )

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
) -> None:
    """Plot median convergence curves with IQR shading across trials."""
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
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.set_title(titles[prob_label])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Objective value")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(
        r"$f(x,y)=\tfrac{1}{2}\|x\|^2+\tfrac{1}{2}(y-Cx)^\top A(y-Cx)$,"
        rf" $n_x={dim_x}$, $n_y={dim_y}$",
        fontsize=9,
    )
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


def save_csv(trial_results: dict, dim_x: int, dim_y: int) -> None:
    for prob_label, method_dict in trial_results.items():
        for method, traces_list in method_dict.items():
            dfs = []
            for trial_idx, traces in enumerate(traces_list):
                iters, total_calls, fvals, upper_calls, lower_calls = traces
                dfs.append(
                    pd.DataFrame(
                        {
                            "trial": trial_idx,
                            "iteration": iters,
                            "upper_calls": upper_calls,
                            "lower_calls": lower_calls,
                            "total_calls": total_calls,
                            "f_upper_min": fvals,
                        }
                    )
                )
            safe_method = method.replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(
                DATA_DIR, f"x{dim_x}_y{dim_y}_{prob_label}_{safe_method}.csv"
            )
            pd.concat(dfs, ignore_index=True).to_csv(path, index=False)
            print(f"saved {path}")


def run_single_trial(
    trial_idx: int,
    dim_y: int,
    C_mat: np.ndarray,
    R_inner: np.ndarray,
) -> dict:
    """Run all methods for one trial. Returns {prob_label: {method: traces}}."""
    n = DIM_X + dim_y
    xb = np.array([[-BOUND] * DIM_X, [BOUND] * DIM_X])
    yb = np.array([[-BOUND] * dim_y, [BOUND] * dim_y])
    full_bounds = np.array([[-BOUND] * n, [BOUND] * n])

    problems = [
        ("qcoupling_cond1", *make_quadratic_coupling(1.0, dim_y, C_mat, R_inner)),
        ("qcoupling_cond1e4", *make_quadratic_coupling(1e4, dim_y, C_mat, R_inner)),
    ]

    method_labels = [cfg["label"] for cfg in configs] + [
        "CMA-ES (black-box)",
        "BFGS (white-box)",
    ]
    trial_traces: dict[str, dict[str, tuple]] = {pl: {} for pl, *_ in problems}

    for prob_label, f_upper, _, df_lower, __ in problems:
        for cfg in configs:
            common_kwargs = dict(
                f_upper=f_upper,
                f_lower=None,
                dim_x=DIM_X,
                dim_y=dim_y,
                xb=xb,
                yb=yb,
                df_lower=df_lower,
                single_objective=True,
                variance_update_x=True,
                true_upper_min=0.0,
                random_seed=trial_idx,
                cmax=8,
                max_iter=10**18,
                max_f_calls=10**6,
                n_omega=1,
                restart=False,
                gap_window_size_x=100,
            )
            kwargs = {**common_kwargs, **cfg["extra"]}
            if "cmax" in cfg:
                kwargs["cmax"] = cfg["cmax"]
            solver = cfg["cls"](**kwargs)
            x0 = mirror(solver.x_mean, xb[0], xb[1])
            f0 = solver.f_upper(x0, solver.y[0])

            _, _, _, _, _, _, log_df = solver.optimize()

            iters = np.arange(len(log_df) + 1)
            upper_calls = np.concatenate([[0], log_df["upper_call_count"].to_numpy()])
            lower_calls = np.concatenate([[0], log_df["lower_call_count"].to_numpy()])
            total_calls = upper_calls + lower_calls
            fvals = np.concatenate([[f0], log_df["f_upper_min"].to_numpy()])
            trial_traces[prob_label][cfg["label"]] = (
                iters,
                total_calls,
                fvals,
                upper_calls,
                lower_calls,
            )

    for prob_label, f_upper_prob, f_batch_prob, _, __ in problems:
        rng_cma = np.random.default_rng(trial_idx)
        f_calls = 0
        f_best = np.inf
        z_best = None
        step = 0

        mean0 = rng_cma.random(n) * (2 * BOUND) - BOUND
        D0 = np.full(n, BOUND / 2)
        cma = MyDdCma(
            mean0,
            D0,
            flg_variance_update=True,
            beta_eig=10 * n**2,
            random_seed=trial_idx,
        )

        f_init_cma = float(f_batch_prob(mean0.reshape(1, -1))[0])
        calls_trace: list[int] = [0]
        fcalls_trace: list[int] = [0]
        fval_trace: list[float] = [f_init_cma]

        while f_calls < 10**6:
            arx, ary, arz = cma.sample()
            arx_m = mirror(arx, full_bounds[0], full_bounds[1])
            fvals_cma = f_batch_prob(arx_m)
            f_calls += len(fvals_cma)

            idx_b = int(np.argmin(fvals_cma))
            if fvals_cma[idx_b] < f_best:
                f_best = float(fvals_cma[idx_b])
                z_best = arx_m[idx_b].copy()

            cma.update(np.argsort(fvals_cma), arx, ary, arz)

            log_det_D = np.sum(np.log(cma.D))
            cma.D /= np.exp(log_det_D / len(cma.D))
            cma.sigma *= np.exp(log_det_D / len(cma.D))
            cma.D = np.minimum(cma.D, (2 * BOUND) / 4 / cma.sigma)

            step += 1
            calls_trace.append(step)
            fcalls_trace.append(f_calls)
            fval_trace.append(f_best)

            if f_best < GAP_TOL:
                break
            if cma.S.max() / cma.S.min() > CXMAX:
                break
            if cma.coordinate_std.max() < VXMIN:
                break

        iters_arr = np.array(calls_trace)
        fcalls_arr = np.array(fcalls_trace)
        fval_arr = np.array(fval_trace)
        trial_traces[prob_label]["CMA-ES (black-box)"] = (
            iters_arr,
            fcalls_arr,
            fval_arr,
            np.zeros_like(iters_arr),
            fcalls_arr,
        )

    for prob_label, f_upper_prob, _, __, df_full_prob in problems:

        def _make_bfgs_fns(fu, dfu):
            def f(z):
                return fu(z[:DIM_X], z[DIM_X:])

            def g(z):
                return dfu(z[:DIM_X], z[DIM_X:])

            return f, g

        f_bfgs, g_bfgs = _make_bfgs_fns(f_upper_prob, df_full_prob)
        z_init = np.random.default_rng(trial_idx).uniform(-BOUND, BOUND, size=n)

        fcall_counter = [0]

        def f_counted(z):
            fcall_counter[0] += 1
            return f_bfgs(z)

        iter_trace: list[int] = [0]
        fcalls_trace_bfgs: list[int] = [0]
        fval_trace_bfgs: list[float] = [f_bfgs(z_init)]
        step_counter = [0]

        def callback(xk):
            step_counter[0] += 1
            fval_trace_bfgs.append(f_counted(xk))
            iter_trace.append(step_counter[0])
            fcalls_trace_bfgs.append(fcall_counter[0])

        scipy_minimize(
            f_counted,
            z_init,
            jac=g_bfgs,
            method="BFGS",
            callback=callback,
            options={"gtol": 1e-10, "maxiter": 10**6},
        )

        iters_arr = np.array(iter_trace)
        fcalls_arr = np.array(fcalls_trace_bfgs)
        fval_arr = np.array(fval_trace_bfgs)
        trial_traces[prob_label]["BFGS (white-box)"] = (
            iters_arr,
            fcalls_arr,
            fval_arr,
            fcalls_arr,
            np.zeros_like(iters_arr),
        )

    print(f"  [trial {trial_idx:2d} | dim_y={dim_y}] done", flush=True)
    return trial_traces


if __name__ == "__main__":
    for DIM_Y in DIM_Y_LIST:
        print(f"\n{'#' * 60}")
        print(f"# DIM_Y = {DIM_Y}  (N = {DIM_X + DIM_Y})  —  {N_TRIALS} trials")
        print(f"{'#' * 60}")

        # Fixed randomness: objective function geometry
        rng = np.random.default_rng(42)
        _Q_C, _ = np.linalg.qr(rng.standard_normal((DIM_Y, DIM_X)))
        C_mat = _Q_C

        _Q_R, _R_R = np.linalg.qr(rng.standard_normal((DIM_Y, DIM_Y)))
        R_inner = _Q_R * np.sign(np.diag(_R_R))

        method_labels = [cfg["label"] for cfg in configs] + [
            "CMA-ES (black-box)",
            "BFGS (white-box)",
        ]
        prob_labels = ["qcoupling_cond1", "qcoupling_cond1e4"]
        trial_results: dict[str, dict[str, list]] = {
            pl: {ml: [] for ml in method_labels} for pl in prob_labels
        }

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {
                executor.submit(run_single_trial, t, DIM_Y, C_mat, R_inner): t
                for t in range(N_TRIALS)
            }
            for future in as_completed(futures):
                trial_traces = future.result()
                for prob_label, method_dict in trial_traces.items():
                    for method, traces in method_dict.items():
                        trial_results[prob_label][method].append(traces)

        save_csv(trial_results, DIM_X, DIM_Y)
        make_plot(
            trial_results,
            DIM_X,
            DIM_Y,
            0,
            "Iterations",
            f"qcoupling_iters_x{DIM_X}_y{DIM_Y}.pdf",
        )
        make_plot(
            trial_results,
            DIM_X,
            DIM_Y,
            1,
            r"Total $f$-calls + 1",
            f"qcoupling_fcalls_x{DIM_X}_y{DIM_Y}.pdf",
            log_x=True,
        )
