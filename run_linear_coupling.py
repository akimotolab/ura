import argparse
import inspect
import os

import numpy as np
import pandas as pd
import torch
from bilevel_cma import BilevelCMA, BilevelCMAGrad, BilevelCMALBFGS, MyDdCma, mirror

DIM_X = 10
DIM_Y_LIST = [10, 30, 100, 300, 1000]
N_TRIALS = 20
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
    {"label": "URA-CMA-ES", "cls": BilevelCMA, "extra": {}},
]


def save_csv(trial_traces: dict, dim_x: int, dim_y: int, trial_idx: int) -> None:
    """Save one trial's traces. Filename carries the trial index so that
    parallel invocations (one per trial, launched from the shell script)
    never write the same path."""
    for prob_label, method_dict in trial_traces.items():
        for method, traces in method_dict.items():
            iters, total_calls, fvals, upper_calls, lower_calls = traces
            df = pd.DataFrame(
                {
                    "trial": trial_idx,
                    "iteration": iters,
                    "upper_calls": upper_calls,
                    "lower_calls": lower_calls,
                    "total_calls": total_calls,
                    "f_upper_min": fvals,
                }
            )
            safe_method = method.replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(
                DATA_DIR,
                f"x{dim_x}_y{dim_y}_{prob_label}_{safe_method}_trial{trial_idx:02d}.csv",
            )
            df.to_csv(path, index=False)
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
        "L-BFGS (white-box)",
    ]
    trial_traces: dict[str, dict[str, tuple]] = {pl: {} for pl, *_ in problems}

    for prob_label, f_upper, _, df_lower, __ in problems:
        for cfg in configs:
            # BilevelCMA (fully black-box) has no single_objective flag; passing
            # f_lower=f_upper directly reproduces the same "minimize, don't play
            # minimax" semantics that single_objective=True gives the other solvers.
            common_kwargs = dict(
                f_upper=f_upper,
                f_lower=f_upper if cfg["cls"] is BilevelCMA else None,
                dim_x=DIM_X,
                dim_y=dim_y,
                xb=xb,
                yb=yb,
                df_lower=df_lower,
                single_objective=True,
                variance_update_x=True,
                true_upper_min=0.0,
                random_seed=trial_idx,
                cmax=20,
                max_iter=10**18,
                max_f_calls=10**6,
                n_omega=1,
                restart=False,
                gap_window_size_x=60,
            )
            kwargs = {**common_kwargs, **cfg["extra"]}
            if "cmax" in cfg:
                kwargs["cmax"] = cfg["cmax"]
            # Drop kwargs the target class's constructor doesn't accept (e.g.
            # BilevelCMA has no df_lower/single_objective).
            accepted = set(inspect.signature(cfg["cls"].__init__).parameters)
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
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
            beta_eig=10 * n,
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
            # ddcma.py only advances .t inside onestep(), which we bypass here;
            # advance it ourselves so the lazy-eigendecomposition schedule (teig,
            # set from beta_eig at construction) actually engages.
            cma.t += 1

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
        z_init = np.random.default_rng(trial_idx).uniform(-BOUND, BOUND, size=n)

        param = torch.nn.Parameter(torch.tensor(z_init.copy(), dtype=torch.float64))
        # Same hyperparameters as URA-L-BFGS's inner solver (see BilevelCMALBFGS
        # defaults) so the full-space baseline is a fair comparison.
        optimizer = torch.optim.LBFGS(
            [param],
            lr=1.0,
            history_size=10,
            line_search_fn="strong_wolfe",
            max_iter=1,
        )

        fcall_counter = [0]
        last_f = [float("inf")]
        last_g = [None]

        def closure():
            optimizer.zero_grad()
            z_np = param.detach().numpy().copy()
            f_val = float(f_upper_prob(z_np[:DIM_X], z_np[DIM_X:]))
            g_val = df_full_prob(z_np[:DIM_X], z_np[DIM_X:])
            param.grad = torch.tensor(g_val.copy(), dtype=torch.float64)
            fcall_counter[0] += 1
            last_f[0] = f_val
            last_g[0] = g_val
            return torch.tensor(f_val, dtype=torch.float64)

        iter_trace: list[int] = [0]
        fcalls_trace_bfgs: list[int] = [0]
        fval_trace_bfgs: list[float] = [f_upper_prob(z_init[:DIM_X], z_init[DIM_X:])]

        for step_idx in range(1, 10**4 + 1):
            optimizer.step(closure)
            iter_trace.append(step_idx)
            fcalls_trace_bfgs.append(fcall_counter[0])
            fval_trace_bfgs.append(last_f[0])
            # Same target as the CMA-ES baseline; strong_wolfe line search can
            # stall near machine precision without ever satisfying a tight gtol.
            if last_f[0] < GAP_TOL or np.max(np.abs(last_g[0])) < 1e-10:
                break

        iters_arr = np.array(iter_trace)
        fcalls_arr = np.array(fcalls_trace_bfgs)
        fval_arr = np.array(fval_trace_bfgs)
        trial_traces[prob_label]["L-BFGS (white-box)"] = (
            iters_arr,
            fcalls_arr,
            fval_arr,
            fcalls_arr,
            np.zeros_like(iters_arr),
        )

    print(f"  [trial {trial_idx:2d} | dim_y={dim_y}] done", flush=True)
    return trial_traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one trial (all methods) for one dim_y. "
        "Launch multiple copies (see run_linear_coupling.sh) to parallelize "
        "across trials/dim_y instead of parallelizing inside this script."
    )
    parser.add_argument("--dim-y", type=int, required=True, choices=DIM_Y_LIST)
    parser.add_argument("--trial", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DIM_Y = args.dim_y
    trial_idx = args.trial

    print(f"[dim_y={DIM_Y} trial={trial_idx}] starting")

    # Fixed randomness: objective function geometry (depends only on DIM_Y,
    # so every trial/process for a given DIM_Y reconstructs the same problem).
    rng = np.random.default_rng(42)
    _Q_C, _ = np.linalg.qr(rng.standard_normal((DIM_Y, DIM_X)))
    C_mat = _Q_C

    _Q_R, _R_R = np.linalg.qr(rng.standard_normal((DIM_Y, DIM_Y)))
    R_inner = _Q_R * np.sign(np.diag(_R_R))

    trial_traces = run_single_trial(trial_idx, DIM_Y, C_mat, R_inner)
    save_csv(trial_traces, DIM_X, DIM_Y, trial_idx)
