from typing import Callable, Optional, Union

import numpy as np
from pandas import DataFrame
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import kendalltau

from ddcma import DdCma


class MyDdCma(DdCma):
    def set_dynamic_parameters(
        self, init_matrix: np.ndarray, init_S: np.ndarray, init_sigma
    ) -> None:
        self.Z = np.array(init_matrix[0, :, :], copy=True)
        self.C = np.array(init_matrix[1, :, :], copy=True)
        self.B = np.array(init_matrix[2, :, :], copy=True)
        self.sqrtC = np.array(init_matrix[3, :, :], copy=True)
        self.invsqrtC = np.array(init_matrix[4, :, :], copy=True)
        self.S = np.array(init_S[:], copy=True)
        self.sigma = init_sigma


def mirror(z: np.ndarray, lbound: np.ndarray, ubound: np.ndarray) -> np.ndarray:
    """Mirroring constraint handling

    Parameters
    ----------
    z : np.ndarray (1D)
        solution vector to be mirrored. S
    lbound, ubound : np.ndarray (1D)
        lower and upper bound of the box constraint

    Returns
    -------
    np.ndarray (1D) : mirrored solution
    """

    width = ubound - lbound
    return ubound - np.abs(np.mod(z - lbound, 2 * width) - width)


class BilevelCMA:
    def __init__(
        self,
        # Objective function settings
        f_upper: Callable[[np.ndarray, np.ndarray], float],
        f_lower: Optional[Callable[[np.ndarray, np.ndarray], float]],
        dim_x: int,
        dim_y: int,
        xb: np.ndarray,
        yb: np.ndarray,
        is_x_bounded: bool = True,
        is_y_bounded: bool = True,
        random_seed: int = 42,
        # True upper-level fn
        true_upper_min: float = 0.0,
        # Population sizes
        lambda_x: Optional[int] = None,
        lambda_y: Optional[int] = None,
        # Outer solver settings
        Vxmin: float = 1e-8,
        Cxmax: float = 1e7,
        # Inner solver settings
        Vymin: float = 1e-4,
        Cymax: float = 1e7,
        Tmin: int = 10,
        # WRA settings
        cmax: int = 1,
        tau_thr: float = 0.7,
        pn: float = 0.05,
        pp: float = 0.4,
        p_thr: float = 0.1,
        n_omega: Optional[int] = None,
        # Termination conditions
        gap_tolerance_x: float = 1e-6,
        gap_tolerance_y: float = 1e-6,
        gap_window_size_x: int = 60,
        gap_window_size_y: int = 20,
        max_f_calls: int = 10000000,
        max_iter: int = 100000000000000000000,
        max_iter_lower: int = 50,
        restart: bool = True,
        # CMA variance update
        variance_update_x: bool = False,
        variance_update_y: bool = False,
        # For ablation study
        reset_all: bool = False,
    ) -> None:
        self.f_upper = f_upper
        self.f_lower = self.f_lower_minmax if f_lower is None else f_lower

        self.dim_x = dim_x
        self.dim_y = dim_y

        self.xb = xb
        self.yb = yb

        self.is_x_bounded = is_x_bounded
        self.is_y_bounded = is_y_bounded

        self.lambda_x = (
            lambda_x if lambda_x is not None else 4 + int(3 * np.log(self.dim_x))
        )
        self.lambda_y = (
            lambda_y if lambda_y is not None else 4 + int(3 * np.log(self.dim_y))
        )
        self.n_omega = n_omega if n_omega is not None else 3 * self.lambda_x

        self.reset_all = reset_all
        self.true_upper_min = true_upper_min
        self.variance_update_x = variance_update_x
        self.variance_update_y = variance_update_y

        # Outer solver settings
        self.Vxmin = Vxmin
        self.Cxmax = Cxmax

        # Inner solver settings
        self.cmax = cmax
        self.Vymin = Vymin
        self.Cymax = Cymax
        self.Tmin = Tmin

        self.tau_thr = tau_thr

        self.p = np.ones(self.n_omega)
        self.pp = pp
        self.pn = pn
        self.p_thr = p_thr
        self.rng = np.random.default_rng(random_seed)

        # Initialize upper solver parameters
        self.x_mean = (
            self.rng.random(size=self.dim_x) * (self.xb[1, :] - self.xb[0, :])
            + self.xb[0, :]
        )
        self.x = np.repeat(self.x_mean[np.newaxis], self.lambda_x, axis=0)
        self.D_x = (self.xb[1, :] - self.xb[0, :]) / 4
        self.upper_cma = self._init_upper_cma()

        # Initialize lower solver parameters
        (
            self.y,
            self.y_mean,
            self.D_y,
            self.init_matrix,
            self.init_S,
            self.init_sigma,
        ) = self._init_bulk_lower_parameters()

        # For use in URA
        self.f_upper_min = np.empty(self.lambda_x, dtype=float)
        self.f_lower_min = np.empty(self.lambda_x, dtype=float)
        self.k_min = np.empty(self.lambda_x, dtype=int)

        self.f_upper_calls = 0
        self.f_lower_calls = 0

        self.current_cma_idx = 0

        # Termination conditions
        self.gap_tolerance_x = gap_tolerance_x
        self.gap_tolerance_y = gap_tolerance_y
        self.gap_window_size_x = gap_window_size_x
        self.gap_window_size_y = gap_window_size_y
        self.max_f_calls = max_f_calls
        self.max_iter = max_iter
        self.max_iter_lower = max_iter_lower
        self.restart = restart

    def f_lower_minmax(self, x: np.ndarray, y: np.ndarray) -> float:
        return -1.0 * self.f_upper(x, y)

    def _init_bulk_lower_parameters(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        delta_y = self.yb[1] - self.yb[0]
        y_mean = self.rng.random((self.n_omega, self.dim_y)) * delta_y + self.yb[0]
        D_y = np.repeat((delta_y / 4)[np.newaxis], self.n_omega, axis=0)
        yz = self.rng.standard_normal((self.n_omega, self.dim_y))
        y = np.clip(y_mean + yz * D_y, self.yb[0], self.yb[1])

        # Inner solver dynamic parameters
        template = np.zeros((5, self.dim_y, self.dim_y))
        template[1:] = np.eye(self.dim_y)  # Set indices 1-4 to identity matrices
        init_matrix = np.repeat(template[np.newaxis], self.n_omega, axis=0)
        init_S = np.ones((self.n_omega, self.dim_y))
        init_sigma = np.ones(self.n_omega)

        return y, y_mean, D_y, init_matrix, init_S, init_sigma

    def _get_init_dp(self):
        """Initialize dynamic parameters for the inner CMA-ES configuration."""
        init_matrix = np.zeros((5, self.dim_y, self.dim_y))  # Z is initialized as zeros
        init_matrix[1:] = np.tile(
            np.eye(self.dim_y), (4, 1, 1)
        )  # C, B, sqrtC, invsqrtC
        return (
            init_matrix,
            np.ones(self.dim_y),
            1.0,
        )  # (init_matrix, init_S, init_sigma)

    def _reset_y(self, j: int) -> None:
        # Initialize dynamic parameters
        self.init_matrix[j], self.init_S[j], self.init_sigma[j] = self._get_init_dp()
        self.y_mean[j] = self.rng.uniform(
            low=self.yb[0], high=self.yb[1], size=self.dim_y
        )
        self.D_y[j] = (self.yb[1] - self.yb[0]) / 4
        self.y[j] = np.clip(
            self.y_mean[j] + self.D_y[j] * self.rng.standard_normal(self.dim_y),
            self.yb[0],
            self.yb[1],
        )
        self.p[j] = 1.0

    def _init_upper_cma(self) -> MyDdCma:
        return MyDdCma(
            self.x_mean,
            self.D_x,
            self.lambda_x,
            flg_variance_update=self.variance_update_x,
            beta_eig=10 * self.dim_x**2,
            random_seed=self.rng.integers(999999),
        )

    def _reset_solvers(self) -> None:
        (
            self.y,
            self.y_mean,
            self.D_y,
            self.init_matrix,
            self.init_S,
            self.init_sigma,
        ) = self._init_bulk_lower_parameters()
        self.upper_cma = self._init_upper_cma()

    def ura(self):
        fx_arr = np.array([[self.f_lower(x, y) for y in self.y] for x in self.x])
        self.f_lower_calls += self.lambda_x * self.n_omega
        self.k_min: np.ndarray = np.argmin(fx_arr, axis=1)  # (lambda_x,)

        prev_f_lower_min = np.min(fx_arr, axis=1)
        prev_f_upper_min = np.array(
            [self.f_upper(x, y) for x, y in zip(self.x, self.y[self.k_min])]
        )
        self.f_upper_calls += self.lambda_x

        self.f_lower_min = prev_f_lower_min.copy()
        self.f_upper_min = prev_f_upper_min.copy()

        # *_tilde array shapes are (self.lambda_x, ...)
        # since self.k_min is an array, the advanced indexing makes copies of the data
        y_tilde = self.y[self.k_min]
        y_mean_tilde = self.y_mean[self.k_min]
        D_y_tilde = self.D_y[self.k_min]
        init_matrix_tilde = self.init_matrix[self.k_min]
        init_S_tilde = self.init_S[self.k_min]
        init_sigma_tilde = self.init_sigma[self.k_min]

        cma_instances = [
            MyDdCma(
                y_mean_tilde[i],
                D_y_tilde[i],
                self.lambda_y,
                flg_variance_update=self.variance_update_y,
                beta_eig=10 * self.dim_y**2,
                random_seed=self.rng.integers(999999),
            )
            for i in range(self.lambda_x)
        ]
        for i in range(self.lambda_x):
            cma_instances[i].set_dynamic_parameters(
                init_matrix_tilde[i], init_S_tilde[i], init_sigma_tilde[i]
            )

        # Early stopping strategy
        h = np.ones(self.lambda_x, dtype=bool)
        update_n = np.zeros(self.lambda_x)
        cma_iter_count = np.zeros(self.lambda_x)

        tau = -1.0

        while tau <= self.tau_thr:
            if not h.any():
                break
            for i in range(self.lambda_x):
                if not h[i]:
                    continue
                lower_cma = cma_instances[i]
                orig_D_y = lower_cma.D.copy()
                orig_matrix = np.stack(
                    [
                        lower_cma.Z,
                        lower_cma.C,
                        lower_cma.B,
                        lower_cma.sqrtC,
                        lower_cma.invsqrtC,
                    ]
                )  # np.stack makes a copy
                orig_S = lower_cma.S.copy()
                orig_sigma = lower_cma.sigma

                current_f_lower_min = prev_f_lower_min[i]
                current_f_upper_min = prev_f_upper_min[i]

                update_n[i] += 1
                c = 0
                self.current_cma_idx = i

                hist = []

                while cma_iter_count[i] < self.max_iter_lower and c < self.cmax:
                    urax, uray, uraz = lower_cma.sample()
                    mirrored_y = (
                        mirror(urax, self.yb[0], self.yb[1])
                        if self.is_y_bounded
                        else urax.copy()
                    )
                    fy = np.array([self.f_lower(self.x[i], y) for y in mirrored_y])
                    self.f_lower_calls += lower_cma.lam

                    min_fy = np.min(fy)
                    hist.append(min_fy)
                    if len(hist) > self.gap_window_size_y:
                        gap = np.ptp(hist[-self.gap_window_size_y :])
                        if gap < self.gap_tolerance_y:
                            break

                    if min_fy <= current_f_lower_min:
                        yid = np.argmin(fy)
                        y_tilde[i] = mirrored_y[yid]
                        current_f_lower_min = min_fy
                        c += 1

                    lower_cma.update(np.argsort(fy), urax, uray, uraz)
                    cma_iter_count[i] += 1

                    # numerical stability
                    log_det_D = np.sum(np.log(lower_cma.D))
                    lower_cma.D /= np.exp(log_det_D / len(lower_cma.D))
                    lower_cma.sigma *= np.exp(log_det_D / len(lower_cma.D))

                    if lower_cma.S.max() / lower_cma.S.min() > self.Cymax:
                        h[i] = False
                        lower_cma.set_dynamic_parameters(
                            orig_matrix, orig_S, orig_sigma
                        )
                        lower_cma.D = orig_D_y
                        break

                    # Update coordinate-wise std with vectorized operations
                    lower_cma.D = np.minimum(
                        lower_cma.D, (self.yb[1] - self.yb[0]) / 4 / lower_cma.sigma
                    )

                    if (
                        lower_cma.coordinate_std.max() < self.Vymin
                        and cma_iter_count[i] >= self.Tmin
                    ):
                        lower_cma.D = np.maximum(
                            lower_cma.D, self.Vymin / lower_cma.sigma
                        )
                        h[i] = False
                        break

                if cma_iter_count[i] >= self.max_iter_lower:
                    h[i] = False

                if c >= 1:
                    current_f_upper_min = self.f_upper(self.x[i], mirrored_y[yid])
                    self.f_upper_calls += 1

                self.f_lower_min[i] = current_f_lower_min
                self.f_upper_min[i] = current_f_upper_min

            # Compute tau wrt the upper function
            tau, _ = kendalltau(self.f_upper_min, prev_f_upper_min)
            prev_f_lower_min = self.f_lower_min.copy()
            prev_f_upper_min = self.f_upper_min.copy()

        # Update parameters for all configurations
        for i in range(self.lambda_x):
            lower_cma = cma_instances[i]
            y_mean_tilde[i] = lower_cma.xmean
            D_y_tilde[i] = lower_cma.D
            init_matrix_tilde[i] = np.stack(
                [
                    lower_cma.Z,
                    lower_cma.C,
                    lower_cma.B,
                    lower_cma.sqrtC,
                    lower_cma.invsqrtC,
                ]
            )
            init_S_tilde[i] = lower_cma.S
            init_sigma_tilde[i] = lower_cma.sigma

        k_min_set = np.unique(self.k_min)
        for idx in k_min_set:
            self.p[idx] = np.clip(self.p[idx] + self.pp, 0, 1)
            mask = self.k_min == idx
            f_upper_min_subset = self.f_upper_min[mask]
            min_subset_idx = np.argmin(f_upper_min_subset)
            true_idx = np.where(mask)[0][min_subset_idx]

            self.y[idx] = y_tilde[true_idx]
            self.y_mean[idx] = y_mean_tilde[true_idx]
            self.init_matrix[idx] = init_matrix_tilde[true_idx]
            self.init_S[idx] = init_S_tilde[true_idx]
            self.init_sigma[idx] = init_sigma_tilde[true_idx]
            self.D_y[idx] = D_y_tilde[true_idx]

        non_min_idxs = np.setdiff1d(np.arange(self.n_omega), k_min_set)
        self.p[non_min_idxs] -= self.pn
        reset_mask = self.p[non_min_idxs] <= self.p_thr
        for k in non_min_idxs[reset_mask]:
            self._reset_y(k)

        if self.reset_all:
            for k in range(self.n_omega):
                self._reset_y(k)

        rankings = np.argsort(self.f_upper_min)
        return rankings, y_tilde

    @property
    def total_f_calls(self) -> int:
        return self.f_upper_calls + self.f_lower_calls

    def optimize(
        self,
        on_step: Optional[Callable[[int, float, int], None]] = None,
    ) -> tuple[float, float, int, int, np.ndarray, np.ndarray, DataFrame]:
        """Run the bilevel optimization."""
        f_upper_hist = []
        f_lower_hist = []
        conv_flag = 0

        log_entries: list[dict[str, Union[int, float]]] = []

        # Store x and y values before each restart
        nominal_x: list[np.ndarray] = []
        nominal_y: list[np.ndarray] = []

        for _ in range(self.max_iter):
            arx, ary, arz = self.upper_cma.sample()
            self.x = mirror(arx, self.xb[0], self.xb[1]) if self.is_x_bounded else arx

            rankings, y_tilde = self.ura()

            self.upper_cma.update(rankings, arx, ary, arz)

            log_det_D = np.sum(np.log(self.upper_cma.D))
            self.upper_cma.D /= np.exp(log_det_D / len(self.upper_cma.D))
            self.upper_cma.sigma *= np.exp(log_det_D / len(self.upper_cma.D))

            self.upper_cma.D = np.minimum(
                self.upper_cma.D, (self.xb[1] - self.xb[0]) / 4 / self.upper_cma.sigma
            )

            f_upper_hist.append(np.min(self.f_upper_min))
            f_lower_hist.append(np.min(self.f_lower_min))

            mirrored_x_mean = (
                mirror(self.upper_cma.xmean, self.xb[0], self.xb[1])
                if self.is_x_bounded
                else self.upper_cma.xmean
            )

            entry = {
                "lower_call_count": self.f_lower_calls,
                "upper_call_count": self.f_upper_calls,
                "sigma_x": self.upper_cma.sigma,
                "f_lower_min": self.f_lower_min.min(),
                "f_upper_min": self.f_upper_min.min(),
                **{f"mean_{i}": mirrored_x_mean[i] for i in range(self.dim_x)},
                **{
                    f"std_{i}": self.upper_cma.coordinate_std[i]
                    for i in range(self.dim_x)
                },
                **{f"eigen_{i}": self.upper_cma.S[i] for i in range(self.dim_x)},
                **{
                    f"f_lower_min_{i}": self.f_lower_min[i]
                    for i in range(self.lambda_x)
                },
                **{
                    f"f_upper_min_{i}": self.f_upper_min[i]
                    for i in range(self.lambda_x)
                },
            }
            log_entries.append(entry)

            if on_step is not None:
                on_step(len(log_entries), self.f_upper_min.min(), self.total_f_calls)

            if (
                np.abs(self.f_upper_min.min() - self.true_upper_min)
                < self.gap_tolerance_x
            ):
                conv_flag = 1
            if len(f_upper_hist) > self.gap_window_size_x:
                gap = np.ptp(f_upper_hist[-self.gap_window_size_x :])
                if gap <= self.gap_tolerance_x:
                    conv_flag = 2  # gap tolerance
            if self.upper_cma.S.max() / self.upper_cma.S.min() > self.Cxmax:
                conv_flag = 3
            if self.upper_cma.coordinate_std.max() < self.Vxmin:
                conv_flag = 4  # vxmin
            if self.total_f_calls >= self.max_f_calls:
                conv_flag = 5

            x_best = self.x[self.f_upper_min.argmin()]
            y_best = y_tilde[self.f_upper_min.argmin()]

            x, y = self.x.copy(), self.y.copy()

            if conv_flag:
                flag_names = {1: "success", 2: "stagnation", 3: "ill-conditioned", 4: "collapsed", 5: "budget"}
                print(f"  [stop] conv_flag={conv_flag} ({flag_names[conv_flag]}), f={self.f_upper_min.min():.3e}, iter={len(log_entries)}")
                nominal_x.extend(x)
                nominal_y.extend(y)
                self._reset_solvers()
                if self.restart and conv_flag not in (1, 5, 6):
                    conv_flag = 0
                else:
                    break

        nominal_x.extend(x)
        nominal_y.extend(y)

        columns = [
            "lower_call_count",
            "upper_call_count",
            "sigma_x",
            "f_lower_min",
            "f_upper_min",
        ]
        columns += [f"mean_{i}" for i in range(self.dim_x)]
        columns += [f"std_{i}" for i in range(self.dim_x)]
        columns += [f"eigen_{i}" for i in range(self.dim_x)]
        columns += [f"f_lower_min_{i}" for i in range(self.lambda_x)]
        columns += [f"f_upper_min_{i}" for i in range(self.lambda_x)]

        log_df = DataFrame(log_entries, columns=columns)

        f_lower_arr = np.array(
            [[self.f_lower(x, y) for y in nominal_y] for x in nominal_x]
        )
        self.f_lower_calls += len(nominal_x) * len(nominal_y)
        idx_y_arr = f_lower_arr.argmin(axis=1)
        f_upper_arr = np.array(
            [self.f_upper(x, nominal_y[k]) for x, k in zip(nominal_x, idx_y_arr)]
        )
        self.f_upper_calls += len(nominal_x)

        idx_x = f_upper_arr.argmin()
        idx_y = idx_y_arr[idx_x]
        x_min, y_min = nominal_x[idx_x], nominal_y[idx_y]

        f_upper_min: float = f_upper_arr[idx_x]
        f_lower_min: float = self.f_lower(x_min, y_min)
        self.f_lower_calls += 1

        f_upper_best: float = self.f_upper(x_best, y_best)
        self.f_upper_calls += 1
        f_lower_best: float = self.f_lower(x_best, y_best)
        self.f_lower_calls += 1

        if np.abs(f_upper_best - self.true_upper_min) < np.abs(
            f_upper_min - self.true_upper_min
        ):
            f_upper_min = f_upper_best
            f_lower_min = f_lower_best
            x_min, y_min = x_best, y_best

        return (
            f_upper_min,
            f_lower_min,
            self.f_upper_calls,
            self.f_lower_calls,
            x_min,
            y_min,
            log_df,
        )


class BilevelCMAGrad:
    """Bilevel optimizer with CMA-ES over x and adaptive gradient descent over y.

    The lower-level optimizer uses df_lower directly: it takes an adaptive step
    (grow alpha on success, backtrack on overshoot) with an optional RMSProp
    preconditioner. Each pool slot warmstarts its y position and step size
    alpha across outer iterations.

    Parameters
    ----------
    f_upper : Callable
        Upper-level objective f_upper(x, y).
    f_lower : Callable or None
        Lower-level objective minimized over y. None defaults to -f_upper.
    dim_x, dim_y : int
        Dimensions of x and y.
    xb, yb : ndarray, shape (2, dim)
        Box bounds: row 0 lower, row 1 upper.
    df_lower : Callable
        Gradient of f_lower w.r.t. y.
    beta_lower : float
        Step-size factor: alpha /= beta on success, *= beta on overshoot.
    Umin_lower : float
        Backtracking stops when max|alpha * d| <= Umin_lower.
    use_rmsprop : bool
        Precondition gradient with per-dimension RMSProp.
    rmsprop_beta, rmsprop_eps : float
        RMSProp decay and stability constant.
    single_objective : bool
        f_lower is overridden to f_upper and calls are counted jointly.
    lambda_x : int or None
        Upper CMA population size. Default 4 + floor(3 ln dim_x).
    n_omega : int or None
        Pool size. Default 3 * lambda_x.
    Vxmin, Cxmax : float
        Upper CMA stopping thresholds (coordinate std, condition number).
    cmax : int
        Max successful gradient steps per x[i] per tau-loop pass.
    tau_thr : float
        Kendall tau ranking-stability threshold.
    pn, pp, p_thr : float
        Pool probability decay, growth, and reset threshold.
    gap_tolerance_x, gap_window_size_x : float, int
        Upper-level gap stopping criterion.
    max_f_calls, max_iter, max_iter_lower : int
        Budget limits.
    restart : bool
        Restart on convergence unless exact optimum or budget exhausted.
    variance_update_x : bool
        Full covariance adaptation in upper CMA-ES.
    reset_all : bool
        Reset all pool slots each outer iteration (ablation).
    true_upper_min : float
        Known optimum value for early stopping.
    random_seed : int
    """

    @property
    def total_f_calls(self) -> int:
        return self.f_upper_calls + self.f_lower_calls

    def __init__(
        self,
        f_upper: Callable[[np.ndarray, np.ndarray], float],
        f_lower: Optional[Callable[[np.ndarray, np.ndarray], float]],
        dim_x: int,
        dim_y: int,
        xb: np.ndarray,
        yb: np.ndarray,
        df_lower: Callable[[np.ndarray, np.ndarray], np.ndarray],
        beta_lower: float = 0.5,
        Umin_lower: float = 1e-5,
        use_rmsprop: bool = False,
        rmsprop_beta: float = 0.9,
        rmsprop_eps: float = 1e-8,
        single_objective: bool = False,
        is_x_bounded: bool = True,
        is_y_bounded: bool = True,
        lambda_x: Optional[int] = None,
        n_omega: Optional[int] = None,
        Vxmin: float = 1e-8,
        Cxmax: float = 1e7,
        cmax: int = 1,
        tau_thr: float = 0.7,
        pn: float = 0.05,
        pp: float = 0.4,
        p_thr: float = 0.1,
        gap_tolerance_x: float = 1e-6,
        gap_window_size_x: int = 60,
        max_f_calls: int = 10000000,
        max_iter: int = 100000000000000000000,
        max_iter_lower: int = 50,
        restart: bool = True,
        variance_update_x: bool = False,
        reset_all: bool = False,
        true_upper_min: float = 0.0,
        random_seed: int = 42,
    ) -> None:
        self.f_upper_calls = 0
        self.f_lower_calls = 0

        self.f_upper = f_upper
        self.f_lower = (lambda x, y: -f_upper(x, y)) if f_lower is None else f_lower
        self.single_objective = single_objective
        if single_objective:
            self.f_lower = self.f_upper

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.xb = xb
        self.yb = yb
        self.is_x_bounded = is_x_bounded
        self.is_y_bounded = is_y_bounded

        self.df_lower = df_lower
        self.beta_lower = beta_lower
        self.Umin_lower = Umin_lower
        self.use_rmsprop = use_rmsprop
        self.rmsprop_beta = rmsprop_beta
        self.rmsprop_eps = rmsprop_eps

        self.lambda_x = lambda_x if lambda_x is not None else 4 + int(3 * np.log(dim_x))
        self.n_omega = n_omega if n_omega is not None else 3 * self.lambda_x

        self.Vxmin = Vxmin
        self.Cxmax = Cxmax
        self.cmax = cmax
        self.tau_thr = tau_thr
        self.pp = pp
        self.pn = pn
        self.p_thr = p_thr
        self.gap_tolerance_x = gap_tolerance_x
        self.gap_window_size_x = gap_window_size_x
        self.max_f_calls = max_f_calls
        self.max_iter = max_iter
        self.max_iter_lower = max_iter_lower
        self.restart = restart
        self.variance_update_x = variance_update_x
        self.reset_all = reset_all
        self.true_upper_min = true_upper_min

        self.rng = np.random.default_rng(random_seed)

        self.x_mean = self.rng.random(size=dim_x) * (xb[1] - xb[0]) + xb[0]
        self.D_x = (xb[1] - xb[0]) / 4
        self.x = np.repeat(self.x_mean[np.newaxis], self.lambda_x, axis=0)
        self.upper_cma = self._make_upper_cma()

        self.p = np.ones(self.n_omega)
        self.y, self.alpha_y, self.v_y = self._init_pool()

        self.f_upper_min = np.empty(self.lambda_x, dtype=float)
        self.f_lower_min = np.empty(self.lambda_x, dtype=float)
        self.k_min = np.empty(self.lambda_x, dtype=int)

    def _make_upper_cma(self) -> MyDdCma:
        return MyDdCma(
            self.x_mean,
            self.D_x,
            self.lambda_x,
            flg_variance_update=self.variance_update_x,
            beta_eig=10 * self.dim_x**2,
            random_seed=self.rng.integers(999999),
        )

    def _init_pool(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta_y = self.yb[1] - self.yb[0]
        y = self.rng.random((self.n_omega, self.dim_y)) * delta_y + self.yb[0]
        alpha = np.ones(self.n_omega) * float(np.mean(delta_y)) / 2
        v = np.ones((self.n_omega, self.dim_y))
        return y, alpha, v

    def _reset_y(self, j: int) -> None:
        delta_y = self.yb[1] - self.yb[0]
        self.y[j] = self.rng.random(self.dim_y) * delta_y + self.yb[0]
        self.alpha_y[j] = float(np.mean(delta_y)) / 2
        self.v_y[j] = np.ones(self.dim_y)
        self.p[j] = 1.0

    def _reset_solvers(self) -> None:
        self.y, self.alpha_y, self.v_y = self._init_pool()
        self.upper_cma = self._make_upper_cma()

    def _Lsearch_grad(
        self,
        x_i: np.ndarray,
        y_curr: np.ndarray,
        alpha: float,
        v: np.ndarray,
        fold: float,
    ) -> tuple[float, np.ndarray, np.ndarray, float, int]:
        """Single adaptive gradient step minimizing f_lower(x_i, y)."""
        fcalls = 0
        g = self.df_lower(x_i, y_curr)

        if self.use_rmsprop:
            v = self.rmsprop_beta * v + (1 - self.rmsprop_beta) * g**2
            d = g / (np.sqrt(v) + self.rmsprop_eps)
        else:
            d = g

        y_prop = y_curr - alpha * d
        bcount = 0
        if self.is_y_bounded:
            bcount = int(np.sum((y_prop >= self.yb[1]) | (y_prop <= self.yb[0])))
        y_next = (
            np.clip(y_prop, self.yb[0], self.yb[1])
            if self.is_y_bounded
            else y_prop.copy()
        )

        fnew = self.f_lower(x_i, y_next)
        fcalls += 1

        if fnew < fold and bcount != self.dim_y:
            alpha = alpha / self.beta_lower
        elif fnew > fold:
            while fnew > fold:
                alpha = self.beta_lower * alpha
                y_prop = y_curr - alpha * d
                y_next = (
                    np.clip(y_prop, self.yb[0], self.yb[1])
                    if self.is_y_bounded
                    else y_prop.copy()
                )
                fnew = self.f_lower(x_i, y_next)
                fcalls += 1
                if np.max(np.abs(alpha * d)) <= self.Umin_lower:
                    break

        return alpha, v, y_next, fnew, fcalls

    def ura(self) -> tuple[np.ndarray, np.ndarray]:
        """Upper-level ranking assessment using gradient inner solver."""
        fx_arr = np.array([[self.f_lower(x, y) for y in self.y] for x in self.x])
        self.f_lower_calls += self.lambda_x * self.n_omega
        self.k_min = np.argmin(fx_arr, axis=1)

        Fold = np.min(fx_arr, axis=1)
        if self.single_objective:
            prev_f_upper_min = fx_arr[np.arange(self.lambda_x), self.k_min]
        else:
            prev_f_upper_min = np.array(
                [self.f_upper(x, y) for x, y in zip(self.x, self.y[self.k_min])]
            )
            self.f_upper_calls += self.lambda_x

        self.f_lower_min = Fold.copy()
        self.f_upper_min = prev_f_upper_min.copy()

        y_tilde = self.y[self.k_min].copy()
        alpha_tilde = self.alpha_y[self.k_min].copy()
        v_tilde = self.v_y[self.k_min].copy()

        h = np.ones(self.lambda_x, dtype=bool)
        iter_count = np.zeros(self.lambda_x, dtype=int)
        Fnew = Fold.copy()

        tau = -1.0
        while tau <= self.tau_thr:
            if not h.any():
                break
            for i in range(self.lambda_x):
                if not h[i]:
                    continue

                Fdash = Fnew[i]
                c = 0
                y_curr = y_tilde[i].copy()

                while c < self.cmax:
                    if iter_count[i] >= self.max_iter_lower:
                        h[i] = False
                        break

                    alpha_tilde[i], v_tilde[i], y_curr, fnew, lcalls = (
                        self._Lsearch_grad(
                            self.x[i], y_curr, alpha_tilde[i], v_tilde[i], Fdash
                        )
                    )
                    self.f_lower_calls += lcalls
                    iter_count[i] += 1

                    if fnew >= Fdash:
                        h[i] = False
                        break
                    else:
                        y_tilde[i] = y_curr
                        Fdash = fnew
                        c += 1

                if c >= 1:
                    Fnew[i] = Fdash
                    if self.single_objective:
                        self.f_upper_min[i] = Fdash
                    else:
                        self.f_upper_min[i] = self.f_upper(self.x[i], y_tilde[i])
                        self.f_upper_calls += 1

                self.f_lower_min[i] = Fnew[i]

            tau, _ = kendalltau(self.f_upper_min, prev_f_upper_min)
            prev_f_upper_min = self.f_upper_min.copy()

        k_min_set = np.unique(self.k_min)
        for idx in k_min_set:
            self.p[idx] = np.clip(self.p[idx] + self.pp, 0, 1)
            mask = self.k_min == idx
            best_local = int(np.argmin(self.f_upper_min[mask]))
            true_idx = int(np.where(mask)[0][best_local])

            self.y[idx] = y_tilde[true_idx]
            self.alpha_y[idx] = alpha_tilde[true_idx]
            self.v_y[idx] = v_tilde[true_idx]

        non_min_idxs = np.setdiff1d(np.arange(self.n_omega), k_min_set)
        self.p[non_min_idxs] -= self.pn
        reset_mask = self.p[non_min_idxs] <= self.p_thr
        for k in non_min_idxs[reset_mask]:
            self._reset_y(k)

        if self.reset_all:
            for k in range(self.n_omega):
                self._reset_y(k)

        rankings = np.argsort(self.f_upper_min)
        return rankings, y_tilde

    def optimize(
        self,
        on_step: Optional[Callable[[int, float, int], None]] = None,
    ) -> tuple[float, float, int, int, np.ndarray, np.ndarray, DataFrame]:
        """Run the bilevel optimization."""
        f_upper_hist: list[float] = []
        conv_flag = 0
        log_entries: list[dict[str, Union[int, float]]] = []
        nominal_x: list[np.ndarray] = []
        nominal_y: list[np.ndarray] = []

        for _ in range(self.max_iter):
            arx, ary, arz = self.upper_cma.sample()
            self.x = mirror(arx, self.xb[0], self.xb[1]) if self.is_x_bounded else arx

            rankings, y_tilde = self.ura()

            self.upper_cma.update(rankings, arx, ary, arz)

            log_det_D = np.sum(np.log(self.upper_cma.D))
            self.upper_cma.D /= np.exp(log_det_D / len(self.upper_cma.D))
            self.upper_cma.sigma *= np.exp(log_det_D / len(self.upper_cma.D))
            self.upper_cma.D = np.minimum(
                self.upper_cma.D, (self.xb[1] - self.xb[0]) / 4 / self.upper_cma.sigma
            )

            f_upper_hist.append(float(self.f_upper_min.min()))

            mirrored_x_mean = (
                mirror(self.upper_cma.xmean, self.xb[0], self.xb[1])
                if self.is_x_bounded
                else self.upper_cma.xmean
            )
            log_entries.append(
                {
                    "lower_call_count": self.f_lower_calls,
                    "upper_call_count": self.f_upper_calls,
                    "sigma_x": self.upper_cma.sigma,
                    "f_lower_min": float(self.f_lower_min.min()),
                    "f_upper_min": float(self.f_upper_min.min()),
                    **{f"mean_{i}": mirrored_x_mean[i] for i in range(self.dim_x)},
                    **{
                        f"std_{i}": self.upper_cma.coordinate_std[i]
                        for i in range(self.dim_x)
                    },
                    **{f"eigen_{i}": self.upper_cma.S[i] for i in range(self.dim_x)},
                    **{
                        f"f_lower_min_{i}": self.f_lower_min[i]
                        for i in range(self.lambda_x)
                    },
                    **{
                        f"f_upper_min_{i}": self.f_upper_min[i]
                        for i in range(self.lambda_x)
                    },
                }
            )

            if on_step is not None:
                on_step(
                    len(log_entries), float(self.f_upper_min.min()), self.total_f_calls
                )

            if (
                np.abs(self.f_upper_min.min() - self.true_upper_min)
                < self.gap_tolerance_x
            ):
                conv_flag = 1
            if len(f_upper_hist) > self.gap_window_size_x:
                if (
                    np.ptp(f_upper_hist[-self.gap_window_size_x :])
                    <= self.gap_tolerance_x
                ):
                    conv_flag = 2
            if self.upper_cma.S.max() / self.upper_cma.S.min() > self.Cxmax:
                conv_flag = 3
            if self.upper_cma.coordinate_std.max() < self.Vxmin:
                conv_flag = 4
            if self.total_f_calls >= self.max_f_calls:
                conv_flag = 5

            x_best = self.x[self.f_upper_min.argmin()].copy()
            y_best = y_tilde[self.f_upper_min.argmin()].copy()
            x, y = self.x.copy(), self.y.copy()

            if conv_flag:
                flag_names = {1: "success", 2: "stagnation", 3: "ill-conditioned", 4: "collapsed", 5: "budget"}
                print(f"  [stop] conv_flag={conv_flag} ({flag_names[conv_flag]}), f={self.f_upper_min.min():.3e}, iter={len(log_entries)}")
                nominal_x.extend(x)
                nominal_y.extend(y)
                self._reset_solvers()
                if self.restart and conv_flag not in (1, 5, 6):
                    conv_flag = 0
                else:
                    break

        nominal_x.extend(x)
        nominal_y.extend(y)

        columns = [
            "lower_call_count",
            "upper_call_count",
            "sigma_x",
            "f_lower_min",
            "f_upper_min",
        ]
        columns += [f"mean_{i}" for i in range(self.dim_x)]
        columns += [f"std_{i}" for i in range(self.dim_x)]
        columns += [f"eigen_{i}" for i in range(self.dim_x)]
        columns += [f"f_lower_min_{i}" for i in range(self.lambda_x)]
        columns += [f"f_upper_min_{i}" for i in range(self.lambda_x)]
        log_df = DataFrame(log_entries, columns=columns)

        f_lower_arr = np.array(
            [[self.f_lower(x, y) for y in nominal_y] for x in nominal_x]
        )
        self.f_lower_calls += len(nominal_x) * len(nominal_y)
        idx_y_arr = f_lower_arr.argmin(axis=1)
        f_upper_arr = np.array(
            [self.f_upper(x, nominal_y[k]) for x, k in zip(nominal_x, idx_y_arr)]
        )
        self.f_upper_calls += len(nominal_x)

        idx_x = int(f_upper_arr.argmin())
        x_min, y_min = nominal_x[idx_x], nominal_y[idx_y_arr[idx_x]]
        f_upper_min: float = float(f_upper_arr[idx_x])
        f_lower_min: float = self.f_lower(x_min, y_min)
        self.f_lower_calls += 1

        f_upper_best: float = self.f_upper(x_best, y_best)
        self.f_upper_calls += 1
        f_lower_best: float = self.f_lower(x_best, y_best)
        self.f_lower_calls += 1

        if np.abs(f_upper_best - self.true_upper_min) < np.abs(
            f_upper_min - self.true_upper_min
        ):
            f_upper_min, f_lower_min = f_upper_best, f_lower_best
            x_min, y_min = x_best, y_best

        return (
            f_upper_min,
            f_lower_min,
            self.f_upper_calls,
            self.f_lower_calls,
            x_min,
            y_min,
            log_df,
        )


class BilevelCMABFGS:
    """Bilevel optimizer with CMA-ES over x and BFGS via scipy.optimize.minimize over y.

    Uses scipy's BFGS implementation (method='BFGS') with inverse-Hessian warmstart
    via hess_inv0. Pool state is (y, B_y): y is the warmstarted starting point and
    B_y caches result.hess_inv per slot across outer iterations. cmax is passed as
    maxiter to scipy, so it counts optimizer iterations rather than successful steps.

    Parameters
    ----------
    f_upper : Callable
        Upper-level objective f_upper(x, y).
    f_lower : Callable or None
        Lower-level objective minimized over y. None defaults to -f_upper.
    dim_x, dim_y : int
        Dimensions of x and y.
    xb, yb : ndarray, shape (2, dim)
        Box bounds: row 0 lower, row 1 upper.
    df_lower : Callable
        Gradient of f_lower w.r.t. y.
    c1, c2 : float
        Wolfe condition constants (Armijo and curvature). Standard BFGS defaults.
    single_objective : bool
        f_lower is overridden to f_upper and calls are counted jointly.
    lambda_x : int or None
        Upper CMA population size. Default 4 + floor(3 ln dim_x).
    n_omega : int or None
        Pool size. Default 3 * lambda_x.
    Vxmin, Cxmax : float
        Upper CMA stopping thresholds (coordinate std, condition number).
    cmax : int
        Max successful BFGS steps per x[i] per tau-loop pass.
    tau_thr : float
        Kendall tau ranking-stability threshold.
    pn, pp, p_thr : float
        Pool probability decay, growth, and reset threshold.
    gap_tolerance_x, gap_window_size_x : float, int
        Upper-level gap stopping criterion.
    max_f_calls, max_iter, max_iter_lower : int
        Budget limits.
    restart : bool
        Restart on convergence unless exact optimum or budget exhausted.
    variance_update_x : bool
        Full covariance adaptation in upper CMA-ES.
    reset_all : bool
        Reset all pool slots each outer iteration (ablation).
    true_upper_min : float
        Known optimum value for early stopping.
    random_seed : int
    """

    @property
    def total_f_calls(self) -> int:
        return self.f_upper_calls + self.f_lower_calls

    def __init__(
        self,
        f_upper: Callable[[np.ndarray, np.ndarray], float],
        f_lower: Optional[Callable[[np.ndarray, np.ndarray], float]],
        dim_x: int,
        dim_y: int,
        xb: np.ndarray,
        yb: np.ndarray,
        df_lower: Callable[[np.ndarray, np.ndarray], np.ndarray],
        method: str = "BFGS",
        c1: float = 1e-4,
        c2: float = 0.9,
        single_objective: bool = False,
        is_x_bounded: bool = True,
        is_y_bounded: bool = True,
        lambda_x: Optional[int] = None,
        n_omega: Optional[int] = None,
        Vxmin: float = 1e-8,
        Cxmax: float = 1e7,
        cmax: int = 1,
        tau_thr: float = 0.7,
        pn: float = 0.05,
        pp: float = 0.4,
        p_thr: float = 0.1,
        gap_tolerance_x: float = 1e-6,
        gap_window_size_x: int = 60,
        max_f_calls: int = 10000000,
        max_iter: int = 100000000000000000000,
        max_iter_lower: int = 50,
        restart: bool = True,
        variance_update_x: bool = False,
        reset_all: bool = False,
        true_upper_min: float = 0.0,
        random_seed: int = 42,
    ) -> None:
        self.f_upper_calls = 0
        self.f_lower_calls = 0

        self.f_upper = f_upper
        self.f_lower = (lambda x, y: -f_upper(x, y)) if f_lower is None else f_lower
        self.single_objective = single_objective
        if single_objective:
            self.f_lower = self.f_upper

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.xb = xb
        self.yb = yb
        self.is_x_bounded = is_x_bounded
        self.is_y_bounded = is_y_bounded

        self.df_lower = df_lower
        self.method = method
        self.c1 = c1
        self.c2 = c2

        self.lambda_x = lambda_x if lambda_x is not None else 4 + int(3 * np.log(dim_x))
        self.n_omega = n_omega if n_omega is not None else 3 * self.lambda_x

        self.Vxmin = Vxmin
        self.Cxmax = Cxmax
        self.cmax = cmax
        self.tau_thr = tau_thr
        self.pp = pp
        self.pn = pn
        self.p_thr = p_thr
        self.gap_tolerance_x = gap_tolerance_x
        self.gap_window_size_x = gap_window_size_x
        self.max_f_calls = max_f_calls
        self.max_iter = max_iter
        self.max_iter_lower = max_iter_lower
        self.restart = restart
        self.variance_update_x = variance_update_x
        self.reset_all = reset_all
        self.true_upper_min = true_upper_min

        self.rng = np.random.default_rng(random_seed)

        self.x_mean = self.rng.random(size=dim_x) * (xb[1] - xb[0]) + xb[0]
        self.D_x = (xb[1] - xb[0]) / 4
        self.x = np.repeat(self.x_mean[np.newaxis], self.lambda_x, axis=0)
        self.upper_cma = self._make_upper_cma()

        self.p = np.ones(self.n_omega)
        self.y, self.B_y = self._init_pool()

        self.f_upper_min = np.empty(self.lambda_x, dtype=float)
        self.f_lower_min = np.empty(self.lambda_x, dtype=float)
        self.k_min = np.empty(self.lambda_x, dtype=int)

    def _make_upper_cma(self) -> MyDdCma:
        return MyDdCma(
            self.x_mean,
            self.D_x,
            self.lambda_x,
            flg_variance_update=self.variance_update_x,
            beta_eig=10 * self.dim_x**2,
            random_seed=self.rng.integers(999999),
        )

    def _init_pool(self) -> tuple[np.ndarray, np.ndarray]:
        delta_y = self.yb[1] - self.yb[0]
        y = self.rng.random((self.n_omega, self.dim_y)) * delta_y + self.yb[0]
        B = np.tile(np.eye(self.dim_y), (self.n_omega, 1, 1))
        return y, B

    def _reset_y(self, j: int) -> None:
        delta_y = self.yb[1] - self.yb[0]
        self.y[j] = self.rng.random(self.dim_y) * delta_y + self.yb[0]
        self.B_y[j] = np.eye(self.dim_y)
        self.p[j] = 1.0

    def _reset_solvers(self) -> None:
        self.y, self.B_y = self._init_pool()
        self.upper_cma = self._make_upper_cma()

    def _run_bfgs(
        self,
        x_i: np.ndarray,
        y_curr: np.ndarray,
        B: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, int]:
        """Run scipy optimizer from y_curr to convergence, warmstarting B for BFGS.

        Returns (y_next, fnew, B_new, nfev). For L-BFGS-B, B is returned unchanged
        since L-BFGS-B does not expose a dense inverse Hessian.
        """
        if self.method == "L-BFGS-B":
            bounds = list(zip(self.yb[0], self.yb[1])) if self.is_y_bounded else None
            result = scipy_minimize(
                lambda y: self.f_lower(x_i, y),
                y_curr,
                jac=lambda y: self.df_lower(x_i, y),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.cmax} if self.cmax is not None else {},
            )
            return result.x.copy(), float(result.fun), B, result.nfev

        def f_ls(y):
            y_eval = np.clip(y, self.yb[0], self.yb[1]) if self.is_y_bounded else y
            return self.f_lower(x_i, y_eval)

        def g_ls(y):
            y_eval = np.clip(y, self.yb[0], self.yb[1]) if self.is_y_bounded else y
            return self.df_lower(x_i, y_eval)

        opts = {"hess_inv0": B, "c1": self.c1, "c2": self.c2}
        if self.cmax is not None:
            opts["maxiter"] = self.cmax

        try:
            result = scipy_minimize(f_ls, y_curr, jac=g_ls, method="BFGS", options=opts)
        except ValueError:
            opts["hess_inv0"] = np.eye(self.dim_y)
            result = scipy_minimize(f_ls, y_curr, jac=g_ls, method="BFGS", options=opts)

        y_next = (
            np.clip(result.x, self.yb[0], self.yb[1])
            if self.is_y_bounded
            else result.x.copy()
        )
        return y_next, float(result.fun), result.hess_inv, result.nfev

    def ura(self) -> tuple[np.ndarray, np.ndarray]:
        """Upper-level ranking assessment using BFGS inner solver."""
        fx_arr = np.array([[self.f_lower(x, y) for y in self.y] for x in self.x])
        self.f_lower_calls += self.lambda_x * self.n_omega
        self.k_min = np.argmin(fx_arr, axis=1)

        Fold = np.min(fx_arr, axis=1)
        if self.single_objective:
            prev_f_upper_min = fx_arr[np.arange(self.lambda_x), self.k_min]
        else:
            prev_f_upper_min = np.array(
                [self.f_upper(x, y) for x, y in zip(self.x, self.y[self.k_min])]
            )
            self.f_upper_calls += self.lambda_x

        self.f_lower_min = Fold.copy()
        self.f_upper_min = prev_f_upper_min.copy()

        y_tilde = self.y[self.k_min].copy()
        B_tilde = self.B_y[self.k_min].copy()

        h = np.ones(self.lambda_x, dtype=bool)
        iter_count = np.zeros(self.lambda_x, dtype=int)
        Fnew = Fold.copy()

        tau = -1.0
        while tau <= self.tau_thr:
            if not h.any():
                break
            for i in range(self.lambda_x):
                if not h[i]:
                    continue
                if iter_count[i] >= self.max_iter_lower:
                    h[i] = False
                    continue

                fold = Fnew[i]
                y_next, fnew, B_next, nfev = self._run_bfgs(
                    self.x[i], y_tilde[i], B_tilde[i]
                )
                self.f_lower_calls += nfev
                iter_count[i] += 1

                if fnew < fold:
                    y_tilde[i] = y_next
                    B_tilde[i] = B_next
                    Fnew[i] = fnew
                    if self.single_objective:
                        self.f_upper_min[i] = fnew
                    else:
                        self.f_upper_min[i] = self.f_upper(self.x[i], y_tilde[i])
                        self.f_upper_calls += 1
                else:
                    h[i] = False

                self.f_lower_min[i] = Fnew[i]

            tau, _ = kendalltau(self.f_upper_min, prev_f_upper_min)
            prev_f_upper_min = self.f_upper_min.copy()

        k_min_set = np.unique(self.k_min)
        for idx in k_min_set:
            self.p[idx] = np.clip(self.p[idx] + self.pp, 0, 1)
            mask = self.k_min == idx
            best_local = int(np.argmin(self.f_upper_min[mask]))
            true_idx = int(np.where(mask)[0][best_local])

            self.y[idx] = y_tilde[true_idx]
            self.B_y[idx] = B_tilde[true_idx]

        non_min_idxs = np.setdiff1d(np.arange(self.n_omega), k_min_set)
        self.p[non_min_idxs] -= self.pn
        reset_mask = self.p[non_min_idxs] <= self.p_thr
        for k in non_min_idxs[reset_mask]:
            self._reset_y(k)

        if self.reset_all:
            for k in range(self.n_omega):
                self._reset_y(k)

        rankings = np.argsort(self.f_upper_min)
        return rankings, y_tilde

    def optimize(
        self,
        on_step: Optional[Callable[[int, float, int], None]] = None,
    ) -> tuple[float, float, int, int, np.ndarray, np.ndarray, DataFrame]:
        """Run the bilevel optimization."""
        f_upper_hist: list[float] = []
        conv_flag = 0
        log_entries: list[dict[str, Union[int, float]]] = []
        nominal_x: list[np.ndarray] = []
        nominal_y: list[np.ndarray] = []

        for _ in range(self.max_iter):
            arx, ary, arz = self.upper_cma.sample()
            self.x = mirror(arx, self.xb[0], self.xb[1]) if self.is_x_bounded else arx

            rankings, y_tilde = self.ura()

            self.upper_cma.update(rankings, arx, ary, arz)

            log_det_D = np.sum(np.log(self.upper_cma.D))
            self.upper_cma.D /= np.exp(log_det_D / len(self.upper_cma.D))
            self.upper_cma.sigma *= np.exp(log_det_D / len(self.upper_cma.D))
            self.upper_cma.D = np.minimum(
                self.upper_cma.D, (self.xb[1] - self.xb[0]) / 4 / self.upper_cma.sigma
            )

            f_upper_hist.append(float(self.f_upper_min.min()))

            mirrored_x_mean = (
                mirror(self.upper_cma.xmean, self.xb[0], self.xb[1])
                if self.is_x_bounded
                else self.upper_cma.xmean
            )
            log_entries.append(
                {
                    "lower_call_count": self.f_lower_calls,
                    "upper_call_count": self.f_upper_calls,
                    "sigma_x": self.upper_cma.sigma,
                    "f_lower_min": float(self.f_lower_min.min()),
                    "f_upper_min": float(self.f_upper_min.min()),
                    **{f"mean_{i}": mirrored_x_mean[i] for i in range(self.dim_x)},
                    **{
                        f"std_{i}": self.upper_cma.coordinate_std[i]
                        for i in range(self.dim_x)
                    },
                    **{f"eigen_{i}": self.upper_cma.S[i] for i in range(self.dim_x)},
                    **{
                        f"f_lower_min_{i}": self.f_lower_min[i]
                        for i in range(self.lambda_x)
                    },
                    **{
                        f"f_upper_min_{i}": self.f_upper_min[i]
                        for i in range(self.lambda_x)
                    },
                }
            )

            if on_step is not None:
                on_step(
                    len(log_entries), float(self.f_upper_min.min()), self.total_f_calls
                )

            if (
                np.abs(self.f_upper_min.min() - self.true_upper_min)
                < self.gap_tolerance_x
            ):
                conv_flag = 1
            if len(f_upper_hist) > self.gap_window_size_x:
                if (
                    np.ptp(f_upper_hist[-self.gap_window_size_x :])
                    <= self.gap_tolerance_x
                ):
                    conv_flag = 2
            if self.upper_cma.S.max() / self.upper_cma.S.min() > self.Cxmax:
                conv_flag = 3
            if self.upper_cma.coordinate_std.max() < self.Vxmin:
                conv_flag = 4
            if self.total_f_calls >= self.max_f_calls:
                conv_flag = 5

            x_best = self.x[self.f_upper_min.argmin()].copy()
            y_best = y_tilde[self.f_upper_min.argmin()].copy()
            x, y = self.x.copy(), self.y.copy()

            if conv_flag:
                nominal_x.extend(x)
                nominal_y.extend(y)
                self._reset_solvers()
                if self.restart and conv_flag not in (1, 5, 6):
                    conv_flag = 0
                else:
                    break

        nominal_x.extend(x)
        nominal_y.extend(y)

        columns = [
            "lower_call_count",
            "upper_call_count",
            "sigma_x",
            "f_lower_min",
            "f_upper_min",
        ]
        columns += [f"mean_{i}" for i in range(self.dim_x)]
        columns += [f"std_{i}" for i in range(self.dim_x)]
        columns += [f"eigen_{i}" for i in range(self.dim_x)]
        columns += [f"f_lower_min_{i}" for i in range(self.lambda_x)]
        columns += [f"f_upper_min_{i}" for i in range(self.lambda_x)]
        log_df = DataFrame(log_entries, columns=columns)

        f_lower_arr = np.array(
            [[self.f_lower(x, y) for y in nominal_y] for x in nominal_x]
        )
        self.f_lower_calls += len(nominal_x) * len(nominal_y)
        idx_y_arr = f_lower_arr.argmin(axis=1)
        f_upper_arr = np.array(
            [self.f_upper(x, nominal_y[k]) for x, k in zip(nominal_x, idx_y_arr)]
        )
        self.f_upper_calls += len(nominal_x)

        idx_x = int(f_upper_arr.argmin())
        x_min, y_min = nominal_x[idx_x], nominal_y[idx_y_arr[idx_x]]
        f_upper_min: float = float(f_upper_arr[idx_x])
        f_lower_min: float = self.f_lower(x_min, y_min)
        self.f_lower_calls += 1

        f_upper_best: float = self.f_upper(x_best, y_best)
        self.f_upper_calls += 1
        f_lower_best: float = self.f_lower(x_best, y_best)
        self.f_lower_calls += 1

        if np.abs(f_upper_best - self.true_upper_min) < np.abs(
            f_upper_min - self.true_upper_min
        ):
            f_upper_min, f_lower_min = f_upper_best, f_lower_best
            x_min, y_min = x_best, y_best

        return (
            f_upper_min,
            f_lower_min,
            self.f_upper_calls,
            self.f_lower_calls,
            x_min,
            y_min,
            log_df,
        )


class BilevelCMALBFGS:
    """Bilevel optimizer with CMA-ES over x and warm-started L-BFGS (torch.optim.LBFGS) over y.

    Per-slot state persists (old_dirs, old_stps, ro, H_diag, prev_flat_grad) across outer
    iterations -- the curvature warm-start equivalent of what BilevelCMABFGS does for dense BFGS.
    history_size controls how many (s, y) pairs are kept; cmax sets max_iter per step() call.

    Parameters
    ----------
    history_size : int
        Number of (s, y) curvature pairs kept per pool slot. Default 10.
    lr : float
        Step size passed to LBFGS (leave at 1.0 when using strong_wolfe).
    line_search_fn : str or None
        'strong_wolfe' or None (Armijo-only backtracking).
    cmax : int
        max_iter per optimizer.step() call -- L-BFGS iterations per tau-loop pass.
    (all other parameters identical to BilevelCMABFGS)
    """

    @property
    def total_f_calls(self) -> int:
        return self.f_upper_calls + self.f_lower_calls

    def __init__(
        self,
        f_upper,
        f_lower,
        dim_x: int,
        dim_y: int,
        xb,
        yb,
        df_lower,
        history_size: int = 10,
        lr: float = 1.0,
        line_search_fn = "strong_wolfe",
        single_objective: bool = False,
        is_x_bounded: bool = True,
        is_y_bounded: bool = True,
        lambda_x = None,
        n_omega = None,
        Vxmin: float = 1e-8,
        Cxmax: float = 1e7,
        cmax: int = 20,
        tau_thr: float = 0.7,
        pn: float = 0.05,
        pp: float = 0.4,
        p_thr: float = 0.1,
        gap_tolerance_x: float = 1e-6,
        gap_window_size_x: int = 60,
        max_f_calls: int = 10000000,
        max_iter: int = 100000000000000000000,
        max_iter_lower: int = 50,
        restart: bool = True,
        variance_update_x: bool = False,
        reset_all: bool = False,
        true_upper_min: float = 0.0,
        random_seed: int = 42,
    ) -> None:
        import numpy as np
        self.f_upper_calls = 0
        self.f_lower_calls = 0

        self.f_upper = f_upper
        self.f_lower = (lambda x, y: -f_upper(x, y)) if f_lower is None else f_lower
        self.single_objective = single_objective
        if single_objective:
            self.f_lower = self.f_upper

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.xb = xb
        self.yb = yb
        self.is_x_bounded = is_x_bounded
        self.is_y_bounded = is_y_bounded

        self.df_lower = df_lower
        self.history_size = history_size
        self.lr = lr
        self.line_search_fn = line_search_fn

        self.lambda_x = lambda_x if lambda_x is not None else 4 + int(3 * np.log(dim_x))
        self.n_omega = n_omega if n_omega is not None else 3 * self.lambda_x

        self.Vxmin = Vxmin
        self.Cxmax = Cxmax
        self.cmax = cmax
        self.tau_thr = tau_thr
        self.pp = pp
        self.pn = pn
        self.p_thr = p_thr
        self.gap_tolerance_x = gap_tolerance_x
        self.gap_window_size_x = gap_window_size_x
        self.max_f_calls = max_f_calls
        self.max_iter = max_iter
        self.max_iter_lower = max_iter_lower
        self.restart = restart
        self.variance_update_x = variance_update_x
        self.reset_all = reset_all
        self.true_upper_min = true_upper_min

        self.rng = np.random.default_rng(random_seed)

        self.x_mean = self.rng.random(size=dim_x) * (xb[1] - xb[0]) + xb[0]
        self.D_x = (xb[1] - xb[0]) / 4
        self.x = np.repeat(self.x_mean[np.newaxis], self.lambda_x, axis=0)
        self.upper_cma = self._make_upper_cma()

        self.p = np.ones(self.n_omega)
        self.y, self.lbfgs_state = self._init_pool()

        self.f_upper_min = np.empty(self.lambda_x, dtype=float)
        self.f_lower_min = np.empty(self.lambda_x, dtype=float)
        self.k_min = np.empty(self.lambda_x, dtype=int)

    def _make_upper_cma(self):
        return MyDdCma(
            self.x_mean,
            self.D_x,
            self.lambda_x,
            flg_variance_update=self.variance_update_x,
            beta_eig=10 * self.dim_x**2,
            random_seed=self.rng.integers(999999),
        )

    def _init_pool(self):
        delta_y = self.yb[1] - self.yb[0]
        y = self.rng.random((self.n_omega, self.dim_y)) * delta_y + self.yb[0]
        return y, [None] * self.n_omega

    def _reset_y(self, j: int) -> None:
        delta_y = self.yb[1] - self.yb[0]
        self.y[j] = self.rng.random(self.dim_y) * delta_y + self.yb[0]
        self.lbfgs_state[j] = None
        self.p[j] = 1.0

    def _reset_solvers(self) -> None:
        self.y, self.lbfgs_state = self._init_pool()
        self.upper_cma = self._make_upper_cma()

    def _run_lbfgs(self, x_i, y_curr, saved_state):
        """Run one optimizer.step() of torch L-BFGS, warm-starting from saved_state."""
        import torch
        import numpy as np

        param = torch.nn.Parameter(torch.tensor(y_curr.copy(), dtype=torch.float64))
        lbfgs_kwargs = dict(
            lr=self.lr,
            history_size=self.history_size,
            line_search_fn=self.line_search_fn,
        )
        if self.cmax is not None:
            lbfgs_kwargs["max_iter"] = self.cmax
        optimizer = torch.optim.LBFGS([param], **lbfgs_kwargs)

        # Inject saved curvature state. optimizer.state is a defaultdict(dict), so accessing
        # optimizer.state[param] before any step creates an empty entry we can populate.
        # LBFGS only re-initializes state when len(state) == 0, so pre-populating here
        # causes it to reuse the saved history on the first step.
        if saved_state is not None:
            try:
                state = optimizer.state[param]
                state["func_evals"] = 0
                state["n_iter"] = saved_state["n_iter"]
                state["old_dirs"] = [
                    torch.tensor(v, dtype=torch.float64) for v in saved_state["old_dirs"]
                ]
                state["old_stps"] = [
                    torch.tensor(v, dtype=torch.float64) for v in saved_state["old_stps"]
                ]
                state["ro"] = list(saved_state["ro"])
                if saved_state.get("H_diag") is not None:
                    state["H_diag"] = torch.tensor(saved_state["H_diag"], dtype=torch.float64)
                if saved_state.get("prev_flat_grad") is not None:
                    state["prev_flat_grad"] = torch.tensor(
                        saved_state["prev_flat_grad"], dtype=torch.float64
                    )
                if saved_state.get("d") is not None:
                    state["d"] = torch.tensor(saved_state["d"], dtype=torch.float64)
                if saved_state.get("t") is not None:
                    state["t"] = saved_state["t"]
            except Exception:
                optimizer.state[param].clear()

        last_f = [float("inf")]
        nfev = [0]

        def closure():
            optimizer.zero_grad()
            y_np = param.detach().numpy().copy()
            if self.is_y_bounded:
                y_np = np.clip(y_np, self.yb[0], self.yb[1])
            f_val = float(self.f_lower(x_i, y_np))
            g_val = self.df_lower(x_i, y_np)
            param.grad = torch.tensor(g_val.copy(), dtype=torch.float64)
            nfev[0] += 1
            last_f[0] = f_val
            return torch.tensor(f_val, dtype=torch.float64)

        optimizer.step(closure)

        y_next = param.detach().numpy().copy()
        if self.is_y_bounded:
            y_next = np.clip(y_next, self.yb[0], self.yb[1])
        fnew = last_f[0]

        new_state = None
        raw = optimizer.state.get(param)
        if raw:
            new_state = {
                "n_iter": raw.get("n_iter", 0),
                "old_dirs": [v.detach().numpy().copy() for v in raw.get("old_dirs", [])],
                "old_stps": [v.detach().numpy().copy() for v in raw.get("old_stps", [])],
                "ro": list(raw.get("ro", [])),
                "H_diag": float(raw["H_diag"]) if "H_diag" in raw else None,
                "prev_flat_grad": (
                    raw["prev_flat_grad"].detach().numpy().copy()
                    if "prev_flat_grad" in raw
                    else None
                ),
                "d": raw["d"].detach().numpy().copy() if "d" in raw else None,
                "t": float(raw["t"]) if "t" in raw else None,
            }

        return y_next, fnew, new_state, nfev[0]

    def ura(self):
        """Upper-level ranking assessment using warm-started torch L-BFGS inner solver."""
        import numpy as np
        from scipy.stats import kendalltau
        fx_arr = np.array([[self.f_lower(x, y) for y in self.y] for x in self.x])
        self.f_lower_calls += self.lambda_x * self.n_omega
        self.k_min = np.argmin(fx_arr, axis=1)

        Fold = np.min(fx_arr, axis=1)
        if self.single_objective:
            prev_f_upper_min = fx_arr[np.arange(self.lambda_x), self.k_min]
        else:
            prev_f_upper_min = np.array(
                [self.f_upper(x, y) for x, y in zip(self.x, self.y[self.k_min])]
            )
            self.f_upper_calls += self.lambda_x

        self.f_lower_min = Fold.copy()
        self.f_upper_min = prev_f_upper_min.copy()

        y_tilde = self.y[self.k_min].copy()
        lbfgs_state_tilde = [self.lbfgs_state[k] for k in self.k_min]

        h = np.ones(self.lambda_x, dtype=bool)
        iter_count = np.zeros(self.lambda_x, dtype=int)
        Fnew = Fold.copy()

        tau = -1.0
        while tau <= self.tau_thr:
            if not h.any():
                break
            for i in range(self.lambda_x):
                if not h[i]:
                    continue
                if iter_count[i] >= self.max_iter_lower:
                    h[i] = False
                    continue

                fold = Fnew[i]
                y_next, fnew, new_state, nfev = self._run_lbfgs(
                    self.x[i], y_tilde[i], lbfgs_state_tilde[i]
                )
                self.f_lower_calls += nfev
                iter_count[i] += 1

                if fnew < fold:
                    y_tilde[i] = y_next
                    lbfgs_state_tilde[i] = new_state
                    Fnew[i] = fnew
                    if self.single_objective:
                        self.f_upper_min[i] = fnew
                    else:
                        self.f_upper_min[i] = self.f_upper(self.x[i], y_tilde[i])
                        self.f_upper_calls += 1
                else:
                    h[i] = False

                self.f_lower_min[i] = Fnew[i]

            tau, _ = kendalltau(self.f_upper_min, prev_f_upper_min)
            prev_f_upper_min = self.f_upper_min.copy()

        k_min_set = np.unique(self.k_min)
        for idx in k_min_set:
            self.p[idx] = np.clip(self.p[idx] + self.pp, 0, 1)
            mask = self.k_min == idx
            best_local = int(np.argmin(self.f_upper_min[mask]))
            true_idx = int(np.where(mask)[0][best_local])

            self.y[idx] = y_tilde[true_idx]
            self.lbfgs_state[idx] = lbfgs_state_tilde[true_idx]

        non_min_idxs = np.setdiff1d(np.arange(self.n_omega), k_min_set)
        self.p[non_min_idxs] -= self.pn
        reset_mask = self.p[non_min_idxs] <= self.p_thr
        for k in non_min_idxs[reset_mask]:
            self._reset_y(k)

        if self.reset_all:
            for k in range(self.n_omega):
                self._reset_y(k)

        rankings = np.argsort(self.f_upper_min)
        return rankings, y_tilde

    def optimize(self, on_step=None):
        """Run the bilevel optimization."""
        import numpy as np
        from pandas import DataFrame
        f_upper_hist = []
        conv_flag = 0
        log_entries = []
        nominal_x = []
        nominal_y = []

        for _ in range(self.max_iter):
            arx, ary, arz = self.upper_cma.sample()
            self.x = mirror(arx, self.xb[0], self.xb[1]) if self.is_x_bounded else arx

            rankings, y_tilde = self.ura()

            self.upper_cma.update(rankings, arx, ary, arz)

            log_det_D = np.sum(np.log(self.upper_cma.D))
            self.upper_cma.D /= np.exp(log_det_D / len(self.upper_cma.D))
            self.upper_cma.sigma *= np.exp(log_det_D / len(self.upper_cma.D))
            self.upper_cma.D = np.minimum(
                self.upper_cma.D, (self.xb[1] - self.xb[0]) / 4 / self.upper_cma.sigma
            )

            f_upper_hist.append(float(self.f_upper_min.min()))

            mirrored_x_mean = (
                mirror(self.upper_cma.xmean, self.xb[0], self.xb[1])
                if self.is_x_bounded
                else self.upper_cma.xmean
            )
            log_entries.append(
                {
                    "lower_call_count": self.f_lower_calls,
                    "upper_call_count": self.f_upper_calls,
                    "sigma_x": self.upper_cma.sigma,
                    "f_lower_min": float(self.f_lower_min.min()),
                    "f_upper_min": float(self.f_upper_min.min()),
                    **{f"mean_{i}": mirrored_x_mean[i] for i in range(self.dim_x)},
                    **{f"std_{i}": self.upper_cma.coordinate_std[i] for i in range(self.dim_x)},
                    **{f"eigen_{i}": self.upper_cma.S[i] for i in range(self.dim_x)},
                    **{f"f_lower_min_{i}": self.f_lower_min[i] for i in range(self.lambda_x)},
                    **{f"f_upper_min_{i}": self.f_upper_min[i] for i in range(self.lambda_x)},
                }
            )

            if on_step is not None:
                on_step(len(log_entries), float(self.f_upper_min.min()), self.total_f_calls)

            if np.abs(self.f_upper_min.min() - self.true_upper_min) < self.gap_tolerance_x:
                conv_flag = 1
            if len(f_upper_hist) > self.gap_window_size_x:
                if np.ptp(f_upper_hist[-self.gap_window_size_x:]) <= self.gap_tolerance_x:
                    conv_flag = 2
            if self.upper_cma.S.max() / self.upper_cma.S.min() > self.Cxmax:
                conv_flag = 3
            if self.upper_cma.coordinate_std.max() < self.Vxmin:
                conv_flag = 4
            if self.total_f_calls >= self.max_f_calls:
                conv_flag = 5

            x_best = self.x[self.f_upper_min.argmin()].copy()
            y_best = y_tilde[self.f_upper_min.argmin()].copy()
            x, y = self.x.copy(), self.y.copy()

            if conv_flag:
                flag_names = {1: "success", 2: "stagnation", 3: "ill-conditioned", 4: "collapsed", 5: "budget"}
                print(f"  [stop] conv_flag={conv_flag} ({flag_names[conv_flag]}), f={self.f_upper_min.min():.3e}, iter={len(log_entries)}")
                nominal_x.extend(x)
                nominal_y.extend(y)
                self._reset_solvers()
                if self.restart and conv_flag not in (1, 5, 6):
                    conv_flag = 0
                else:
                    break

        nominal_x.extend(x)
        nominal_y.extend(y)

        columns = ["lower_call_count", "upper_call_count", "sigma_x", "f_lower_min", "f_upper_min"]
        columns += [f"mean_{i}" for i in range(self.dim_x)]
        columns += [f"std_{i}" for i in range(self.dim_x)]
        columns += [f"eigen_{i}" for i in range(self.dim_x)]
        columns += [f"f_lower_min_{i}" for i in range(self.lambda_x)]
        columns += [f"f_upper_min_{i}" for i in range(self.lambda_x)]
        log_df = DataFrame(log_entries, columns=columns)

        f_lower_arr = np.array([[self.f_lower(x, y) for y in nominal_y] for x in nominal_x])
        self.f_lower_calls += len(nominal_x) * len(nominal_y)
        idx_y_arr = f_lower_arr.argmin(axis=1)
        f_upper_arr = np.array(
            [self.f_upper(x, nominal_y[k]) for x, k in zip(nominal_x, idx_y_arr)]
        )
        self.f_upper_calls += len(nominal_x)

        idx_x = int(f_upper_arr.argmin())
        x_min, y_min = nominal_x[idx_x], nominal_y[idx_y_arr[idx_x]]
        f_upper_min = float(f_upper_arr[idx_x])
        f_lower_min = self.f_lower(x_min, y_min)
        self.f_lower_calls += 1

        f_upper_best = self.f_upper(x_best, y_best)
        self.f_upper_calls += 1
        f_lower_best = self.f_lower(x_best, y_best)
        self.f_lower_calls += 1

        if np.abs(f_upper_best - self.true_upper_min) < np.abs(f_upper_min - self.true_upper_min):
            f_upper_min, f_lower_min = f_upper_best, f_lower_best
            x_min, y_min = x_best, y_best

        return (f_upper_min, f_lower_min, self.f_upper_calls, self.f_lower_calls, x_min, y_min, log_df)
