from typing import Callable, Optional, Union

import numpy as np
from pandas import DataFrame
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
            flg_variance_update=False,
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
                flg_variance_update=False,
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

    def optimize(
        self,
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
            if self.f_upper_calls + self.f_lower_calls >= self.max_f_calls:
                conv_flag = 5

            x_best = self.x[self.f_upper_min.argmin()]
            y_best = y_tilde[self.f_upper_min.argmin()]

            x, y = self.x.copy(), self.y.copy()

            if conv_flag:
                nominal_x.extend(x)
                nominal_y.extend(y)
                self._reset_solvers()
                if self.restart and conv_flag not in (1, 5, 6):
                    # print("Restarting ...")
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
