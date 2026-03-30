from datetime import datetime
import json
import os

from multiprocessing import Pool, cpu_count
from typing import Callable, Optional
from scipy.stats import iqr
import numpy as np
from numpy.typing import NDArray

import journal_minmaxf as ff
from smd import SMD_PROBLEMS
from bilevel_cma import BilevelCMA

WRA_FUNCTION_NAMES: dict[str, str] = {
    "WRA1": "flxly",
    "WRA2": "fcnc",
    "WRA3": "fncc",
    "WRA4": "fmsp",
    "WRA5": "fqcc",
    "WRA6": "fnsscc",
    "WRA7": "fnscc",
    "WRA8": "fnscnsc",
    "WRA9": "fcvncc",
    "WRA10": "fc4",
    "WRA11": "fellqcc",
}

URA_PARAMS = dict(
    cmax=1,
    Vxmin=1e-12,
    Vymin=1e-4,
    Cxmax=1e7,
    Cymax=1e7,
    gap_tolerance_x=1e-6,
    gap_tolerance_y=1e-6,
    gap_window_size_x=60,
    gap_window_size_y=20,
    max_iter_lower=50,
    restart=True,
)

EXPERIMENT_CONFIGS = {
    "standard": {},
    "ablate_es": {"tau_thr": 1.1},
    "ablate_ws_a": {"n_omega": 1},
    "ablate_ws_b": {"n_omega": 1, "reset_all": True},
}


def run_optimize(
    optimizer: BilevelCMA,
    output_dir: str,
    output_name: str,
    true_upper_min: float = 0.0,
    true_lower_min: float = 0.0,
):
    f_upper_min, f_lower_min, f_upper_calls, f_lower_calls, x_min, y_min, log_df = (
        optimizer.optimize()
    )
    os.makedirs(output_dir, exist_ok=True)
    log_df.to_csv(f"{output_dir}/{output_name}_log.csv", index=False)
    with open(f"{output_dir}/{output_name}_final.csv", "w") as f:
        f.write(f"f_upper_acc,{np.abs(f_upper_min - true_upper_min)}\n")
        f.write(f"f_lower_acc,{np.abs(f_lower_min - true_lower_min)}\n")
        f.write(f"f_upper_calls,{f_upper_calls}\n")
        f.write(f"f_lower_calls,{f_lower_calls}\n")
    return f_upper_min, f_lower_min, f_upper_calls, f_lower_calls, x_min, y_min, log_df


def run_experiment(
    fn_name: str,
    f_upper: Callable[[NDArray, NDArray], float],
    f_lower: Optional[Callable[[NDArray, NDArray], float]],
    dim_x: int,
    dim_y: int,
    xb: NDArray,
    yb: NDArray,
    n_runs: int,
    output_dir: str,
    true_upper_min: float = 0.0,
    true_lower_min: float = 0.0,
    n_workers: Optional[int] = None,
    **ura_overrides,
) -> dict:
    if n_workers is None:
        n_workers = min(n_runs, cpu_count() - 1 or 1)

    ura_params = {**URA_PARAMS, **ura_overrides}

    with Pool(n_workers) as pool:
        out = pool.starmap(
            run_optimize,
            zip(
                [
                    BilevelCMA(
                        f_upper,
                        f_lower,
                        dim_x,
                        dim_y,
                        xb,
                        yb,
                        is_x_bounded=True,
                        is_y_bounded=True,
                        true_upper_min=true_upper_min,
                        random_seed=run_no,
                        **ura_params,
                    )
                    for run_no in range(n_runs)
                ],
                [output_dir] * n_runs,
                [f"{fn_name}_{dim_x}_{dim_y}_{run_no}" for run_no in range(n_runs)],
                [true_upper_min] * n_runs,
                [true_lower_min] * n_runs,
            ),
        )

    results = compute_results(
        out=out,
        true_upper_min=true_upper_min,
        true_lower_min=true_lower_min,
    )

    print_results(
        fn_name=fn_name,
        dim_x=dim_x,
        dim_y=dim_y,
        n_runs=n_runs,
        results=results,
    )

    return results


def compute_results(
    out: list,
    true_upper_min: float,
    true_lower_min: float,
) -> dict:
    """Compute results dictionary for a single problem."""
    f_upper_acc_raw = [float(abs(res[0] - true_upper_min)) for res in out]
    f_lower_acc_raw = [float(abs(res[1] - true_lower_min)) for res in out]
    f_upper_calls_raw = [int(res[2]) for res in out]
    f_lower_calls_raw = [int(res[3]) for res in out]

    # For median/iqr calculations, clamp accuracy to 1e-6
    f_upper_acc_clamped = np.maximum(np.array(f_upper_acc_raw), 1e-6)
    f_upper_calls_arr = np.array(f_upper_calls_raw)
    f_lower_calls_arr = np.array(f_lower_calls_raw)

    total_calls_raw = [u + l for u, l in zip(f_upper_calls_raw, f_lower_calls_raw)]
    total_calls_arr = np.array(total_calls_raw)

    return {
        "upper_accuracy": {
            "median": float(np.median(f_upper_acc_clamped)),
            "iqr": float(iqr(f_upper_acc_clamped)),
            "raw": f_upper_acc_raw,
        },
        "lower_accuracy": {
            "median": float(np.median(f_lower_acc_raw)),
            "iqr": float(iqr(f_lower_acc_raw)),
            "raw": f_lower_acc_raw,
        },
        "upper_fe": {
            "median": float(np.median(f_upper_calls_arr)),
            "iqr": float(iqr(f_upper_calls_arr)),
            "raw": f_upper_calls_raw,
        },
        "lower_fe": {
            "median": float(np.median(f_lower_calls_arr)),
            "iqr": float(iqr(f_lower_calls_arr)),
            "raw": f_lower_calls_raw,
        },
        "total_fe": {
            "median": float(np.median(total_calls_arr)),
            "iqr": float(iqr(total_calls_arr)),
            "raw": total_calls_raw,
        },
    }


def print_results(
    fn_name: str,
    dim_x: int,
    dim_y: int,
    n_runs: int,
    results: dict,
):
    def fmt(val: float) -> str:
        return f"{val:.2E}".replace("E-0", "E-").replace("E+0", "E+")

    print(f" {fn_name} ({dim_x}+{dim_y}, {n_runs} runs) ".center(35, "="))

    print(f"f_u acc  : median {fmt(results['upper_accuracy']['median'])} ({fmt(results['upper_accuracy']['iqr'])})")
    print(f"f_u call : median {fmt(results['upper_fe']['median'])} ({fmt(results['upper_fe']['iqr'])})")

    print("-" * 35)

    print(f"f_l acc  : median {fmt(results['lower_accuracy']['median'])} ({fmt(results['lower_accuracy']['iqr'])})")
    print(f"f_l call : median {fmt(results['lower_fe']['median'])} ({fmt(results['lower_fe']['iqr'])})")

    print("-" * 35)

    print(f"tot call : median {fmt(results['total_fe']['median'])} ({fmt(results['total_fe']['iqr'])})")

    print("=" * 35)


def save_summary(summary: dict, output_dir: str):
    """Save the summary dictionary to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_path}")


if __name__ == "__main__":
    n_runs = 20
    dim_x, dim_y = 20, 20
    output_base = "out"
    n_workers = None

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    timestamp_dir = f"{output_base}/{timestamp}"

    summary: dict[str, dict[str, dict]] = {}

    for config_name, config_params in EXPERIMENT_CONFIGS.items():
        output_dir = f"{timestamp_dir}/{config_name}"
        print(f" Running {config_name} ".center(35, "*"))
        for key, value in config_params.items():
            print(f"* {key} = {value}".ljust(34) + "*")
            print("*" * 35)

        summary[config_name] = {}

        for fn_name, attr in WRA_FUNCTION_NAMES.items():
            true_upper_min = 6.6 * dim_x if fn_name == "WRA3" else 0.0

            B = ff.setB(dim_y, dim_x, 1).setB()

            xb = np.array([[-3.0] * dim_x, [3.0] * dim_x])
            yb = np.array([[-3.0] * dim_y, [3.0] * dim_y])

            f_upper = getattr(ff.minmaxf(B, yb[:, 0], xb[:, 0]), attr)

            results = run_experiment(
                fn_name=fn_name,
                f_upper=f_upper,
                f_lower=None,
                dim_x=dim_x,
                dim_y=dim_y,
                xb=xb,
                yb=yb,
                n_runs=n_runs,
                output_dir=output_dir,
                true_upper_min=true_upper_min,
                true_lower_min=-true_upper_min,
                n_workers=n_workers,
                **config_params,
            )
            summary[config_name][fn_name] = results

        # SMD problems
        for SMD in SMD_PROBLEMS:
            smd = SMD(dim_x, dim_y)
            fn_name = smd.__class__.__name__

            results = run_experiment(
                fn_name=fn_name,
                f_upper=smd.f_upper,
                f_lower=smd.f_lower,
                dim_x=dim_x,
                dim_y=dim_y,
                xb=smd.xu_bound,
                yb=smd.xl_bound,
                n_runs=n_runs,
                output_dir=output_dir,
                true_upper_min=0.0,
                true_lower_min=0.0,
                n_workers=n_workers,
                **config_params,
            )
            summary[config_name][fn_name] = results

        print("*" * 35)

    save_summary(summary, timestamp_dir)

