import numpy as np
from numpy.typing import NDArray


def safe_log(x: NDArray[float], clip_thr: float = 1e-10) -> NDArray[float]:
    clipped_x = np.where(x > clip_thr, x, clip_thr)
    return np.log(clipped_x)


# TODO: support 2d vectorization
class SMDBase:
    xu_bound: NDArray[float]
    xl_bound: NDArray[float]

    def __init__(self, dim_xu: int, dim_xl: int, use_s: bool = False) -> None:
        if dim_xu > dim_xl:
            raise ValueError("dim_xu must be less than or equal to dim_xl.")
        r = dim_xu // 2
        p = dim_xu - r
        if use_s:  # Used only for SMD6 problem
            eps = 1.0e-5
            q = np.floor((dim_xl - r) / 2.0 - eps).astype(int)
            s = np.ceil((dim_xl - r) / 2.0 + eps).astype(int)
        else:
            q = dim_xl - r
            s = 0

        self.p, self.q, self.r, self.s = p, q, r, s
        self.dim_xu, self.dim_xl = dim_xu, dim_xl

    def split_vars(
        self, xu: NDArray[float], xl: NDArray[float]
    ) -> tuple[NDArray[float], NDArray[float], NDArray[float], NDArray[float]]:
        xu1, xu2 = xu[: self.p], xu[self.p :]
        xl1, xl2 = xl[: self.q + self.s], xl[self.q + self.s :]
        assert xu1.shape == (self.p,)
        assert xl1.shape == (self.q + self.s,)
        assert xu2.shape == xl2.shape == (self.r,)
        return xu1, xu2, xl1, xl2

    def f_upper(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        raise NotImplementedError

    def f_lower(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        raise NotImplementedError


class SMD1(SMDBase):
    def __init__(self, dim_xu: int, dim_xl: int) -> None:
        super().__init__(dim_xu, dim_xl)
        xu_bound = np.zeros((2, self.dim_xu))
        xu_bound[0], xu_bound[1] = -5, 10
        self.xu_bound = xu_bound

        xl_bound = np.zeros((2, self.dim_xl))
        xl_bound[0, : self.q] = -5
        xl_bound[0, self.q :] = -np.pi / 2
        xl_bound[1, : self.q] = 10
        xl_bound[1, self.q :] = np.pi / 2
        self.xl_bound = xl_bound

    def f_upper(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        F1 = np.sum(xu1**2)
        F2 = np.sum(xl1**2)
        F3 = np.sum(xu2**2) + np.sum((xu2 - np.tan(xl2)) ** 2)
        return F1 + F2 + F3

    def f_lower(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        f1 = np.sum(xu1**2)
        f2 = np.sum(xl1**2)
        f3 = np.sum((xu2 - np.tan(xl2)) ** 2)
        return f1 + f2 + f3


class SMD2(SMDBase):
    def __init__(self, dim_xu: int, dim_xl: int) -> None:
        super().__init__(dim_xu, dim_xl)
        xu_bound = np.zeros((2, self.dim_xu))
        xu_bound[0] = -5
        xu_bound[1, : self.p] = 10
        xu_bound[1, self.p :] = 1
        self.xu_bound = xu_bound

        xl_bound = np.zeros((2, self.dim_xl))
        xl_bound[0, : self.q] = -5
        xl_bound[0, self.q :] = 0
        xl_bound[1, : self.q] = 10
        xl_bound[1, self.q :] = np.e
        self.xl_bound = xl_bound

    def f_upper(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        F1 = np.sum(xu1**2)
        F2 = -np.sum(xl1**2)
        F3 = np.sum(xu2**2) - np.sum((xu2 - safe_log(xl2)) ** 2)
        return F1 + F2 + F3

    def f_lower(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        f1 = np.sum(xu1**2)
        f2 = np.sum(xl1**2)
        f3 = np.sum((xu2 - safe_log(xl2)) ** 2)
        return f1 + f2 + f3


class SMD3(SMDBase):
    def __init__(self, dim_xu: int, dim_xl: int) -> None:
        super().__init__(dim_xu, dim_xl)
        xu_bound = np.zeros((2, self.dim_xu))
        xu_bound[0], xu_bound[1] = -5, 10
        self.xu_bound = xu_bound

        xl_bound = np.zeros((2, self.dim_xl))
        xl_bound[0, : self.q] = -5
        xl_bound[0, self.q :] = -np.pi / 2
        xl_bound[1, : self.q] = 10
        xl_bound[1, self.q :] = np.pi / 2
        self.xl_bound = xl_bound

    def f_upper(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        F1 = np.sum(xu1**2)
        F2 = np.sum(xl1**2)
        F3 = np.sum(xu2**2) + np.sum((xu2**2 - np.tan(xl2)) ** 2)
        return F1 + F2 + F3

    def f_lower(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        f1 = np.sum(xu1**2)
        f2 = self.q + np.sum((xl1**2 - np.cos(2 * np.pi * xl1)))
        f3 = np.sum((xu2**2 - np.tan(xl2)) ** 2)
        return f1 + f2 + f3


class SMD4(SMDBase):
    def __init__(self, dim_xu: int, dim_xl: int) -> None:
        super().__init__(dim_xu, dim_xl)
        xu_bound = np.zeros((2, self.dim_xu))
        xu_bound[0, : self.p] = -5
        xu_bound[0, self.p :] = -1
        xu_bound[1, : self.p] = 10
        xu_bound[1, self.p :] = 1
        self.xu_bound = xu_bound

        xl_bound = np.zeros((2, self.dim_xl))
        xl_bound[0, : self.q] = -5
        xl_bound[0, self.q :] = 0
        xl_bound[1, : self.q] = 10
        xl_bound[1, self.q :] = np.e
        self.xl_bound = xl_bound

    def f_upper(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        F1 = np.sum(xu1**2)
        F2 = -np.sum(xl1**2)
        F3 = np.sum(xu2**2) - np.sum((np.abs(xu2) - safe_log(1 + xl2)) ** 2)
        return F1 + F2 + F3

    def f_lower(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        f1 = np.sum(xu1**2)
        f2 = self.q + np.sum((xl1**2 - np.cos(2 * np.pi * xl1)))
        f3 = np.sum((np.abs(xu2) - safe_log(1 + xl2)) ** 2)
        return f1 + f2 + f3


class SMD5(SMDBase):
    def __init__(self, dim_xu: int, dim_xl: int) -> None:
        super().__init__(dim_xu, dim_xl)
        xu_bound = np.zeros((2, self.dim_xu))
        xu_bound[0], xu_bound[1] = -5, 10
        self.xu_bound = xu_bound

        xl_bound = np.zeros((2, self.dim_xl))
        xl_bound[0], xl_bound[1] = -5, 10
        self.xl_bound = xl_bound

    def f_upper(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        F1 = np.sum(xu1**2)
        F2 = -np.sum((xl1[1:] - xl1[:-1] ** 2) ** 2 + (xl1[:-1] - 1) ** 2)
        F3 = np.sum(xu2**2) - np.sum((np.abs(xu2) - xl2**2) ** 2)
        return F1 + F2 + F3

    def f_lower(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        f1 = np.sum(xu1**2)
        f2 = np.sum((xl1[1:] - xl1[:-1] ** 2) ** 2 + (xl1[:-1] - 1) ** 2)
        f3 = np.sum((np.abs(xu2) - xl2**2) ** 2)
        return f1 + f2 + f3


class SMD6(SMDBase):
    def __init__(self, dim_xu: int, dim_xl: int) -> None:
        super().__init__(dim_xu, dim_xl, use_s=True)
        xu_bound = np.zeros((2, self.dim_xu))
        xu_bound[0], xu_bound[1] = -5, 10
        self.xu_bound = xu_bound

        xl_bound = np.zeros((2, self.dim_xl))
        xl_bound[0], xl_bound[1] = -5, 10
        self.xl_bound = xl_bound

    def f_upper(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        F1 = np.sum(xu1**2)
        F2 = -np.sum(xl1[: self.q] ** 2) + np.sum(xl1[self.q :] ** 2)
        F3 = np.sum(xu2**2) - np.sum((xu2 - xl2) ** 2)
        return F1 + F2 + F3

    def f_lower(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        f1 = np.sum(xu1**2)
        f2 = np.sum(xl1[: self.q] ** 2) + np.sum(
            (
                xl1[self.q + 1 : self.q + self.s : 2]
                - xl1[self.q : self.q + self.s - 1 : 2]
            )
            ** 2
        )
        f3 = np.sum((xu2 - xl2) ** 2)
        return f1 + f2 + f3


class SMD7(SMDBase):
    def __init__(self, dim_xu: int, dim_xl: int) -> None:
        super().__init__(dim_xu, dim_xl)
        xu_bound = np.zeros((2, self.dim_xu))
        xu_bound[0] = -5
        xu_bound[1, : self.p] = 10
        xu_bound[1, self.p :] = 1
        self.xu_bound = xu_bound

        xl_bound = np.zeros((2, self.dim_xl))
        xl_bound[0, : self.q] = -5
        xl_bound[0, self.q :] = 0
        xl_bound[1, : self.q] = 10
        xl_bound[1, self.q :] = np.e
        self.xl_bound = xl_bound

    def f_upper(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        F1 = (
            1
            + np.sum(xu1**2) / 400.0
            - np.prod(np.cos(xu1 / np.sqrt(np.arange(1, self.p + 1))))
        )
        F2 = -np.sum(xl1**2)
        F3 = np.sum(xu2**2) - np.sum((xu2 - safe_log(xl2)) ** 2)
        return F1 + F2 + F3

    def f_lower(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        f1 = np.sum(xu1**3)
        f2 = np.sum(xl1**2)
        f3 = np.sum((xu2 - safe_log(xl2)) ** 2)
        return f1 + f2 + f3


class SMD8(SMDBase):
    def __init__(self, dim_xu: int, dim_xl: int) -> None:
        super().__init__(dim_xu, dim_xl)
        xu_bound = np.zeros((2, self.dim_xu))
        xu_bound[0], xu_bound[1] = -5, 10
        self.xu_bound = xu_bound

        xl_bound = np.zeros((2, self.dim_xl))
        xl_bound[0], xl_bound[1] = -5, 10
        self.xl_bound = xl_bound

    def f_upper(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        F1 = (
            20
            + np.e
            - 20 * np.exp(-0.2 * np.sqrt(np.sum(xu1**2) / self.p))
            - np.exp(np.sum(np.cos(2 * np.pi * xu1)) / self.p)
        )
        F2 = -np.sum((xl1[1:] - xl1[:-1] ** 2) ** 2 + (xl1[:-1] - 1) ** 2)
        F3 = np.sum(xu2**2) - np.sum((xu2 - xl2**3) ** 2)
        return F1 + F2 + F3

    def f_lower(self, xu: NDArray[float], xl: NDArray[float]) -> float:
        xu1, xu2, xl1, xl2 = self.split_vars(xu, xl)
        f1 = np.sum(np.abs(xu1))
        f2 = np.sum((xl1[1:] - xl1[:-1] ** 2) ** 2 + (xl1[:-1] - 1) ** 2)
        f3 = np.sum((xu2 - xl2**3) ** 2)
        return f1 + f2 + f3


SMD_PROBLEMS = [SMD1, SMD2, SMD3, SMD4, SMD5, SMD6, SMD7, SMD8]
