import warnings
import math
import numpy as np


class DdCma:
    def __init__(
        self,
        xmean0,
        sigma0,
        lam=None,
        flg_covariance_update=True,
        flg_variance_update=True,
        flg_active_update=True,
        flg_force_correlation=None,
        beta_eig=None,
        beta_thresh=2.0,
        random_seed=42,
    ):
        """
        Parameters
        ----------
        xmean0 : 1d array-like
            initial mean vector
        sigma0 : 1d array-like
            initial diagonal decoding
        lam : int, optional (default = None)
            population size
        flg_covariance_update : bool, optional (default = True)
            update C if this is True
        flg_variance_update : bool, optional (default = True)
            update D if this is True
        flg_active_update : bool, optional (default = True)
            update C and D with active update
        flg_force_correlation : bool or None, optional (default = None)
            force C to be a correlation matrix if True
            None : flg_force_correlation = flg_variance_update
        beta_eig : float, optional (default = None)
            coefficient to control the frequency of matrix decomposition
        beta_thresh : float, optional (default = 2.)
            threshold parameter for beta control
        """
        self.N = len(xmean0)
        self.chiN = np.sqrt(self.N) * (
            1.0 - 1.0 / (4.0 * self.N) + 1.0 / (21.0 * self.N * self.N)
        )

        # options
        self.flg_covariance_update = flg_covariance_update
        self.flg_variance_update = flg_variance_update
        self.flg_active_update = flg_active_update
        self.flg_force_correlation = (
            flg_variance_update
            if flg_force_correlation is None
            else flg_force_correlation
        )
        self.beta_eig = beta_eig if beta_eig else 10.0 * self.N
        self.beta_thresh = beta_thresh

        # parameters for recombination and step-size adaptation
        self.lam = lam if lam else 4 + int(3 * math.log(self.N))
        assert self.lam > 2
        w = math.log((self.lam + 1) / 2.0) - np.log(np.arange(1, self.lam + 1))
        w[w > 0] /= np.sum(np.abs(w[w > 0]))
        w[w < 0] /= np.sum(np.abs(w[w < 0]))
        self.mueff_positive = 1.0 / np.sum(w[w > 0] ** 2)
        self.mueff_negative = 1.0 / np.sum(w[w < 0] ** 2)
        self.cm = 1.0
        self.cs = (self.mueff_positive + 2.0) / (self.N + self.mueff_positive + 5.0)
        self.ds = (
            1.0
            + self.cs
            + 2.0
            * max(0.0, math.sqrt((self.mueff_positive - 1.0) / (self.N + 1.0)) - 1.0)
        )

        # parameters for covariance matrix adaptation
        expo = 0.75
        mu_prime = (
            self.mueff_positive
            + 1.0 / self.mueff_positive
            - 2.0
            + self.lam / (2.0 * self.lam + 10.0)
        )
        m = self.N * (self.N + 1) / 2
        self.cone = 1.0 / (
            2 * (m / self.N + 1.0) * (self.N + 1.0) ** expo + self.mueff_positive / 2.0
        )
        self.cmu = min(1.0 - self.cone, mu_prime * self.cone)
        self.cc = math.sqrt(self.mueff_positive * self.cone) / 2.0
        self.w = np.array(w)
        self.w[w < 0] *= min(
            1.0 + self.cone / self.cmu,
            1.0 + 2.0 * self.mueff_negative / (self.mueff_positive + 2.0),
        )

        # parameters for diagonal decoding
        m = self.N
        self.cdone = 1.0 / (
            2 * (m / self.N + 1.0) * (self.N + 1.0) ** expo + self.mueff_positive / 2.0
        )
        self.cdmu = min(1.0 - self.cdone, mu_prime * self.cdone)
        self.cdc = math.sqrt(self.mueff_positive * self.cdone) / 2.0
        self.wd = np.array(w)
        self.wd[w < 0] *= min(
            1.0 + self.cdone / self.cdmu,
            1.0 + 2.0 * self.mueff_negative / (self.mueff_positive + 2.0),
        )

        # dynamic parameters
        self.xmean = np.array(xmean0)
        self.D = np.array(sigma0)
        self.sigma = 1.0
        self.C = np.eye(self.N)
        self.S = np.ones(self.N)
        self.B = np.eye(self.N)
        self.sqrtC = np.eye(self.N)
        self.invsqrtC = np.eye(self.N)
        self.Z = np.zeros((self.N, self.N))
        self.pc = np.zeros(self.N)
        self.pdc = np.zeros(self.N)
        self.ps = np.zeros(self.N)
        self.pc_factor = 0.0
        self.pdc_factor = 0.0
        self.ps_factor = 0.0

        # others
        self.teig = max(1, int(1.0 / (self.beta_eig * (self.cone + self.cmu))))
        self.neval = 0
        self.t = 0
        self.beta = 1.0

        # strage for checker and logger
        self.arf = np.zeros(self.lam)
        self.arx = np.zeros((self.lam, self.N))

        # rng for reproducibility
        self.rng = np.random.default_rng(seed=random_seed)

    def transform(self, z):
        y = np.dot(z, self.sqrtC) if self.flg_covariance_update else z
        return y * (self.D * self.sigma)

    def transform_inverse(self, y):
        z = y / (self.D * self.sigma)
        return np.dot(z, self.invsqrtC) if self.flg_covariance_update else z

    def sample(self):
        arz = self.rng.standard_normal((self.lam, self.N))
        ary = np.dot(arz, self.sqrtC) if self.flg_covariance_update else arz
        arx = ary * (self.D * self.sigma) + self.xmean
        return arx, ary, arz

    def update(self, idx, arx, ary, arz):
        # shortcut
        w = self.w
        wc = self.w
        wd = self.wd
        sarz = arz[idx]
        sary = ary[idx]

        # recombination
        dz = np.dot(w[w > 0], sarz[w > 0])
        dy = np.dot(w[w > 0], sary[w > 0])
        self.xmean += self.cm * self.sigma * self.D * dy

        # step-size adaptation
        self.ps_factor = (1 - self.cs) ** 2 * self.ps_factor + self.cs * (2 - self.cs)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(
            self.cs * (2 - self.cs) * self.mueff_positive
        ) * dz
        normsquared = np.sum(self.ps * self.ps)
        hsig = normsquared / self.ps_factor / self.N < 2.0 + 4.0 / (self.N + 1)
        self.sigma *= math.exp(
            (math.sqrt(normsquared) / self.chiN - math.sqrt(self.ps_factor))
            * self.cs
            / self.ds
        )

        # C (intermediate) update
        if self.flg_covariance_update:
            # Rank-mu
            if self.cmu == 0:
                rank_mu = 0.0
            elif self.flg_active_update:
                rank_mu = np.dot(sarz[wc > 0].T * wc[wc > 0], sarz[wc > 0]) - np.sum(
                    wc[wc > 0]
                ) * np.eye(self.N)
                rank_mu += np.dot(
                    sarz[wc < 0].T
                    * (wc[wc < 0] * self.N / np.linalg.norm(sarz[wc < 0], axis=1) ** 2),
                    sarz[wc < 0],
                ) - np.sum(wc[wc < 0]) * np.eye(self.N)
            else:
                rank_mu = np.dot(sarz[wc > 0].T * wc[wc > 0], sarz[wc > 0]) - np.sum(
                    wc[wc > 0]
                ) * np.eye(self.N)
            # Rank-one
            if self.cone == 0:
                rank_one = 0.0
            else:
                self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(
                    self.cc * (2 - self.cc) * self.mueff_positive
                ) * self.D * dy
                self.pc_factor = (
                    1 - self.cc
                ) ** 2 * self.pc_factor + hsig * self.cc * (2 - self.cc)
                zpc = np.dot(self.pc / self.D, self.invsqrtC)
                rank_one = np.outer(zpc, zpc) - self.pc_factor * np.eye(self.N)
            # Update
            self.Z += self.cmu * rank_mu + self.cone * rank_one

        # D update
        if self.flg_variance_update:
            # Cumulation
            self.pdc = (1 - self.cdc) * self.pdc + hsig * math.sqrt(
                self.cdc * (2 - self.cdc) * self.mueff_positive
            ) * self.D * dy
            self.pdc_factor = (
                1 - self.cdc
            ) ** 2 * self.pdc_factor + hsig * self.cdc * (2 - self.cdc)
            DD = self.cdone * (
                np.dot(self.pdc / self.D, self.invsqrtC) ** 2 - self.pdc_factor
            )
            if self.flg_active_update:
                # positive and negative update
                DD += self.cdmu * np.dot(wd[wd > 0], sarz[wd > 0] ** 2)
                DD += self.cdmu * np.dot(
                    wd[wd < 0] * self.N / np.linalg.norm(sarz[wd < 0], axis=1) ** 2,
                    sarz[wd < 0] ** 2,
                )
                DD -= self.cdmu * np.sum(wd)
            else:
                # positive update
                DD += self.cdmu * np.dot(wd[wd > 0], sarz[wd > 0] ** 2)
                DD -= self.cdmu * np.sum(wd[wd > 0])
            if self.flg_covariance_update:
                self.beta = 1 / max(
                    1, np.max(self.S) / np.min(self.S) - self.beta_thresh + 1.0
                )
            else:
                self.beta = 1.0
            self.D *= np.exp((self.beta / 2) * DD)

        # update C
        if self.flg_covariance_update and (self.t + 1) % self.teig == 0:
            D = np.linalg.eigvalsh(self.Z)
            fac = min(0.75 / abs(D.min()), 1.0)
            self.C = np.dot(
                np.dot(self.sqrtC, np.eye(self.N) + fac * self.Z), self.sqrtC
            )

            # force C to be correlation matrix
            if self.flg_force_correlation:
                cd = np.sqrt(np.diag(self.C))
                self.D *= cd
                self.C = (self.C / cd).T / cd

            # decomposition
            DD, self.B = np.linalg.eigh(self.C)
            self.S = np.sqrt(DD)
            self.sqrtC = np.dot(self.B * self.S, self.B.T)
            self.invsqrtC = np.dot(self.B / self.S, self.B.T)
            self.Z[:, :] = 0.0

    def onestep(self, func):
        """
        Parameter
        ---------
        func : callable
            parameter : 2d array-like with candidate solutions (x) as elements
            return    : 1d array-like with f(x) as elements
        """
        # sampling
        arx, ary, arz = self.sample()

        # evaluation
        arf = func(arx)
        self.neval += len(arf)

        # sort
        idx = np.argsort(arf)
        if not np.all(arf[idx[1:]] - arf[idx[:-1]] > 0.0):
            warnings.warn("assumed no tie, but there exists", RuntimeWarning)

        # update
        self.update(idx, arx, ary, arz)

        # finalize
        self.t += 1
        self.arf = arf
        self.arx = arx

    def upper_bounding_coordinate_std(self, coordinate_length):
        """Upper-bounding coordinate-wise standard deviation

        When some design variables are periodic, the coordinate-wise standard deviation
        should be upper-bounded by r_i / 4, where r_i is the period of the ith variable.
        The correction of the overall covariance matrix, Sigma, is done as follows:
            Sigma = Correction * Sigma * Correction,
        where Correction is a diagonal matrix defined as
            Correction_i = min( r_i / (4 * Sigma_{i,i}^{1/2}), 1 ).

        In DD-CMA, the correction matrix is simply multiplied to D.

        For example, if a mirroring box constraint handling is used for a box constraint
        [l_i, u_i], the variables become periodic on [l_i - (u_i-l_i)/2, u_i + (u_i-l_i)/2].
        Therefore, the period is
            r_i = 2 * (u_i - l_i).

        Parameters
        ----------
        coordinate_length : ndarray (1D) or float
            coordinate-wise search length r_i.
        """
        correction = np.fmin(coordinate_length / self.coordinate_std / 4.0, 1)
        self.D *= correction

    @property
    def coordinate_std(self):
        if self.flg_covariance_update:
            return self.sigma * self.D * np.sqrt(np.diag(self.C))
        else:
            return self.sigma * self.D


def mirror(z, lbound, ubound, flg_periodic):
    """Mirroring Box-Constraint Handling and Periodic Constraint Handling

    Parameters
    ----------
    z : ndarray (1D or 2D)
        solutions to be corrected
    lbound, ubound : ndarray (1D)
        lower and upper bounds
        If some variables are not bounded, set np.inf or -np.inf
    flg_periodic : ndarray (1D, bool)
        flag for periodic variables

    Returns
    -------
    projected solution in [lbound, ubound]
    """
    zz = np.copy(z)
    flg_lower = np.isfinite(lbound) * np.logical_not(np.isfinite(ubound) + flg_periodic)
    flg_upper = np.isfinite(ubound) * np.logical_not(np.isfinite(lbound) + flg_periodic)
    flg_box = np.isfinite(lbound) * np.isfinite(ubound) * np.logical_not(flg_periodic)
    width = ubound - lbound
    if zz.ndim == 1:
        zz[flg_periodic] = lbound[flg_periodic] + np.mod(
            zz[flg_periodic] - lbound[flg_periodic], width[flg_periodic]
        )
        zz[flg_lower] = lbound[flg_lower] + np.abs(zz[flg_lower] - lbound[flg_lower])
        zz[flg_upper] = ubound[flg_upper] - np.abs(zz[flg_upper] - ubound[flg_upper])
        zz[flg_box] = ubound[flg_box] - np.abs(
            np.mod(zz[flg_box] - lbound[flg_box], 2 * width[flg_box]) - width[flg_box]
        )
    elif zz.ndim == 2:
        zz[:, flg_periodic] = lbound[flg_periodic] + np.mod(
            zz[:, flg_periodic] - lbound[flg_periodic], width[flg_periodic]
        )
        zz[:, flg_lower] = lbound[flg_lower] + np.abs(
            zz[:, flg_lower] - lbound[flg_lower]
        )
        zz[:, flg_upper] = ubound[flg_upper] - np.abs(
            zz[:, flg_upper] - ubound[flg_upper]
        )
        zz[:, flg_box] = ubound[flg_box] - np.abs(
            np.mod(zz[:, flg_box] - lbound[flg_box], 2 * width[flg_box])
            - width[flg_box]
        )
    return zz
