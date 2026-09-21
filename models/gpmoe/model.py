"""
A Matern-1/2 Gaussian process, alone and inside the mixture.

`MaternGP` is the GP on its own, so you can see what it is worth by itself
before asking what it adds to a bank of autoregressions. It is an
Ornstein-Uhlenbeck process observed in noise, which is a one-dimensional
linear state-space model, so the Kalman filter is exact inference for it. The
amplitude is tracked by the same volatility chain stage 4 uses, because
without one nothing on this bridge is comparable to anything else.

`GPMixtureOfExperts` is stage 6 with that GP added to the bank as one more
branch. The switch, its Dirichlet transition, the other experts and every
setting are identical, so the difference between this and `bmoe` is the
branch and nothing else.
"""

from __future__ import annotations

import numpy as np

from ..base import OnlineModel, Predictive, gaussian
from ..blocks import MaternHalf, VolatilityChain
from ..mixture import bank_with_gp
from ..bmoe.model import BayesianMixtureOfExperts


class MaternGP(OnlineModel):
    """y_t = f_t + e_t, with f a Matern-1/2 GP and a tracked noise scale."""

    name = "matern"
    label = "Matern-1/2 GP"

    def __init__(self, order: int = 4, length: float = 4.0, snr: float = 4.0,
                 omega: float = 0.0, vartheta: float = 1e2, kappa: float = 1.0,
                 p0: float = 1.0, sigma0: float = 1.0):
        # `order` is accepted and ignored: this model has no regressors, it
        # reads the signal directly. It is kept so every stage builds the same.
        self.snr = float(snr)
        self.r = 1.0 / float(snr)
        self.gp = MaternHalf(length, p0=p0)
        self.z = VolatilityChain(kappa=kappa, omega=omega, theta=vartheta,
                                 mu0=0.0, sigma0=sigma0)
        self._scale = 1.0
        self._mu_u = 0.0
        self._var_u = 1.0

    def predict(self, x: np.ndarray) -> Predictive:
        # The GP runs in normalised units; the volatility chain carries the
        # scale. See models/mixture.py:GPExpert for why the two must be
        # separated rather than tied together.
        self.z.predict()
        self._scale = float(np.sqrt(max(self.z.expected_variance(), 1e-300)))
        m, P = self.gp.predict(stationary_var=1.0)
        self._mu_u, self._var_u = m, P + self.r
        return gaussian(self._scale * m, self._scale ** 2 * self._var_u)

    def update(self, x: np.ndarray, y: float) -> None:
        s = self._scale
        self.gp.update(y / s, obs_var=self.r)
        eps = (y - s * self._mu_u) ** 2 / max(self._var_u, 1e-300)
        self.z.update(eps)

    def diagnostics(self):
        return dict(f=np.array([self.gp.m * self._scale]),
                    z=np.array([self.z.mu]))


class GPMixtureOfExperts(BayesianMixtureOfExperts):
    """Stage 6, with a Matern-1/2 GP as one extra branch."""

    name = "bmoe_gp"
    label = "Bayes. mixture + GP"

    def __init__(self, order: int = 4, n_experts: int = 3, omega: float = 0.0,
                 vartheta: float = 1e1, alpha_lo: float = 1e4,
                 alpha_hi: float = 1e9, kappa: float = 1.0,
                 gp_length: float = 4.0, gp_snr: float = 4.0,
                 c_off: float = 1.0, c_diag: float = 50.0):
        super().__init__(
            order=order, n_experts=n_experts, omega=omega, vartheta=vartheta,
            alpha_lo=alpha_lo, alpha_hi=alpha_hi, kappa=kappa,
            c_off=c_off, c_diag=c_diag,
            experts=bank_with_gp(order, n_experts, omega, vartheta, alpha_lo,
                                 alpha_hi, gp_length, gp_snr, kappa))


def build_gp(order: int = 4, **kw) -> MaternGP:
    return MaternGP(order=order, **kw)


def build(order: int = 4, **kw) -> GPMixtureOfExperts:
    return GPMixtureOfExperts(order=order, **kw)
