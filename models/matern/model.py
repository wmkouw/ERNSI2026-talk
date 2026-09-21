"""
A Matern-1/2 Gaussian process in state-space form.

This is the second expert of the mixture, run on its own first so you can see
what it is worth before asking what it adds. A Matern-1/2 GP is an
Ornstein-Uhlenbeck process observed in noise, which is a one-dimensional
linear state-space model, so the Kalman filter is exact inference for it:

    f_t = a f_{t-1} + w_t,   a = exp(-dt/ell),   w_t ~ N(0, gamma^2 (1 - a^2))
    y_t = f_t + e_t,         e_t ~ N(0, gamma^2 / snr)

As a factor graph that is a chain of transition nodes with an equality
fan-out on f_t and a Gaussian observation factor per sample, which is the
construction Nuijten2026 draws for general nu and which collapses to a single
scalar state at nu = 1/2. Nothing about it is autoregressive in the sense the
rest of the talk uses: it has no regressor vector and no coefficients to
identify. It says only that the signal has a correlation time.

The amplitude is tracked by the same volatility chain stage 4 uses, because
without one nothing on this bridge is comparable to anything else.
"""

from __future__ import annotations

import numpy as np

from ..base import OnlineModel, Predictive, gaussian
from ..blocks import MaternHalf, VolatilityChain


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


def build(order: int = 4, **kw) -> MaternGP:
    return MaternGP(order=order, **kw)


# Kept under its old name so callers that import `build_gp` still work.
build_gp = build
