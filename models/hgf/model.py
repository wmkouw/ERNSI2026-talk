"""
Stage 4 --- a hierarchical Gaussian filter on the observation precision.

    y_t | theta_t, z_t ~ N(theta_t' x_t, exp(-(kappa z_t + omega)))
    theta_t            = theta_{t-1} + w_t,   w_t ~ N(0, alpha^-1 I)
    alpha              ~ Gamma(a_theta, b_theta)
    z_t                = z_{t-1} + v_t,       v_t ~ N(0, vartheta^-1)

The static Gamma of stages 1 to 3 is replaced by a volatility chain, so the
noise level itself is tracked. On this bridge the excitation intensity spans
three orders of magnitude between a quiet night and a storm, which a single
gamma cannot represent at all: this is where the predictive density, rather
than the point forecast, improves.

kappa is fixed at 1 because it trades off against the scale of z. omega is
the baseline log precision and vartheta the volatility of the volatility;
both are chosen on the burn-in prefix.
"""

from __future__ import annotations

import numpy as np

from ..base import OnlineModel, Predictive, gaussian
from ..blocks import CoefficientChain, VolatilityChain


class VolatileHierarchicalAR(OnlineModel):

    name = "hgf"
    label = "+ HGF noise"

    def __init__(self, order: int = 4, omega: float = 0.0, vartheta: float = 1e3,
                 kappa: float = 1.0, alpha_init: float = 1e6, lr: float = 4.0,
                 half_life: float = 2000.0, p0: float = 1.0, sigma0: float = 1.0):
        self.theta = CoefficientChain(order, learn_alpha=True, alpha_init=alpha_init,
                                      lr=lr, half_life=half_life, p0=p0)
        self.z = VolatilityChain(kappa=kappa, omega=omega, theta=vartheta,
                                 mu0=0.0, sigma0=sigma0)

    def predict(self, x: np.ndarray) -> Predictive:
        m, _ = self.theta.predict()
        self.z.predict()
        loc = float(m @ x)
        var = self.theta.pred_var(x) + self.z.expected_variance()
        return gaussian(loc, var)

    def update(self, x: np.ndarray, y: float) -> None:
        noise_var = 1.0 / self.z.expected_precision()
        eps = self.theta.update(x, y, noise_var=noise_var)
        self.z.update(eps)

    def diagnostics(self):
        return dict(theta=self.theta.m,
                    z=np.array([self.z.mu]),
                    gamma=np.array([self.z.expected_precision()]),
                    alpha=np.array([self.theta.alpha]))


def build(order: int = 4, **kw) -> VolatileHierarchicalAR:
    return VolatileHierarchicalAR(order=order, **kw)
