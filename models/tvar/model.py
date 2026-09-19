"""
Stage 2 --- time-varying autoregressive coefficients.

    y_t | theta_t, gamma ~ N(theta_t' x_t, gamma^-1)
    theta_t              = theta_{t-1} + w_t,   w_t ~ N(0, alpha_0^-1 I)
    gamma                ~ Gamma(a_0, b_0)

One node changed: the prior on theta became a transition. The walk's
precision alpha_0 is still clamped, and gamma is still a single static
precision inferred by variational message passing. Everything else in the
graph, and the schedule, is what stage 1 had.
"""

from __future__ import annotations

import numpy as np

from ..base import OnlineModel, Predictive, student
from ..blocks import CoefficientChain, NoisePrecision


class TVAR(OnlineModel):

    name = "tvar"
    label = "TVAR"

    def __init__(self, order: int = 4, alpha0: float = 1e4,
                 a0: float = 1e-2, b0: float = 1e-6, p0: float = 1.0):
        self.theta = CoefficientChain(order, alpha0=alpha0, learn_alpha=False, p0=p0)
        self.gamma = NoisePrecision(a0, b0)

    def predict(self, x: np.ndarray) -> Predictive:
        m, _ = self.theta.predict()
        loc = float(m @ x)
        scale_sq = self.gamma.var_estimate + self.theta.pred_var(x)
        return student(loc, scale_sq, self.gamma.df)

    def update(self, x: np.ndarray, y: float) -> None:
        eps = self.theta.update(x, y, noise_var=self.gamma.var_estimate)
        self.gamma.update(eps)

    def diagnostics(self):
        return dict(theta=self.theta.m, gamma=np.array([self.gamma.mean]))


def build(order: int = 4, **kw) -> TVAR:
    return TVAR(order=order, **kw)
