"""
Stage 3 --- hierarchical time-varying autoregressive coefficients.

    y_t | theta_t, gamma ~ N(theta_t' x_t, gamma^-1)
    theta_t              = theta_{t-1} + w_t,   w_t ~ N(0, alpha^-1 I)
    alpha                ~ Gamma(a_theta, b_theta)
    gamma                ~ Gamma(a_0, b_0)

The clamped process precision of stage 2 becomes a latent variable with its
own prior, so the model learns how fast the coefficients drift instead of
being told. Nothing below that level changes, and there is no longer a
hyperparameter to choose for the drift rate.
"""

from __future__ import annotations

import numpy as np

from ..base import OnlineModel, Predictive, student
from ..blocks import CoefficientChain, NoisePrecision


class HierarchicalTVAR(OnlineModel):

    name = "htvar"
    label = "hier. TVAR"

    def __init__(self, order: int = 4, alpha_init: float = 1e6, lr: float = 1.0,
                 half_life: float = 2000.0,
                 a0: float = 1e-2, b0: float = 1e-6, p0: float = 1.0):
        self.theta = CoefficientChain(order, learn_alpha=True, alpha_init=alpha_init,
                                      lr=lr, half_life=half_life, p0=p0)
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
        return dict(theta=self.theta.m,
                    gamma=np.array([self.gamma.mean]),
                    alpha=np.array([self.theta.alpha]))


def build(order: int = 4, **kw) -> HierarchicalTVAR:
    return HierarchicalTVAR(order=order, **kw)
