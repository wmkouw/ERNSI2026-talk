"""
Stage 1 --- autoregressive model with static coefficients.

    y_t | theta, gamma ~ N(theta' x_t, gamma^-1)
    theta | gamma      ~ N(m_0, (gamma Lambda_0)^-1)
    gamma              ~ Gamma(a_0, b_0)

The normal-gamma prior is conjugate, so this graph needs no approximation
at all: the sweep is the exact recursive posterior and the one-step-ahead
predictive is a Student-t in closed form. It is the baseline the rest of
the ladder has to beat.
"""

from __future__ import annotations

import numpy as np

from ..base import OnlineModel, Predictive, student, sym


class StaticAR(OnlineModel):

    name = "ar"
    label = "AR"

    def __init__(self, order: int = 4, lam0: float = 1e-2,
                 a0: float = 1e-2, b0: float = 1e-6):
        self.d = order
        self.m = np.zeros(order)
        self.Lam = np.eye(order) * lam0
        self.a = float(a0)
        self.b = float(b0)
        self._Linv = np.linalg.inv(self.Lam)

    # -- forecast ---------------------------------------------------------
    def predict(self, x: np.ndarray) -> Predictive:
        loc = float(self.m @ x)
        scale_sq = (self.b / self.a) * (1.0 + float(x @ self._Linv @ x))
        return student(loc, scale_sq, 2.0 * self.a)

    # -- exact conjugate update ------------------------------------------
    def update(self, x: np.ndarray, y: float) -> None:
        old_quad = float(self.m @ self.Lam @ self.m)
        rhs = self.Lam @ self.m + x * y
        self.Lam = sym(self.Lam + np.outer(x, x))
        self._Linv = np.linalg.inv(self.Lam)
        self.m = self._Linv @ rhs
        new_quad = float(self.m @ self.Lam @ self.m)
        self.a += 0.5
        self.b += 0.5 * max(y * y + old_quad - new_quad, 0.0)

    # -- internals worth plotting ----------------------------------------
    def diagnostics(self):
        return dict(theta=self.m, gamma=np.array([self.a / self.b]))


def build(order: int = 4, **kw) -> StaticAR:
    return StaticAR(order=order, **kw)
