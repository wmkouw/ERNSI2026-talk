"""
Stage 5 --- mixture of experts.

The whole of stage 4 becomes a block M_k and is instantiated K times, each
copy with its own clamped drift precision alpha_k. A selector node picks
which block explains y_t:

    s_t   ~ Cat(pi),   pi clamped to the uniform simplex
    y_t   ~ N(theta_t^(s_t)' x_t, gamma_t^(s_t)^-1)

The mixing weights are fixed, so the switch has no memory: each sample the
responsibilities are recomputed from the branch predictives alone. Each
branch then absorbs the observation with its likelihood tempered by its own
responsibility, which is the ordinary variational rule for a mixture node.

This is the composability payoff: no new inference rules are needed, the
block from the previous slide is simply used as a node.
"""

from __future__ import annotations

import numpy as np

from ..mixture import MixtureBase, default_bank


class MixtureOfExperts(MixtureBase):

    name = "moe"
    label = "mixture"

    def __init__(self, order: int = 4, n_experts: int = 3, omega: float = 0.0,
                 vartheta: float = 1e3, alpha_lo: float = 1e2, alpha_hi: float = 1e8,
                 kappa: float = 1.0):
        super().__init__(default_bank(order, n_experts, omega, vartheta,
                                      alpha_lo, alpha_hi, kappa))
        self._uniform = np.full(self.K, -np.log(self.K))

    def log_prior_weights(self) -> np.ndarray:
        return self._uniform.copy()


def build(order: int = 4, **kw) -> MixtureOfExperts:
    return MixtureOfExperts(order=order, **kw)
