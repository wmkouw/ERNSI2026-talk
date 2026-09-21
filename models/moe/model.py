"""
Stage 5 --- mixture of experts.

The whole of stage 4 becomes a block and is placed behind a selector node
next to a second block that describes the bridge differently. A Matern-1/2
Gaussian process in state-space form is that second block:

    M_1:  theta_t ~ N(theta_{t-1}, alpha^-1 I),  y_t ~ N(theta_t' x_t, gamma_t^-1)
    M_2:  f_t     ~ N(a f_{t-1}, q),             y_t ~ N(f_t, r gamma_t)
    s_t   ~ Cat(pi),   pi clamped to the uniform simplex

The two branches are different models, not one model at two settings. One
says the deck is a resonant structure whose parameters drift, the other says
it is a smooth process with a correlation time and nothing more. The second
is a bad model of a vibrating deck on purpose: it is first order and
non-oscillatory, so it catches what the autoregression cannot, namely a truck
impulse or the first samples of a regime the coefficients have not reached.

The mixing weights are fixed, so the switch has no memory: each sample the
responsibilities are recomputed from the branch predictives alone. Each
branch then absorbs the observation with its likelihood tempered by its own
responsibility, which is the ordinary variational rule for a mixture node.

This is the composability payoff: no new inference rules are needed, two
blocks from earlier slides are simply used as nodes.
"""

from __future__ import annotations

import numpy as np

from ..mixture import MixtureBase, paired_bank


class MixtureOfExperts(MixtureBase):

    name = "moe"
    label = "mixture"

    def __init__(self, order: int = 4, omega: float = 0.0,
                 vartheta: float = 1e3, kappa: float = 1.0,
                 gp_length: float = 4.0, gp_snr: float = 4.0,
                 alpha_init: float = 1e6, lr: float = 4.0,
                 half_life: float = 2000.0, resp_floor: float = 0.2):
        super().__init__(paired_bank(order, omega, vartheta, gp_length,
                                     gp_snr, kappa, lr, half_life,
                                     alpha_init), resp_floor=resp_floor)
        self._uniform = np.full(self.K, -np.log(self.K))

    def log_prior_weights(self) -> np.ndarray:
        return self._uniform.copy()


def build(order: int = 4, **kw) -> MixtureOfExperts:
    return MixtureOfExperts(order=order, **kw)
