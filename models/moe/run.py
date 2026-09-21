"""Fit stage 5 and write its predictions. `python -m models.moe.run`

The GP branch's length scale and signal-to-noise ratio are tuned here, on the
burn-in day, alongside the volatility. They are not inherited from the
standalone Matern run, because the best solo GP and the best complementary
GP are different objects: alone it has to explain the whole signal, in the
bank it only has to explain what the autoregression misses.

The signal-to-noise grid stops at 64 on purpose. The burn-in score is nearly
flat in that direction, and the limit snr -> infinity is degenerate: the
observation noise vanishes, f_t collapses onto y_t, and the Matern-1/2 GP
becomes a plain AR(1) with coefficient exp(-1/ell). Chasing the argmax into
that corner would be reading noise.
"""
from ..base import Protocol, product_grid
from ..runner import run_stage
from .model import build

# What the two mixture stages share, so stage 6 differs from stage 5 in the
# switch alone and nothing else.
BANK = dict(kappa=1.0, omega=0.0, alpha_init=1e6, lr=4.0, half_life=2000.0)


def main(protocol=None):
    return run_stage("moe", "mixture", 5, build,
                     grid=product_grid(vartheta=[1e1, 1e2],
                                       gp_length=[1.0, 2.0, 4.0],
                                       gp_snr=[16.0, 64.0],
                                       resp_floor=[0.02, 0.1, 0.2, 0.3]),
                     fixed=dict(BANK),
                     protocol=protocol or Protocol())

if __name__ == "__main__":
    main()
