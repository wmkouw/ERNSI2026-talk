"""Fit stage 5 and write its predictions. `python -m models.moe.run`"""
from ..base import Protocol, product_grid
from ..runner import run_stage
from .model import build

# The bank the two mixture stages share, so stage 6 differs from stage 5 in
# the switch alone and nothing else.
BANK = dict(n_experts=3, alpha_lo=1e4, alpha_hi=1e9, kappa=1.0, omega=0.0)

def main(protocol=None):
    return run_stage("moe", "mixture", 5, build,
                     grid=product_grid(vartheta=[1e1, 1e2, 1e3]),
                     fixed=dict(BANK),
                     protocol=protocol or Protocol())

if __name__ == "__main__":
    main()
