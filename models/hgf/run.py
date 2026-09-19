"""Fit stage 4 and write its predictions. `python -m models.hgf.run`"""
from ..base import Protocol, product_grid
from ..runner import run_stage
from .model import build

def main(protocol=None):
    # kappa is fixed at 1 (it trades off against the scale of z). omega sets
    # the baseline log precision and vartheta how volatile the volatility is.
    return run_stage("hgf", "+ HGF noise", 4, build,
                     grid=product_grid(vartheta=[1e1, 1e2, 1e3, 1e4],
                                       omega=[0.0, 4.0]),
                     fixed=dict(kappa=1.0, alpha_init=1e6, lr=4.0, half_life=2000.0),
                     protocol=protocol or Protocol())

if __name__ == "__main__":
    main()
