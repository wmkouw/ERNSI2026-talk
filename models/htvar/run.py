"""Fit stage 3 and write its predictions. `python -m models.htvar.run`"""
from ..base import Protocol, product_grid
from ..runner import run_stage
from .model import build

def main(protocol=None):
    # alpha is now inferred, so the drift rate itself is no longer a setting.
    # What remains is how fast its estimate is allowed to move, and the
    # Robbins-Monro decay that makes it settle on one value, as drawn.
    return run_stage("htvar", "hier. TVAR", 3, build,
                     grid=product_grid(lr=[1.0, 4.0, 16.0],
                                       half_life=[2000.0]),
                     fixed=dict(alpha_init=1e6, a0=1e-2, b0=1e-6),
                     protocol=protocol or Protocol())

if __name__ == "__main__":
    main()
