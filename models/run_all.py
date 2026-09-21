"""Fit every stage in order and write results/summary.csv.

    python -m models.run_all
"""
from . import base
from .base import Protocol
from .ar import run as ar_run
from .tvar import run as tvar_run
from .htvar import run as htvar_run
from .hgf import run as hgf_run
from .matern import run as matern_run
from .moe import run as moe_run
from .bmoe import run as bmoe_run

# The four rungs, then the Matern GP on its own, then the two mixture stages
# that put the two of them behind a switch. Matern runs before the mixtures
# because they read its chosen length scale. The Transformer baseline is not
# here: it needs PyTorch and 25 minutes.
STAGES = [ar_run, tvar_run, htvar_run, hgf_run, matern_run, moe_run,
          bmoe_run]

def main(protocol=None):
    protocol = protocol or Protocol()
    for mod in STAGES:
        mod.main(protocol)
        print()
    path = base.write_summary()
    print("summary ->", path)
    with open(path) as fh:
        print(fh.read())

if __name__ == "__main__":
    main()
