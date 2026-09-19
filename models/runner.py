"""
One entry point for fitting a stage and writing its results.

Each `models/<stage>/run.py` is a three-line wrapper around `run_stage`, so
the protocol, the tuning rule and the output format cannot drift apart
between stages. That is the whole point of comparing them.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import numpy as np

from . import base
from .base import Protocol, prepare, run_filter, save, tune


def run_stage(name: str, label: str, stage: int,
              factory: Callable[..., base.OnlineModel],
              grid: Optional[Sequence[dict]] = None,
              fixed: Optional[dict] = None,
              protocol: Optional[Protocol] = None,
              results_dir: str = base.RESULTS,
              verbose: bool = True) -> dict:
    """Tune on the burn-in, filter the whole trace, write predictions.

    `fixed` holds settings that are not searched over. `grid` is a list of
    config dicts; the one with the highest mean predictive log-density on
    the burn-in wins. With no grid the model is run as configured.
    """
    protocol = protocol or Protocol()
    fixed = dict(fixed or {})
    fixed.setdefault("order", protocol.order)

    trace, X, y, idx = prepare(protocol)
    if verbose:
        print(f"[{name}] {len(y)} samples, AR({protocol.order}), "
              f"burn-in {protocol.tune_end}, scored from {protocol.eval_start}")

    chosen, table = {}, []
    if grid:
        t0 = time.time()
        chosen, table = tune(lambda **kw: factory(**fixed, **kw), grid,
                             X, y, idx, protocol.tune_end, verbose=verbose)
        if verbose:
            print(f"[{name}] tuned in {time.time() - t0:.1f}s -> {chosen}")

    cfg = dict(fixed, **chosen)
    t0 = time.time()
    out = run_filter(factory(**cfg), X, y, idx)
    if verbose:
        print(f"[{name}] filtered in {time.time() - t0:.1f}s")

    hyper = dict(cfg)
    hyper["_search"] = table
    path = save(name, label, stage, out, y, trace, protocol, hyper,
                results_dir=results_dir)
    m = base.metrics(out, y, protocol.eval_start)
    if verbose:
        print(f"[{name}] rmse {m['rmse']:.6g}   nlpd {m['nlpd']:.4f}   "
              f"cov90 {m['coverage90']:.3f}   -> {path}")
    return m
