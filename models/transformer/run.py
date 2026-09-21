"""
Fit the Transformer baseline and write its predictions.

    python -m models.transformer.run              # trained once on day one
    python -m models.transformer.run --nightly    # also the retrained variant

Two variants, because the interesting question is not whether a Transformer
can fit this signal. It can. The question is what happens over the following
four days, while the deck freezes and then cracks.

`transformer`
    Trained on day one, then frozen. This is the ordinary train-once,
    deploy setting, and it is the same information the six online models
    had before the scored window opened.

`transformer-nightly`
    Warm-started and retrained at the end of every day on everything seen so
    far. This is what an operator would actually do, and it is the fairer
    comparison: now the Transformer sees the same data the online models saw,
    just in batches and a day late rather than one sample at a time.

Both are scored by `models.base.metrics` on exactly the window and with
exactly the code the factor-graph models use, so the numbers land in the
same `results/summary.csv`.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .. import base
from ..base import FilterOutput, Protocol, lagged, load_trace, metrics, save
from .model import (TransformerConfig, contexts, fit, n_parameters, predict)


LOG2PI = float(np.log(2.0 * np.pi))


def _gaussian_logpdf(y, mu, sd):
    z = (y - mu) / sd
    return -0.5 * LOG2PI - np.log(sd) - 0.5 * z * z


def _as_output(tr, protocol, mu, sd, first):
    """Pack predictions into the same shape the online models produce.

    `first` is the first trace position the network could predict, given its
    context length. Earlier positions are filled with the trivial forecast
    N(0, s0^2) with s0 measured on the burn-in; they sit far outside the
    scored window and exist only so the arrays line up.
    """
    _, y, index = lagged(tr.y, protocol.order)
    mean = np.zeros(len(index))
    std = np.full(len(index), float(np.std(tr.y[:protocol.tune_end])))
    have = index >= first
    mean[have] = mu
    std[have] = sd
    lp = _gaussian_logpdf(y, mean, std)
    return FilterOutput(pred_mean=mean, pred_std=std, logpdf=lp, index=index)


def _report(tag, out, y, protocol, t0):
    m = metrics(out, y, protocol.eval_start)
    print(f"[{tag}] rmse {m['rmse']:.6g}   nlpd {m['nlpd']:.4f}   "
          f"cov90 {m['coverage90']:.3f}   ({time.time() - t0:.0f}s)")
    return m


def main(protocol=None, nightly=False, cfg=None, verbose=True):
    protocol = protocol or Protocol()
    cfg = cfg or TransformerConfig()
    log = print if verbose else (lambda *a, **k: None)

    tr = load_trace(protocol.trace, protocol.column)
    _, y_lag, index = lagged(tr.y, protocol.order)
    n = len(tr.y)
    first = cfg.context
    targets = np.arange(first, n)

    log(f"[transformer] {n} samples, context {cfg.context}, "
        f"train on 0..{protocol.tune_end}, scored from {protocol.eval_start}")

    # ---------------------------------------------------- trained once
    t0 = time.time()
    net, hist = fit(tr.y, protocol.tune_end, cfg, log=log)
    log(f"[transformer] {n_parameters(net)} parameters, "
        f"trained in {time.time() - t0:.0f}s, best val NLL {hist['best_val']:+.4f}")

    mu, sd = predict(net, tr.y, targets, cfg)
    out = _as_output(tr, protocol, mu, sd, first)
    m_once = _report("transformer", out, y_lag, protocol, t0)
    save("transformer", "Transformer", 8, out, y_lag, tr, protocol,
         dict(variant="trained once on day one", **hist["config"],
              n_parameters=n_parameters(net), best_val_nll=hist["best_val"]))

    if not nightly:
        return dict(transformer=m_once)

    # ------------------------------------------- retrained every night
    t0 = time.time()
    day = protocol.tune_end                       # samples per scenario day
    mu_n = np.empty(len(targets))
    sd_n = np.empty(len(targets))

    # day one is predicted by the day-one model, as before
    seg = targets < day
    mu_n[seg], sd_n[seg] = predict(net, tr.y, targets[seg], cfg)

    live = net
    boundary = day
    while boundary < n:
        stop = min(boundary + day, n)
        log(f"[nightly] refit on 0..{boundary}, then predict {boundary}..{stop}")
        live, h = fit(tr.y, boundary, cfg, net=live,
                      epochs=max(4, cfg.epochs // 4), log=log)
        seg = (targets >= boundary) & (targets < stop)
        mu_n[seg], sd_n[seg] = predict(live, tr.y, targets[seg], cfg)
        boundary = stop

    out_n = _as_output(tr, protocol, mu_n, sd_n, first)
    m_night = _report("transformer-nightly", out_n, y_lag, protocol, t0)
    save("transformer_nightly", "Transformer (nightly)", 9, out_n, y_lag, tr,
         protocol, dict(variant="warm-started refit at the end of every day",
                        **hist["config"], n_parameters=n_parameters(net)))

    return dict(transformer=m_once, transformer_nightly=m_night)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nightly", action="store_true",
                    help="also fit the variant retrained at each day boundary")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--context", type=int, default=None)
    args = ap.parse_args()
    c = TransformerConfig()
    if args.epochs:
        c.epochs = args.epochs
    if args.context:
        c.context = args.context
    main(nightly=args.nightly, cfg=c)
    base.write_summary()
