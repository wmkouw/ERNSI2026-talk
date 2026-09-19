"""
Shared machinery for the ERNSI 2026 model ladder.

Everything the six models have in common lives here: loading the bridge
trace, building regressors, the predictive density they all report, the
forecasting metrics, the streaming loop, hyperparameter selection on a
burn-in prefix, and result I/O.

The contract every model obeys
------------------------------
A model is an object with

    predict(x)     -> Predictive          the one-step-ahead density for y_t
                                          given everything before t
    update(x, y)   -> None                absorb the observation

`run_filter` drives that loop once over the trace, so no model ever sees a
sample before it has predicted it. This is what makes the six numbers
comparable.

Units
-----
Everything is in the physical units of the trace (acceleration, m/s^2).
No standardisation, so RMSE is directly readable and NLPD is a density in
those units.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln, logsumexp

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

DEFAULT_TRACE = os.path.join(DATA, "sim_001.csv")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

@dataclass
class Trace:
    """One simulated bridge record, plus the simulator's ground truth."""
    y: np.ndarray              # (N,)   measured acceleration
    t: np.ndarray              # (N,)   seconds
    hour: np.ndarray           # (N,)   scenario hours
    fs: float                  #        sample rate [Hz]
    truth: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)


def load_trace(path: str = DEFAULT_TRACE, column: str = "acc") -> Trace:
    """Read a bridge CSV written by `bridge.save_csv`."""
    raw = np.genfromtxt(path, delimiter=",", names=True)
    names = raw.dtype.names
    t = np.asarray(raw["t"], float)
    fs = 1.0 / np.median(np.diff(t))
    truth_keys = [k for k in ("acc_true", "exc_rms", "T_deck", "damage", "regime",
                              "f1", "f2", "a1", "a2", "a3", "a4", "wind",
                              "traffic_rate") if k in names]
    return Trace(
        y=np.asarray(raw[column], float),
        t=t,
        hour=np.asarray(raw["hour"], float),
        fs=float(fs),
        truth={k: np.asarray(raw[k], float) for k in truth_keys},
    )


def lagged(y: np.ndarray, order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regressors for an AR(`order`) model.

    Returns (X, target, index) with X[i] = [y_{k-1}, ..., y_{k-order}],
    target[i] = y_k and index[i] = k, for k = order .. N-1. The index lets
    a caller line predictions up against the original time axis.
    """
    n = len(y)
    idx = np.arange(order, n)
    X = np.stack([y[idx - (j + 1)] for j in range(order)], axis=1)
    return X, y[idx], idx


# --------------------------------------------------------------------------- #
# Predictive densities
# --------------------------------------------------------------------------- #

@dataclass
class Predictive:
    """A finite mixture of Student-t densities.

    Every model in the ladder reports its one-step-ahead forecast in this
    one form, so a single metric function covers all of them.

    - a Gaussian is `df = inf` with one component
    - the conjugate AR model is one component with finite `df`
    - the mixtures use several components

    `scale` is the Student-t scale, not the standard deviation. For finite
    df the variance is scale^2 * df / (df - 2).
    """
    loc: np.ndarray
    scale: np.ndarray
    df: np.ndarray
    logw: Optional[np.ndarray] = None   # log mixing weights, default uniform

    def __post_init__(self):
        self.loc = np.atleast_1d(np.asarray(self.loc, float))
        self.scale = np.atleast_1d(np.asarray(self.scale, float))
        self.df = np.atleast_1d(np.asarray(self.df, float))
        k = self.loc.size
        if self.logw is None:
            self.logw = np.full(k, -np.log(k))
        else:
            self.logw = np.asarray(self.logw, float)
            self.logw = self.logw - logsumexp(self.logw)

    # -- moments ----------------------------------------------------------
    @property
    def mean(self) -> float:
        return float(np.exp(self.logw) @ self.loc)

    @property
    def var(self) -> float:
        w = np.exp(self.logw)
        v = self.scale ** 2
        fin = np.isfinite(self.df)
        if fin.any():                       # Student-t variance inflation
            nu = self.df[fin]
            v = v.copy()
            v[fin] = v[fin] * np.where(nu > 2, nu / np.maximum(nu - 2, 1e-9), np.inf)
        m = w @ self.loc
        return float(w @ (v + self.loc ** 2) - m ** 2)

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 0.0)))

    # -- density ----------------------------------------------------------
    def logpdf(self, y: float) -> float:
        z = (y - self.loc) / self.scale
        finite = np.isfinite(self.df)
        lp = np.empty_like(self.loc)
        # Gaussian components
        if (~finite).any():
            lp[~finite] = (-0.5 * np.log(2 * np.pi) - np.log(self.scale[~finite])
                           - 0.5 * z[~finite] ** 2)
        # Student-t components
        if finite.any():
            nu = self.df[finite]
            lp[finite] = (gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu)
                          - 0.5 * np.log(np.pi * nu) - np.log(self.scale[finite])
                          - 0.5 * (nu + 1) * np.log1p(z[finite] ** 2 / nu))
        return float(logsumexp(self.logw + lp))


def gaussian(loc: float, var: float) -> Predictive:
    return Predictive(loc=loc, scale=np.sqrt(var), df=np.inf)


def student(loc: float, scale_sq: float, df: float) -> Predictive:
    return Predictive(loc=loc, scale=np.sqrt(scale_sq), df=df)


# --------------------------------------------------------------------------- #
# The streaming loop
# --------------------------------------------------------------------------- #

class OnlineModel:
    """Interface every model in `models/*/model.py` implements."""

    name: str = "model"
    label: str = "model"

    def predict(self, x: np.ndarray) -> Predictive:
        raise NotImplementedError

    def update(self, x: np.ndarray, y: float) -> None:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, np.ndarray]:
        """Optional per-step internals to record (scalars or small vectors)."""
        return {}


@dataclass
class FilterOutput:
    pred_mean: np.ndarray
    pred_std: np.ndarray
    logpdf: np.ndarray
    index: np.ndarray                       # position in the original trace
    diagnostics: Dict[str, np.ndarray] = field(default_factory=dict)


def run_filter(model: OnlineModel, X: np.ndarray, y: np.ndarray,
               index: np.ndarray, record: bool = True) -> FilterOutput:
    """One left-to-right pass: predict, score, then update. No lookahead."""
    n = len(y)
    mu = np.empty(n)
    sd = np.empty(n)
    lp = np.empty(n)
    diag: Dict[str, list] = {}

    for k in range(n):
        p = model.predict(X[k])
        mu[k] = p.mean
        sd[k] = p.std
        lp[k] = p.logpdf(y[k])
        model.update(X[k], y[k])
        if record:
            for key, val in model.diagnostics().items():
                diag.setdefault(key, []).append(np.asarray(val, float).copy())

    return FilterOutput(
        pred_mean=mu, pred_std=sd, logpdf=lp, index=index,
        diagnostics={k: np.asarray(v, dtype=np.float32) for k, v in diag.items()},
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@dataclass
class Protocol:
    """How every model is fitted and scored.

    `order`    AR order; the simulator's true system is AR(4).
    `tune_end` samples 0..tune_end are the burn-in. Hyperparameters are
               chosen on this prefix only, so the scored window is clean.
    `eval_start` metrics are accumulated from here on, which also lets the
               filters forget their priors before being judged.
    """
    order: int = 4
    tune_end: int = 14_400        # day one of the scenario
    eval_start: int = 14_400
    trace: str = DEFAULT_TRACE
    column: str = "acc"

    def slices(self, n: int) -> Tuple[slice, slice]:
        return slice(0, min(self.tune_end, n)), slice(min(self.eval_start, n), n)


def metrics(out: FilterOutput, y: np.ndarray, eval_start: int) -> Dict[str, float]:
    """Forecast quality on the scored window."""
    s = slice(eval_start, len(y))
    err = y[s] - out.pred_mean[s]
    sd = np.maximum(out.pred_std[s], 1e-300)
    z = err / sd
    return dict(
        rmse=float(np.sqrt(np.mean(err ** 2))),
        mae=float(np.mean(np.abs(err))),
        nlpd=float(-np.mean(out.logpdf[s])),
        # calibration: fraction of outcomes inside the central 90% band of a
        # Gaussian with the reported predictive sd (nominal 0.90)
        coverage90=float(np.mean(np.abs(z) < 1.6448536269514722)),
        z_std=float(np.std(z)),
        n=int(len(err)),
    )


def cumulative_curves(out: FilterOutput, y: np.ndarray,
                      eval_start: int, stride: int = 50) -> Dict[str, np.ndarray]:
    """Running RMSE and running mean NLPD over the scored window.

    Subsampled by `stride` so the stored curves stay small; they are
    monotone-ish and plot identically.
    """
    s = slice(eval_start, len(y))
    err = y[s] - out.pred_mean[s]
    lp = out.logpdf[s]
    k = np.arange(1, len(err) + 1)
    run_rmse = np.sqrt(np.cumsum(err ** 2) / k)
    run_nlpd = -np.cumsum(lp) / k
    sel = np.arange(0, len(err), stride)
    return dict(step=out.index[s][sel].astype(np.int32),
                run_rmse=run_rmse[sel].astype(np.float32),
                run_nlpd=run_nlpd[sel].astype(np.float32))


def windowed_curves(out: FilterOutput, y: np.ndarray, hour: np.ndarray,
                    eval_start: int, window: int = 1200) -> Dict[str, np.ndarray]:
    """Block RMSE and block NLPD in non-overlapping windows.

    With the default 1200 samples at 20 Hz each block is one minute of
    vibration, i.e. two scenario hours. These are what show *where* a model
    gains, which the running averages wash out.
    """
    s = slice(eval_start, len(y))
    err = y[s] - out.pred_mean[s]
    lp = out.logpdf[s]
    h = hour[out.index[s]]
    nb = len(err) // window
    err = err[:nb * window].reshape(nb, window)
    lp = lp[:nb * window].reshape(nb, window)
    h = h[:nb * window].reshape(nb, window)
    return dict(win_hour=h.mean(axis=1).astype(np.float32),
                win_rmse=np.sqrt((err ** 2).mean(axis=1)).astype(np.float32),
                win_nlpd=(-lp.mean(axis=1)).astype(np.float32))


# --------------------------------------------------------------------------- #
# Hyperparameter selection on the burn-in
# --------------------------------------------------------------------------- #

def tune(factory: Callable[..., OnlineModel], grid: Sequence[dict],
         X: np.ndarray, y: np.ndarray, index: np.ndarray,
         tune_end: int, score_from: Optional[int] = None,
         verbose: bool = False) -> Tuple[dict, list]:
    """Pick hyperparameters by predictive log-density on the burn-in only.

    Nothing after `tune_end` is touched, so the scored window never informs
    the choice. `score_from` drops the first part of the burn-in from the
    score, so a model is not penalised for its prior.
    """
    if score_from is None:
        score_from = tune_end // 4
    Xb, yb, ib = X[:tune_end], y[:tune_end], index[:tune_end]
    rows = []
    best, best_score = None, -np.inf
    for cfg in grid:
        out = run_filter(factory(**cfg), Xb, yb, ib, record=False)
        score = float(np.mean(out.logpdf[score_from:]))
        rows.append(dict(cfg, mean_logpdf=score))
        if verbose:
            print(f"   {cfg}  ->  mean logpdf {score:+.4f}")
        if score > best_score:
            best, best_score = cfg, score
    return dict(best), rows


def logspace_grid(name: str, lo: float, hi: float, n: int) -> list:
    """`n` values of `name` spaced logarithmically between 10^lo and 10^hi."""
    return [{name: float(v)} for v in np.logspace(lo, hi, n)]


def product_grid(**axes) -> list:
    """Cartesian product of named axes, as a list of config dicts."""
    keys = list(axes)
    out = [{}]
    for k in keys:
        out = [dict(c, **{k: v}) for c in out for v in axes[k]]
    return out


# --------------------------------------------------------------------------- #
# Result I/O
# --------------------------------------------------------------------------- #

def save(name: str, label: str, stage: int, out: FilterOutput, y: np.ndarray,
         trace: Trace, protocol: Protocol, hyper: dict,
         results_dir: str = RESULTS, keep_trace_stride: int = 20) -> str:
    """Write one model's predictions, curves, metrics and settings.

    The full-resolution prediction is subsampled by `keep_trace_stride` for
    storage; the metrics and curves are computed from the full record first,
    so nothing is lost where it matters.
    """
    os.makedirs(results_dir, exist_ok=True)
    m = metrics(out, y, protocol.eval_start)
    payload = dict(
        name=name, label=label, stage=stage,
        index=out.index.astype(np.int32),
        **{k: np.asarray(v) for k, v in cumulative_curves(out, y, protocol.eval_start).items()},
        **{k: np.asarray(v) for k, v in windowed_curves(out, y, trace.hour, protocol.eval_start).items()},
        # a thinned copy of the raw one-step forecast, for trace plots
        thin_index=out.index[::keep_trace_stride].astype(np.int32),
        thin_y=y[::keep_trace_stride].astype(np.float32),
        thin_mean=out.pred_mean[::keep_trace_stride].astype(np.float32),
        thin_std=out.pred_std[::keep_trace_stride].astype(np.float32),
        thin_logpdf=out.logpdf[::keep_trace_stride].astype(np.float32),
    )
    for k, v in out.diagnostics.items():
        payload["diag_" + k] = v[::keep_trace_stride]

    path = os.path.join(results_dir, f"{name}.npz")
    np.savez_compressed(path, **payload)

    meta = dict(name=name, label=label, stage=stage, metrics=m,
                hyper=hyper, protocol=asdict(protocol))
    with open(os.path.join(results_dir, f"{name}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return path


def load_result(name: str, results_dir: str = RESULTS):
    npz = np.load(os.path.join(results_dir, f"{name}.npz"), allow_pickle=False)
    with open(os.path.join(results_dir, f"{name}.json")) as fh:
        meta = json.load(fh)
    return npz, meta


STAGES = [
    ("ar",    1, "AR"),
    ("tvar",  2, "TVAR"),
    ("htvar", 3, "hier. TVAR"),
    ("hgf",   4, "+ HGF noise"),
    ("moe",   5, "mixture"),
    ("bmoe",  6, "Bayesian mixture"),
]


def write_summary(results_dir: str = RESULTS) -> str:
    """Collect every model's metrics into one CSV, ordered by stage."""
    rows = []
    for name, stage, label in STAGES:
        p = os.path.join(results_dir, f"{name}.json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            meta = json.load(fh)
        rows.append(dict(stage=stage, name=name, label=meta["label"], **meta["metrics"]))
    if not rows:
        raise RuntimeError("no model results found; run the models first")
    cols = ["stage", "name", "label", "rmse", "mae", "nlpd", "coverage90", "z_std", "n"]
    path = os.path.join(results_dir, "summary.csv")
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(
                f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Small numerical helpers shared by the models
# --------------------------------------------------------------------------- #

def sym(A: np.ndarray) -> np.ndarray:
    """Force symmetry; covariance updates drift otherwise over 70k steps."""
    return 0.5 * (A + A.T)


def prepare(protocol: Protocol) -> Tuple[Trace, np.ndarray, np.ndarray, np.ndarray]:
    """Load the trace and build the regressors the whole ladder shares."""
    tr = load_trace(protocol.trace, protocol.column)
    X, yt, idx = lagged(tr.y, protocol.order)
    return tr, X, yt, idx
