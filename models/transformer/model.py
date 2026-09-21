"""
The modern baseline: a causal Transformer with a probabilistic head.

This is deliberately the thing a 2026 practitioner would reach for. It reads
the last `context` samples, attends over them, and emits a Gaussian for the
next one, so it is scored by exactly the same RMSE and NLPD as the six
factor-graph models.

Three choices are worth naming, because they are where the comparison is
decided.

**Instance normalisation.** Each context is centred and scaled by its own
mean and standard deviation before it reaches the network, and the output is
scaled back. Without it nothing works: the excitation on this bridge spans
three orders of magnitude, a fixed set of weights cannot cover that range,
and training diverges. This is the RevIN trick that every recent forecasting
paper uses, and it is worth seeing it for what it is. It is a hand-designed
volatility model, wired in ahead of the network because the network cannot
learn one. The hierarchical Gaussian filter in stage 4 does the same job,
except it infers the scale instead of being handed it, and it says how
certain it is about it.

**A Gaussian head, not a point.** The last position maps to a mean and a log
standard deviation, and training minimises the Gaussian negative log
likelihood. A model trained on squared error would have no predictive
density and could not be compared on NLPD at all.

**Normalised loss, physical report.** The loss is the NLL in normalised
units, which is better conditioned. Reporting adds back log(scale), so the
number in the results is a density in m/s^2 like everyone else's.

The network is small on purpose: roughly 40k parameters over 14k training
windows. Making it bigger on this much data trains a better fit to day one
and a worse forecast of day four.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:      # pragma: no cover
    raise ImportError(
        "The Transformer baseline needs PyTorch: pip install torch"
    ) from exc


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class TransformerConfig:
    context: int = 96          # samples of history, 4.8 s at 20 Hz
    d_model: int = 48
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 96
    dropout: float = 0.1
    sigma_floor: float = 1e-6  # well below the 5e-4 accelerometer noise floor
    # training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 256
    epochs: int = 30
    patience: int = 5
    val_frac: float = 0.1
    # Consecutive windows overlap in 95 of their 96 samples, so training on
    # every one of them costs time and buys almost nothing. Evaluation still
    # predicts every sample.
    train_stride: int = 2
    val_blocks: int = 5
    grad_clip: float = 1.0
    seed: int = 0


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #

class CausalTransformer(nn.Module):
    """Context of scalars in, one Gaussian out, in normalised units."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.zeros(cfg.context, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, cfg.d_ff, dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)
        nn.init.zeros_(self.head.bias)
        self.register_buffer(
            "mask", torch.triu(torch.ones(cfg.context, cfg.context,
                                          dtype=torch.bool), diagonal=1),
            persistent=False)

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ctx: (B, L) already normalised. Returns (mu, log_sigma), each (B,)."""
        h = self.embed(ctx.unsqueeze(-1)) + self.pos
        h = self.encoder(h, mask=self.mask)
        out = self.head(self.norm(h[:, -1]))
        return out[:, 0], out[:, 1].clamp(-12.0, 12.0)


# --------------------------------------------------------------------------- #
# Windows and instance normalisation
# --------------------------------------------------------------------------- #

def contexts(y: np.ndarray, targets: np.ndarray, context: int):
    """Stack the `context` samples strictly before each target index.

    `targets` are positions in `y`; every one must be at least `context`, so
    no window can reach a sample at or after the one being predicted.
    """
    targets = np.asarray(targets, np.int64)
    if targets.min() < context:
        raise ValueError("a context would run off the start of the trace")
    offs = np.arange(-context, 0, dtype=np.int64)
    idx = targets[:, None] + offs[None, :]
    return y[idx].astype(np.float32), y[targets].astype(np.float32)


def instance_norm(ctx: np.ndarray, eps: float = 1e-10):
    """Centre and scale each context by its own statistics.

    Returns the normalised contexts and the (mean, scale) needed to put a
    prediction back into m/s^2.
    """
    m = ctx.mean(axis=1, keepdims=True)
    s = ctx.std(axis=1, keepdims=True) + eps
    return (ctx - m) / s, m[:, 0], s[:, 0]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _nll(mu, log_sigma, target):
    """Gaussian negative log likelihood in normalised units."""
    return (log_sigma + 0.5 * math.log(2 * math.pi)
            + 0.5 * ((target - mu) * torch.exp(-log_sigma)) ** 2)


def fit(y: np.ndarray, fit_upto: int, cfg: TransformerConfig,
        net: Optional[CausalTransformer] = None, epochs: Optional[int] = None,
        log=print) -> Tuple[CausalTransformer, dict]:
    """Train on `y[:fit_upto]` only.

    The last `val_frac` of that prefix is held out to stop early. Nothing
    after `fit_upto` is touched, so the scored window stays clean. Pass an
    existing `net` to warm-start, which is what the nightly variant does.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    targets = np.arange(cfg.context, fit_upto)
    # Validation is several contiguous blocks spread through the prefix, not
    # one tail block. A random split would leak, because neighbouring windows
    # share 95 of 96 samples; a single tail block would tune the stopping
    # rule on whatever the last hour of day one happened to be doing.
    nb = max(1, cfg.val_blocks)
    blocks = np.array_split(targets, nb * 2)
    va_t = np.concatenate([b[-max(1, int(len(b) * cfg.val_frac * 2)):]
                           for b in blocks[1::2]])
    va_set = np.isin(targets, va_t)
    tr_t = targets[~va_set][::max(1, cfg.train_stride)]

    def tensors(t):
        ctx, tgt = contexts(y, t, cfg.context)
        ctx_n, m, s = instance_norm(ctx)
        tgt_n = (tgt - m) / s
        return (torch.from_numpy(ctx_n), torch.from_numpy(tgt_n.astype(np.float32)))

    Xtr, Ytr = tensors(tr_t)
    Xva, Yva = tensors(va_t)

    net = net or CausalTransformer(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    n_epochs = cfg.epochs if epochs is None else epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1))

    best, best_state, bad = np.inf, None, 0
    history = []
    for ep in range(n_epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        tot, seen = 0.0, 0
        for i in range(0, len(perm), cfg.batch):
            sel = perm[i:i + cfg.batch]
            mu, ls = net(Xtr[sel])
            loss = _nll(mu, ls, Ytr[sel]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            tot += float(loss) * len(sel)
            seen += len(sel)
        sched.step()

        net.eval()
        with torch.no_grad():
            vl = float(_nll(*net(Xva), Yva).mean())
        history.append(dict(epoch=ep, train=tot / seen, val=vl))
        log(f"   epoch {ep:3d}  train {tot / seen:+.4f}  val {vl:+.4f}")

        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"   early stop at epoch {ep}")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, dict(history=history, best_val=best, config=asdict(cfg))


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict(net: CausalTransformer, y: np.ndarray, targets: np.ndarray,
            cfg: TransformerConfig, chunk: int = 2048):
    """One-step-ahead mean and standard deviation, in m/s^2.

    Each target is predicted from the `context` samples strictly before it,
    so this is the same forecasting task the online models were given.
    """
    net.eval()
    targets = np.asarray(targets, np.int64)
    mu_out = np.empty(len(targets))
    sd_out = np.empty(len(targets))
    for i in range(0, len(targets), chunk):
        t = targets[i:i + chunk]
        ctx, _ = contexts(y, t, cfg.context)
        ctx_n, m, s = instance_norm(ctx)
        mu_n, ls_n = net(torch.from_numpy(ctx_n))
        mu_out[i:i + chunk] = m + s * mu_n.numpy()
        sd_out[i:i + chunk] = np.maximum(s * np.exp(ls_n.numpy()), cfg.sigma_floor)
    return mu_out, sd_out


def n_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def build(**kw) -> CausalTransformer:
    """For symmetry with the other stages; the run script does the fitting."""
    return CausalTransformer(TransformerConfig(**kw))
