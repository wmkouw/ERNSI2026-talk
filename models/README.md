# models

The six models from the keynote's factor graphs, applied to the bridge trace
in `../data/sim_001.csv`, with their one-step-ahead forecasts written to
`../results/`.

```
models/
  base.py       data loading, regressors, the predictive density, the
                streaming loop, the metrics, hyperparameter selection, I/O
  blocks.py     the reusable bands of the graph: the coefficient chain, the
                static noise precision, the volatility chain
  mixture.py    the two experts treated as nodes, plus what the two mixture
                stages share
  runner.py     one entry point so every stage is fitted the same way
  run_all.py    fit all six and write results/summary.csv
  ar/ tvar/ htvar/ hgf/ moe/ bmoe/
                one folder per stage: model.py has the model, run.py fits it
  matern/       a Matern-1/2 GP in state-space form, scored on its own; it is
                the second expert of stages 5 and 6
  decision.py   the closing slide: one hour, one threshold, mu + sigma
  transformer/  the modern baseline, outside the ladder; needs PyTorch
```

Run everything with

```
cd ERNSI2026-talk
python -m models.run_all
```

which fits the six rungs plus the standalone Matern reference, or one stage with `python -m models.hgf.run`. The Transformer baseline is
separate, because it needs PyTorch and about twenty-five minutes on two cores:

```
pip install torch
python -m models.transformer.run --nightly
```
 Then `python bridge_demo.py`
redraws the slide figures into `../figures/`, or `marimo edit bridge_demo.py`
to work on them interactively. The notebook's live cells call `build(...)` and
`run_filter` directly on a chosen window, warming the filter up for a few hours
first, which is the quickest way to see what a model is actually doing.

## Why NumPy and not RxInfer

Three of the six need a node whose message is not off the shelf as used here
(the exponential link into the observation precision with a tempering weight,
the switch with Dirichlet-distributed transition rows, and the free-energy
gradient on the drift precision described below). The recursions are short, so
they are written out directly. Every update is still the message the
corresponding node emits; nothing is fitted by a generic optimiser.

## The evaluation

The same protocol for all six, in `base.Protocol`:

- AR order 4, which is the order the simulator's two modes actually imply
- day one of the scenario (14 400 samples) is the burn-in; hyperparameters are
  chosen there, by mean predictive log-density, and nothing later is touched
- days two to five (57 596 samples) are scored
- strictly one-step-ahead: `run_filter` calls `predict` before `update`, so no
  model sees a sample before forecasting it
- everything in the trace's own units, so RMSE is in m/s^2 and NLPD is a
  density in those units

Reported: RMSE, MAE, mean negative log predictive density, and the share of
outcomes inside the central 90% band of the reported predictive (nominal 0.90).

## Results

| stage | model | RMSE [mm/s²] | NLPD [nats] | 90% coverage |
|---|---|---|---|---|
| 1 | AR | 30.0 | -2.081 | 0.963 |
| 2 | TVAR | 30.0 | -2.081 | 0.963 |
| 3 | hier. TVAR | 50.2 | -2.784 | 0.928 |
| 4 | + HGF noise | 34.6 | -3.620 | 0.887 |
| 5 | mixture | 50.7 | -3.665 | 0.983 |
| 6 | Bayesian mixture | 37.4 | -3.756 | 0.977 |
| - | Matern-1/2 GP alone | 88.4 | -2.752 | 0.904 |
| - | Transformer | 57.2 | -3.109 | 0.865 |
| - | Transformer (nightly) | 50.0 | -3.355 | 0.909 |

The predictive density improves at every rung. The point forecast does not,
and the two places it dips are the interesting ones.

**Stage 2 reproduces stage 1.** Tuned on a quiet first day, the drift
precision runs to the static end of its grid. The coefficients genuinely do
move later in the week, when the deck freezes and then cracks, but a value
clamped before that happens cannot anticipate it. This is the argument for
stage 3 stated as a measurement.

**Stage 3 buys density and spends accuracy.** Inferring the drift rate is
worth 0.70 nats and costs 20 mm/s² of RMSE. With no volatility model the only
way the coefficients can account for a loud hour is to move, so they move, and
chase the excitation. The extra freedom is being spent on the wrong job.

**Stage 4 takes that job back.** Tracking the noise is worth another 0.84 nats
and returns most of the RMSE. The excitation intensity in this trace spans
three orders of magnitude between a quiet night and a storm; one gamma cannot
express that, and once the volatility chain can, the coefficients stop
pretending to.

**Stages 5 and 6 buy robustness, not sharpness.** The bank is the stage-4
block and a Matern-1/2 GP, two different descriptions of the bridge rather
than one description at two settings. Behind a clamped switch that is worth
0.045 nats, and the point forecast gets worse, from 34.6 to 50.7 mm/s^2.
Broken out by two-hour window the trade is visible. Over the easier half of
the windows the mixture is worse than stage 4 by 0.16 nats on average; over
the harder half it is better by 0.24. The tail says the same thing more
sharply: stage 4's worst one percent of samples average -18.5 in log
density, the mixture's -3.8 and the Bayesian mixture's -4.8. What the second
branch removes is the disasters, which on a bridge is the part that
matters.

Giving the switch its own transition adds another 0.091 nats and brings the
point forecast back to 37.4 mm/s^2. Both stages reuse the two blocks
unchanged, which is the composability claim stated as a measurement.

An oracle that is handed the simulator's true time-varying coefficients scores
RMSE 29.5 mm/s² on the same window, so from stage 4 on the point forecast is
within a few percent of what the coefficients allow, and the remaining gains
are all in the density.

## The Matern-1/2 GP branch

`matern/` holds a Matern-1/2 Gaussian process written as a state-space model
and scored on its own. It is also the second expert of stages 5 and 6, so
this run is the reference the mixture result is read against.

The kernel and the recursion are the same object. `k(tau) = sigma^2
exp(-|tau|/ell)` is the covariance of an Ornstein-Uhlenbeck process, and
sampling that process at a fixed step gives exactly

```
f_t = a f_{t-1} + w_t,   a = exp(-dt/ell),   Var(w_t) = sigma^2 (1 - a^2)
```

so inference is the Kalman filter and it is exact. No inducing points, no
Cholesky of an n-by-n matrix, no O(n^3). `MaternHalf` in `blocks.py` is
sixty lines; a simulation check in the test notes confirms the induced
autocovariance matches `exp(-tau/ell)` to three decimals at several lags. On
a factor graph it is one more chain, drawn like every other chain here, and
the only thing that distinguishes it from the coefficient chain is that `a`
is below one, so it forgets rather than wanders.

**On its own it is a poor model, as expected.** RMSE 88 mm/s^2, near the
standard deviation of the signal, because a first-order smooth process
cannot predict a deck with two resonances. Its NLPD of -2.752 still beats
the first three rungs, and that is worth understanding: all of that comes
from tracking the amplitude, none of it from predicting the value. It is
almost a pure volatility model.

**As a branch it earns its place, and the pair beats both members.** The
switch gives it 28% of the responsibility on average under stage 5's clamped
weights and 15% once stage 6 learns the transition, in bursts rather than
evenly, and the two stages that contain it score -3.665 and -3.756 against
-3.620 for the stage-4 block alone and -2.752 for the GP alone. That is the
thing a mixture is supposed to do and it does not always happen. The reason
it happens here is that the two branches fail in different places: the
autoregression fails on impulses and on the first samples of a new regime,
which is exactly where a smooth process with a short correlation time is
still making a sane prediction.

**Which is not the same as it being a good model.** The GP branch is three
times worse than its neighbour in RMSE and its predictive standard deviation
is calibrated to that, so carrying it costs about 0.16 nats in the calm
windows and buys about 0.24 in the hard ones. The bulk cost is why stage 5's
RMSE is worse than stage 4's. If the scored window had held
no freeze and no crack, the second branch would have been a loss.

**A third instance of the same failure.** The first version tied the GP's
marginal variance to the noise its own volatility chain was estimating. That
makes the pair unidentified: the residual then scales with the prior, the
chain sees `E[lambda eps] = 1` at every scale, and it random-walks to its
clip while the branch quietly reports a predictive standard deviation of ten
thousand. The mixture's *density* was unharmed, because the switch had
already stopped selecting it, but the reported predictive variance was
nonsense and the 90% coverage came out at 1.000. Running the GP in
normalised units fixes it: the marginal variance is fixed at one, the
predictive variance in those units is a known constant, and dividing the
squared prediction error by that constant leaves a quantity whose
expectation is the scale and nothing else. Which is, exactly, what the
Transformer's instance normalisation does by hand.

**The tuned signal-to-noise ratio runs to the boundary, and that is
informative.** The standalone grid goes to 256 and the optimum still sits at
the top of it, with the gains flattening. Large SNR means vanishing
observation noise, `f_t` collapses onto `y_t`, and the GP degenerates into a
deterministic AR(1) with coefficient `exp(-1/ell)`. Given the freedom, the
Matern-1/2 branch chooses to be an autoregression of order one. It is in
that family, after all.

Inside the mixture the grid is capped at 64 for that reason. The burn-in
score between 4 and 4096 varies by 0.015 nats, which is noise, and the
argmax at the top of an open grid is the degenerate limit rather than an
estimate. The length scale and the SNR are tuned inside the mixture rather
than inherited from the standalone run, because the best solo GP and the
best complementary GP are different objects: alone it has to explain the
whole signal, in the bank it only has to explain what the autoregression
misses.

## The Transformer baseline

A causal Transformer with a Gaussian head, in `transformer/`. 43k parameters,
96 samples of context, trained on day one with early stopping on blocked
validation splits, then scored by the same `metrics` on the same window. Two
variants: frozen after day one, and warm-started and retrained at the end of
every day, which is what an operator would actually do.

It lands between the ladder's third and fourth rungs. Three things are worth
saying about that, in descending order of how much they matter.

**Instance normalisation is doing the volatility modelling.** Each context is
centred and scaled by its own mean and standard deviation before it reaches
the network. Without it nothing trains, because the excitation spans three
orders of magnitude and no fixed set of weights covers that. This is the RevIN
trick from the recent forecasting literature, and it is a volatility model
wired in ahead of the network precisely because the network cannot learn one.
Stage 4 does the same job by inference, and reports how sure it is. Half of
what the Transformer gains over stage 3 it gains from a preprocessing step
that the factor graph makes explicit and estimates.

**The point forecast is the worst of the set.** RMSE 57 mm/s^2 frozen, 50
retrained, against 30 for a plain AR(4). The simulator's bridge genuinely is
an AR(4) with drifting coefficients, so the AR models are exactly specified
and the network has to infer that structure from 96 noisy samples at a time.
This is not a statement about Transformers. It is a statement about what
knowing the model class is worth, and on this problem it is worth a factor of
two in RMSE and a great deal of compute.

**Retraining helps, and does not close the gap.** Nightly refits buy 0.25
nats and fix the calibration, which goes from 0.865 to 0.909 against a
nominal 0.90, the closest of any model here. But the gap to the Bayesian mixture is
0.37 nats on day two, 0.44 on day three, 0.46 on day four and 0.37 on day
five, and it widens rather than closing as the week goes on. Day four is
when the deck freezes. A model fitted once a night cannot track something
that changes between two fits, however well it fits what it has seen.

One honest caveat on the budget. This is a small network trained for thirty
epochs on fourteen thousand windows, on two CPU cores. A larger model, a
longer schedule, or more training days would narrow the density gap. What it
would not change is the RMSE ordering or the day-four behaviour, which are
about structure and adaptation rather than capacity.

## Four places where the obvious thing does not work

These are in the code comments too, but they are the parts worth knowing.

**The volatility update needs a curvature floor.** The Newton step for the
exponential link uses the observed curvature `kappa^2 * lambda * eps / 2`. A
single squared residual is a chi-square with one degree of freedom, so it lands
near zero about a quarter of the time, the curvature vanishes with it, and the
step is unbounded upward. On a quiet night that happens constantly and z runs
away. Flooring the curvature at its expected value `kappa^2 / 2` bounds every
step by `1/kappa` in both directions. At the fixed point `E[lambda eps] = 1`
and the floor is inactive, so nothing is distorted where it matters.

**The drift precision is not identified by the VMP fixed point.** The
variational update for the precision of a Gaussian transition uses the expected
squared increment. Under the filtered posterior that expectation equals `tr(Q)`
in expectation whatever Q is, so every alpha is a fixed point. Under a
fixed-lag smoothed posterior the degeneracy goes away in principle -- it is the
Shumway-Stoffer EM step -- but on this signal the map has a derivative around
0.998 and the estimate never leaves its prior. Both were implemented and
neither moves. What does identify alpha is the marginal likelihood, through the
innovation variance, so the alpha node takes a stochastic gradient step on the
free energy instead:

```
d(-log N(e; 0, S)) / d log q  =  -rho (e^2 / S - 1) / 2,    rho = q x'x / S
```

`rho` is the share of the predictive variance the drift injection is
responsible for, which scales the step automatically: the model only revises
its drift rate when the drift is doing work. Larger innovations than predicted
allow more drift, smaller ones less. Starting the estimate anywhere between
1e5 and 1e7 lands in the same place. A Robbins-Monro decay on the step size
makes it settle on one value, which is the static alpha the factor graph draws;
holding the step size constant instead turns it into a tracking estimate, which
is a crude volatility model in disguise and belongs at stage 4.

**Do not normalise that gradient.** Dividing by its running magnitude, the way
RMSProp and friends do, turns the step into something close to its sign. `e^2/S`
is a chi-square with one degree of freedom whose median is 0.45, not 1, so a
sign-like step drifts steadily downward on pure noise. The raw gradient has the
right mean and does not.

**A starved branch comes back as a catch-all.** In the mixtures, a branch that
loses early stops being updated, its covariance inflates unchecked, and it
returns as a diffuse component that wins on outliers and then holds the switch.
Every branch learns with at least `resp_floor` weight, which is what
interacting-multiple-model filters do for the same reason. The predictive
weights stay exact.

**With two branches that floor stops being a safety net.** With a bank of
near-copies any small floor did the job. With two branches that disagree, the
floor is the setting that matters most, so it is tuned on the burn-in like
anything else, and the burn-in picks 0.3. The same floor is applied wherever
a responsibility is carried forward rather than used as a density, which in
stage 6 means the switch's belief. Without that, the learned transition
saturates near 0.99 persistence, the predictive weight on the branch that
turns out to be right drops to a percent, and a single sample costs five
nats. Measured: at a floor of 0.02 stage 6 scores -3.586 against stage 5's
-3.640, so the Bayesian switch *loses* to the clamped one; at the tuned
floor it is -3.756 against -3.665. The burn-in score moves the same way,
3.589 at a floor of 0.02 to 3.667 at 0.2, so the protocol picks a large
floor without being told to and without seeing the scored window.

## What each stage assumes

| stage | coefficients | observation precision | switch |
|---|---|---|---|
| 1 | static, normal-gamma, exact | static, same normal-gamma | -- |
| 2 | random walk, drift clamped | static Gamma, VMP | -- |
| 3 | random walk, drift inferred | static Gamma, VMP | -- |
| 4 | random walk, drift inferred | `exp(kappa z_t + omega)`, z a random walk | -- |
| 5 | branch 1 drifts, branch 2 has none | one chain per branch | clamped uniform weights |
| 6 | branch 1 drifts, branch 2 has none | one chain per branch | Markov, Dirichlet rows |

In stages 5 and 6 the second branch is the Matern-1/2 GP, which has no
coefficients at all: it carries a scalar state `f_t` with `a = exp(-1/ell)`
and a clamped process noise.

Stage 1 is exact; the rest are variational. Stages 1 to 3 report a Student-t
predictive because a Gamma sits over the precision; stages 4 to 6 report a
Gaussian, or a mixture of them, because a log-normal does.
