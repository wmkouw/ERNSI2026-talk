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
  mixture.py    the stage-4 block treated as a node, plus what the two
                mixture stages share
  runner.py     one entry point so every stage is fitted the same way
  run_all.py    fit all six and write results/summary.csv
  ar/ tvar/ htvar/ hgf/ moe/ bmoe/
                one folder per stage: model.py has the model, run.py fits it
```

Run everything with

```
cd ERNSI2026-talk
python -m models.run_all
```

or one stage with `python -m models.hgf.run`. Then `python results_notebook.py`
redraws the slide figures into `../figures/`, or `marimo edit results_notebook.py`
to work on them interactively.

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
| 5 | mixture | 30.1 | -3.874 | 0.939 |
| 6 | Bayesian mixture | 31.3 | -3.982 | 0.939 |

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

**Stages 5 and 6.** A bank of three drift rates behind one selector adds 0.25
nats with the point forecast back at the stage-1 level, and giving the switch
its own transition adds 0.11 more. Both reuse the stage-4 block unchanged.

An oracle that is handed the simulator's true time-varying coefficients scores
RMSE 29.5 mm/s² on the same window, so from stage 4 on the point forecast is
within a few percent of what the coefficients allow, and the remaining gains
are all in the density.

## Three places where the obvious thing does not work

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

## What each stage assumes

| stage | coefficients | observation precision | switch |
|---|---|---|---|
| 1 | static, normal-gamma, exact | static, same normal-gamma | -- |
| 2 | random walk, drift clamped | static Gamma, VMP | -- |
| 3 | random walk, drift inferred | static Gamma, VMP | -- |
| 4 | random walk, drift inferred | `exp(kappa z_t + omega)`, z a random walk | -- |
| 5 | three clamped drift rates | one chain per branch | clamped uniform weights |
| 6 | three clamped drift rates | one chain per branch | Markov, Dirichlet rows |

Stage 1 is exact; the rest are variational. Stages 1 to 3 report a Student-t
predictive because a Gamma sits over the precision; stages 4 to 6 report a
Gaussian, or a mixture of them, because a log-normal does.
