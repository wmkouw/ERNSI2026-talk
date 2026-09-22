# ERNSI2026-talk

Code for the talk on Bayesian system identification at the European Research
Network on System Identification 2026.

- `bridge.py` --- the road-bridge benchmark: a time-varying, volatile,
  regime-switching system. `data/sim_001.csv` is one five-day scenario at
  20 Hz, with the simulator's ground truth alongside the measurement.
- `models/` --- the six models from the keynote's factor graphs, fitted to
  that trace, plus a Transformer baseline to compare them against. See
  `models/README.md` for the protocol, the results and the three places where
  the textbook update does not work.
- `results/` --- one-step-ahead forecasts, metrics and settings per model,
  plus `summary.csv`.
- `bridge_demo.py` --- one marimo notebook for the whole talk, in two parts.
  Part one is the animated problem statement: a week of the scenario played
  back, with a switch that reveals the hidden truth. Part two draws the slide
  figures from the stored predictions, then imports the models themselves, so
  you can point one at any stretch of the trace and see the signal against its
  forecast at full rate, with a checkbox to run all six over the same window.
  Open it with `marimo edit bridge_demo.py` to change the plotting code live,
  or run `python bridge_demo.py` to regenerate the slide PDFs in `../figures/`.

```
python -m models.run_all                        # fit all six, write results/
python -m models.transformer.run --nightly      # the baseline (needs torch)
python -m models.decision                       # the closing decision slide
python bridge_demo.py                           # redraw ../figures/res_*.pdf
```

## In the browser

`python build_site.py` builds a static copy of the notebook into `site/` that
runs entirely in the browser (marimo's WASM export). It zips `bridge.py`,
`bridge_viz.py`, `models/`, `data/` and `results/` into `public/bundle.zip`,
which the notebook's first cell fetches and unpacks when it runs there. Any
static host will do; `.github/workflows/pages.yml` publishes it to GitHub
Pages on every push to `main`. Preview locally with
`python -m http.server -d site`. Use `--mode edit` to show the code.
