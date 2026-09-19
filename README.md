# ERNSI2026-talk

Code for the talk on Bayesian system identification at the European Research
Network on System Identification 2026.

- `bridge.py` --- the road-bridge benchmark: a time-varying, volatile,
  regime-switching system. `data/sim_001.csv` is one five-day scenario at
  20 Hz, with the simulator's ground truth alongside the measurement.
- `models/` --- the six models from the keynote's factor graphs, fitted to
  that trace. See `models/README.md` for the protocol, the results and the
  three places where the textbook update does not work.
- `results/` --- one-step-ahead forecasts, metrics and settings per model,
  plus `summary.csv`.
- `results_notebook.py` --- a marimo notebook that draws the figures. Open it
  with `marimo edit results_notebook.py` to change the plotting code live, or
  run `python results_notebook.py` to regenerate the slide PDFs in
  `../figures/`.

```
python -m models.run_all      # fit all six, write results/
python results_notebook.py    # redraw ../figures/res_*.pdf
```
