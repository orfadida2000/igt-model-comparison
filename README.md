# IGT Model Comparison

Compare two computational models of decision-making on the Iowa Gambling Task
(IGT):

- Q-learning
- PVL-Delta

The project loads the Steingroever IGT dataset, fits both models per subject,
and exports fit, comparison, and summary CSV files.

## Repository structure

- `igt/` – core package for models, fitting, comparison, logging, and CLI workflow
- `assets/data/IGTdata.rdata` – input dataset
- `assets/results/` – output directory for generated CSV files
- `docs/` – MkDocs documentation source

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
uv sync
```

## Run model fitting and comparison

Default run:

```bash
uv run python -m igt.main
```

Example with custom options:

```bash
uv run python -m igt.main \
  --subjects 20 \
  --workers 4 \
  --q-starts 5 \
  --pvl-starts 32 \
  --q-max-inverse-temperature 20 \
  --max-iterations 1000
```

## CLI options

Key arguments:

- `--rdata-path` – path to the `.rdata` or `.rda` dataset file
- `--output-dir` – directory for result CSV files
- `--q-starts` – maximum number of distinct Q-learning grid-local-minimum starts
- `--pvl-starts` – positive power-of-two number of PVL-Delta Sobol starts
- `--q-max-inverse-temperature` – Q-learning inverse-temperature upper bound
- `--max-iterations` – maximum L-BFGS-B iterations per optimizer run
- `--workers` – number of worker processes; `0` selects serial execution
- `--subjects` – number of subjects to fit; `0` produces empty result CSVs
- `--no-progress` – disable the progress bar

The current main workflow uses the fixed seed configured in
`igt.constants.config`. The parser also supports a user-provided or unfixed
seed when `USE_FIXED_SEED` is disabled.

## Q-learning initialization

The default learning-rate grid contains 31 quadratically spaced points over
`[0, 1]`, concentrating resolution near zero. The inverse-temperature grid is
linear and automatically keeps approximately unit spacing. Therefore:

- maximum `20` gives 21 inverse-temperature values;
- maximum `50` gives 51 values;
- maximum `100` gives 101 values.

The implementation selects up to five distinct grid-local NLL minima and runs
one L-BFGS-B optimization from each selected point.

## Outputs

Running `igt.main` writes three files to the output directory:

- `model_fits_<n>_subjects_<timestamp>.csv` – per-subject, per-model fits and diagnostics
- `model_comparison_<n>_subjects_<timestamp>.csv` – comparison eligibility, AIC/BIC deltas, and winners
- `model_summary_<n>_subjects_<timestamp>.csv` – convergence and valid-comparison summary

Fit diagnostics include:

- uniform-choice NLL;
- NLL improvement over uniform choice;
- a uniform-choice-fit flag;
- counts of parameters at lower and upper bounds.

A boundary estimate is diagnostic and does not automatically invalidate a fit.
A subject enters AIC/BIC comparison only when all model fits for that subject
converged and have finite comparison metrics.

## Documentation

Serve docs locally:

```bash
uv run mkdocs serve
```

Build docs:

```bash
uv run mkdocs build
```
