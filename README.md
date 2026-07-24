# IGT Model Comparison

Compare two computational models of decision-making on the Iowa Gambling Task (IGT):

- Q-learning
- PVL-Delta

The project loads the Steingroever IGT dataset, fits both models per subject, and exports fit/comparison summaries as CSV files.

## Repository structure

- `igt/` – core package (models, fitting, comparison, CLI workflow)
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
  --q-starts 4 \
  --pvl-starts 32 \
  --max-iterations 1000
```

## CLI options

Key arguments:

- `--rdata-path` path to `.rdata` / `.rda` dataset file
- `--output-dir` directory for result CSV files
- `--q-starts` number of Q-learning grid starts (minimum 1)
- `--pvl-starts` number of PVL-Delta Sobol starts (rounded to nearest power of 2)
- `--rng` non-negative RNG seed for Sobol sampling
- `--max-iterations` maximum L-BFGS-B iterations per fit
- `--workers` number of worker processes
- `--subjects` number of subjects to fit (default: all)
- `--no-progress` disable progress bar

## Outputs

Running `igt.main` writes three files to the output directory:

- `model_fits.csv` – per-subject, per-model fit results
- `model_comparison.csv` – model fit table with comparison columns
- `model_summary.csv` – aggregate comparison summary

## Documentation

Serve docs locally:

```bash
uv run mkdocs serve
```

Build docs:

```bash
uv run mkdocs build
```

