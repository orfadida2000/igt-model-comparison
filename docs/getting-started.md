# Getting started

## Environment

The project requires Python 3.12+ and is managed with `uv`.

From the repository root:

```bash
uv sync
```

The default dataset is already included at:

```text
assets/data/IGTdata.rdata
```

## Run the primary model comparison

```bash
uv run python -m igt.main
```

This fits Q-learning and PVL-Delta to all available participants using the configured defaults and writes a timestamped result directory under `assets/results/igt_model_comparison/`.

A typical explicit run is:

```bash
uv run python -m igt.main \
  --workers 4 \
  --q-starts 10 \
  --pvl-starts 32 \
  --q-max-inverse-temperature 100 \
  --max-iterations 1000
```

Use `--analyze` to run the standard result analysis immediately on the newly generated uncorrected result triplet.

!!! note
    The final repository analysis is based on the **corrected** PVL-Delta result triplet, so the standalone analysis script is the appropriate final-analysis entry point after the correction workflow.

## Correct PVL-Delta optimization failures

Given the initial fit-results CSV:

```bash
uv run python -m scripts.correct_pvl_delta_fits \
  assets/results/igt_model_comparison/<run>/model_fits_617_subjects_<timestamp>.csv
```

The correction workflow writes a new timestamped directory containing corrected fit, comparison, and summary CSVs plus a correction audit report. See the [correction script reference](scripts/correct-pvl-delta-fits.md) for its command-line arguments.

## Analyze the corrected result triplet

```bash
uv run python -m scripts.results_analysis \
  --fit-results assets/results/corrected_pvl_delta_fits/<run>/model_fits_corrected_617_subjects_<timestamp>.csv \
  --comparison-results assets/results/corrected_pvl_delta_fits/<run>/model_comparison_corrected_617_subjects_<timestamp>.csv \
  --summary-results assets/results/corrected_pvl_delta_fits/<run>/model_summary_corrected_617_subjects_<timestamp>.csv
```

The analysis command creates a new timestamped directory under `assets/results/analysis/` by default. See the [result-analysis script reference](scripts/results-analysis.md) for its command-line arguments.

## Build the documentation

```bash
uv run mkdocs build --strict
```

For local browsing with live reload:

```bash
uv run mkdocs serve
```
