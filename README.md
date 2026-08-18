# IGT Model Comparison

Computational comparison of **Q-learning** and **PVL-Delta** on the Iowa Gambling Task (IGT), using the bundled Steingroever healthy-participant dataset.

The project fits both models separately to each participant, compares their fit with likelihood-based information criteria, corrects rare PVL-Delta optimization failures using Q-learning-equivalent warm starts, evaluates Q-learning inverse-temperature sensitivity, and produces descriptive and inferential result analyses.

## Research question

The project asks whether healthy participants' IGT choices are adequately explained by learning from objective monetary outcomes, or whether a model that additionally represents subjective outcome sensitivity and loss aversion provides a better account of behavior.

The two fitted models are:

- **Q-learning** — a two-parameter delta-rule model with learning rate and softmax inverse temperature.
- **PVL-Delta** — a four-parameter Prospect Valence Learning model with learning rate, outcome sensitivity, loss aversion, and response consistency.

## Repository layout

```text
igt-model-comparison/
├── assets/
│   ├── data/                     # Original RData and processed long-format CSV
│   ├── presentations/            # Project presentation material
│   └── results/                  # Fitting, correction, sensitivity, and analysis outputs
├── docs/                         # MkDocs documentation source
├── igt/                          # Main Python package
│   ├── analysis/                 # Result validation, tables, plots, statistics, and report generation
│   ├── cli_parsing/              # Shared declarative argparse/type-filter infrastructure
│   ├── constants/                # Stable defaults, paths, schema names, and model constants
│   ├── execution/                # Model fitting, multiprocessing, and fitting pipeline
│   ├── models/                   # Computational model implementations
│   ├── notify/                   # Optional FormSubmit notifications
│   └── utils/                    # Generic I/O and tabular helpers
├── scripts/
│   ├── correct_pvl_delta_fits.py
│   ├── q_inverse_temperature_sensitivity.py
│   └── results_analysis.py
├── mkdocs.yml
├── pyproject.toml
└── uv.lock
```

## Requirements

- Python 3.12 or newer
- `uv` for environment and dependency management

The repository currently pins its local interpreter through `.python-version` and declares all runtime/documentation dependencies in `pyproject.toml` and `uv.lock`.

## Setup

From the repository root:

```bash
uv sync
```

The default dataset path is:

```text
assets/data/IGTdata.rdata
```

A processed long-format copy is also included at:

```text
assets/data/processed/igt_long.csv
```

## Primary fitting workflow

Run the complete Q-learning/PVL-Delta fitting and comparison pipeline with:

```bash
uv run python -m igt.main
```

Important defaults include:

- all 617 available subjects;
- Q-learning maximum inverse temperature of 100;
- up to 10 Q-learning grid-local-minimum starts;
- 32 scrambled Sobol starts for PVL-Delta;
- L-BFGS-B with up to 1,000 iterations per local optimization;
- deterministic seed 42 for the PVL-Delta Sobol initialization;
- serial execution by default.

Useful options include:

```text
--rdata-path
--max-iterations
--q-starts
--pvl-starts
--q-max-inverse-temperature
--workers
--subjects
--output-dir
--logging-dir
--log-level
--no-progress
--analyze
```

For example:

```bash
uv run python -m igt.main \
  --workers 4 \
  --q-starts 10 \
  --pvl-starts 32 \
  --q-max-inverse-temperature 100 \
  --max-iterations 1000
```

Each run receives its own timestamped output directory under `assets/results/igt_model_comparison/` unless another output directory is supplied.

## PVL-Delta correction workflow

Because Q-learning is representable inside PVL-Delta for the relevant parameter mapping, a fitted PVL-Delta NLL that is meaningfully worse than the corresponding Q-learning NLL is treated as evidence that the local optimization did not locate a sufficiently good PVL-Delta solution.

The correction script selects those subjects, maps each fitted Q-learning solution to a Q-equivalent PVL-Delta warm start, refits PVL-Delta with that additional start, replaces only the targeted PVL-Delta rows, regenerates comparison/summary tables, and writes an audit report.

```bash
uv run python -m scripts.correct_pvl_delta_fits \
  path/to/model_fits_617_subjects_<timestamp>.csv
```

The default per-trial NLL comparison tolerance is `1e-8`, so a subject with `N` trials uses an absolute tolerance of `N × 1e-8`.

## Q-learning inverse-temperature sensitivity

The sensitivity script selects converged Q-learning fits whose fitted inverse temperature reaches the configured selection threshold and reruns those participants under the predefined sensitivity settings.

```bash
uv run python -m scripts.q_inverse_temperature_sensitivity \
  path/to/model_fits_617_subjects_<timestamp>.csv
```

This workflow is intended to test whether conclusions depend on the Q-learning inverse-temperature ceiling rather than to replace the primary fitting pipeline.

## Final result analysis

Run the analysis script on a mutually consistent fit/comparison/summary result triplet:

```bash
uv run python -m scripts.results_analysis \
  --fit-results path/to/model_fits_corrected_617_subjects_<timestamp>.csv \
  --comparison-results path/to/model_comparison_corrected_617_subjects_<timestamp>.csv \
  --summary-results path/to/model_summary_corrected_617_subjects_<timestamp>.csv
```

By default, the standalone analysis creates a timestamped directory under:

```text
assets/results/analysis/
```

It performs:

- cross-table schema and integrity validation;
- paired subject-level Q-learning vs PVL-Delta comparisons;
- AIC/BIC win summaries overall and by source study;
- parameter and boundary diagnostics;
- uniform-choice fit diagnostics;
- BCa bootstrap confidence intervals for mean and median AIC/BIC differences;
- two-sided Wilcoxon signed-rank tests for paired AIC/BIC differences;
- exact binomial confidence intervals and two-sided exact binomial tests for PVL-Delta win rates;
- standard figures and a plain-text analysis report.

The statistical defaults are a 95% confidence level, 10,000 BCa bootstrap resamples, and seed 42.

## Final included results

The repository includes the final corrected 617-subject result set and a corresponding analysis run. In that analysis:

- both models converged for all 617 subjects;
- PVL-Delta won for 510 subjects by AIC (82.66%);
- PVL-Delta won for 427 subjects by BIC (69.21%);
- mean and median Q-minus-PVL AIC/BIC differences were positive, with 95% BCa confidence intervals entirely above zero;
- the paired Wilcoxon and exact binomial analyses both indicated strong population-level preference for PVL-Delta.

See [Final results](docs/final-results.md) for the complete interpretation and the exact included output paths.

## Documentation

The MkDocs site is the canonical project documentation. It covers model equations, initialization, fitting, correction, sensitivity analysis, statistical inference, outputs, project architecture, and the Python API.

Serve it locally:

```bash
uv run mkdocs serve
```

Build it strictly:

```bash
uv run mkdocs build --strict
```

Start with [`docs/index.md`](docs/index.md) or the [Getting started](docs/getting-started.md) guide.

## Reproducibility notes

- PVL-Delta Sobol initialization uses the fixed seed configured in `igt.constants.config`.
- The analysis bootstrap uses seed 42 by default.
- Result-producing workflows use timestamped directories so reruns do not overwrite earlier artifacts.
- Boundary estimates are reported as diagnostics; they are not automatically treated as failed fits.
- Population-level inferential summaries treat subjects as independent observational units. Source-study heterogeneity is summarized separately rather than modeled hierarchically.

## License

See [`LICENSE`](LICENSE).
