# `igt.results_analysis`

This subpackage validates the three model-result CSV files and generates
subject-level comparison tables, study summaries, parameter summaries,
diagnostic figures, and a text report.

## Placement

Place this directory at:

```text
igt/
└── results_analysis/
```

## Required shared project definitions

This version intentionally does not duplicate project-wide schema names,
model names, parameter definitions, bounds, path types, CSV helpers, or
Series-normalization helpers. It imports them from:

- `igt.constants.models`
- `igt.constants.schema`
- `igt.typing`
- `igt.io`
- `igt.tabular`
- `igt.subject_selection`

The integration changes required in those modules are documented in the
accompanying handoff message.

## Command-line use

Run from the project root:

```powershell
uv run python -m igt.results_analysis `
    --fits path\to\model_fits_corrected.csv `
    --comparison path\to\model_comparison_corrected.csv `
    --summary path\to\model_summary_corrected.csv `
    --output-directory path\to\results_analysis
```

Generate both PNG and SVG figures:

```powershell
uv run python -m igt.results_analysis `
    --fits path\to\model_fits_corrected.csv `
    --comparison path\to\model_comparison_corrected.csv `
    --summary path\to\model_summary_corrected.csv `
    --output-directory path\to\results_analysis `
    --figure-formats png svg
```

## Python API

```python
from pathlib import Path

from igt.results_analysis import generate_results_analysis

outputs = generate_results_analysis(
    Path("model_fits_corrected.csv"),
    Path("model_comparison_corrected.csv"),
    Path("model_summary_corrected.csv"),
    Path("results_analysis"),
)
```

## Generated tables

- `subject_level_model_comparison.csv`
- `study_model_preference.csv`
- `boundary_summary.csv`
- `parameter_summary.csv`
- `model_win_summary.csv`

## Generated figures

- Signed AIC and BIC difference distributions
- Q-learning versus PVL-Delta NLL, AIC, and BIC scatterplots
- AIC and BIC model-win counts
- PVL-Delta preference rates by source study
- Fit-level parameter-boundary rates
- Improvement over uniform-choice NLL
- One distribution for every configured model parameter

Positive signed differences mean PVL-Delta has the lower criterion value:

```text
criterion(Q-learning) - criterion(PVL-Delta) > 0
```
