# `igt.analysis`

`igt.analysis` is the result-analysis subpackage of the main `igt` package. It is
not a standalone package. It consumes the final fit, comparison, and summary
CSVs produced by the project and uses shared definitions and utilities from the
rest of `igt`.

The analysis has three layers:

1. input validation and cross-table consistency checks;
2. descriptive model-comparison and fitting diagnostics;
3. population-level statistical inference for the paired AIC/BIC comparisons.

## Command-line use

Run from the project root:

```powershell
uv run python -m igt.analysis `
    --fits path\to\model_fits_corrected.csv `
    --comparison path\to\model_comparison_corrected.csv `
    --summary path\to\model_summary_corrected.csv `
    --output-directory path\to\analysis
```

The statistical defaults are a 95% confidence level, 10,000 BCa bootstrap
resamples, and bootstrap seed 42. They can be overridden with:

```powershell
--confidence-level 0.95 `
--bootstrap-resamples 10000 `
--bootstrap-seed 42
```

## Python API

```python
from pathlib import Path

from igt.analysis import generate_results_analysis

outputs = generate_results_analysis(
    Path("model_fits_corrected.csv"),
    Path("model_comparison_corrected.csv"),
    Path("model_summary_corrected.csv"),
    Path("analysis"),
)
```

## Generated tables

- `subject_level_model_comparison.csv`
- `study_model_preference.csv`
- `boundary_summary.csv`
- `parameter_summary.csv`
- `model_win_summary.csv`
- `criterion_difference_inference.csv`
- `model_win_inference.csv`

`criterion_difference_inference.csv` analyzes the paired subject-level
criterion differences

```text
AIC(Q-learning) - AIC(PVL-Delta)
BIC(Q-learning) - BIC(PVL-Delta)
```

Positive values favor PVL-Delta. For each criterion the table reports the mean
and median difference, BCa bootstrap confidence intervals, and a two-sided
Wilcoxon signed-rank test against zero.

`model_win_inference.csv` reports PVL-Delta and Q-learning wins, exact ties, the
PVL-Delta win rate among non-tied subjects, an exact binomial confidence
interval, and a two-sided exact binomial test against a 50% win probability.

## Generated figures

The standard figure set contains:

- signed AIC and BIC difference distributions;
- Q-learning versus PVL-Delta NLL, AIC, and BIC scatterplots;
- AIC and BIC model-win counts;
- mean and median AIC/BIC differences with bootstrap confidence intervals;
- PVL-Delta win rates with exact binomial confidence intervals;
- PVL-Delta preference rates by source study;
- fit-level parameter-boundary rates;
- improvement over uniform-choice NLL;
- one distribution for every configured model parameter.

## Statistical interpretation

The inferential analyses treat subjects as independent observational units.
The source-study summaries remain descriptive; this subpackage does not fit a
hierarchical or study-clustered inferential model.

The Wilcoxon signed-rank test uses the usual symmetry assumption for paired
differences. The exact binomial test is a complementary magnitude-free test of
whether PVL-Delta wins more often than expected under a 50/50 null among
non-tied subjects.
