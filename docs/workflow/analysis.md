# Final result analysis

## Separation of concerns

`igt.analysis` is reusable package code. The command-line workflow is the sibling module:

```text
scripts/results_analysis.py
```

Run it from the repository root as:

```bash
uv run python -m scripts.results_analysis \
  --fit-results path/to/model_fits_corrected.csv \
  --comparison-results path/to/model_comparison_corrected.csv \
  --summary-results path/to/model_summary_corrected.csv
```

For the complete command-line interface, see [Final result-analysis script](../scripts/results-analysis.md).

## Input validation

Before deriving any result, [`validate_result_tables`][igt.analysis.validation.validate_result_tables] checks:

- required columns and duplicate names;
- numeric, integer, Boolean, and string normalization;
- supported model identities;
- exactly one row per subject/model combination;
- source-study consistency between model rows;
- preservation of fit metrics in the comparison table;
- recomputation consistency of the summary table.

The three input CSVs therefore must come from the same completed result set.

## Analysis configuration

Default values are:

| Setting | Default |
|---|---:|
| figure formats | `png` |
| figure DPI | 300 |
| histogram bins | `auto` |
| confidence level | 0.95 |
| bootstrap resamples | 10,000 |
| bootstrap seed | 42 |
| numeric validation tolerance | `1e-10` |
| parameter-boundary tolerance | `1e-8` |

The CLI exposes figure format/DPI, histogram bins, confidence level, and bootstrap resample settings. With the current fixed-seed project configuration, the bootstrap seed is taken from the shared fixed seed rather than exposed as a CLI option.

## Derived tables

The analysis writes seven CSV tables:

```text
subject_level_model_comparison.csv
study_model_preference.csv
boundary_summary.csv
parameter_summary.csv
model_win_summary.csv
criterion_difference_inference.csv
model_win_inference.csv
```

### Subject-level comparison

The paired table places both models on one row per eligible subject and defines signed criterion differences as:

\[
\Delta AIC = AIC_Q - AIC_{PVL}
\]

\[
\Delta BIC = BIC_Q - BIC_{PVL}
\]

Positive values favor PVL-Delta.

### Study preference

`study_model_preference.csv` summarizes AIC/BIC win counts, win rates, and signed criterion-difference summaries separately by `source_study`.

### Boundary and parameter summaries

Boundary tables quantify lower/upper/any-bound solutions. Parameter summaries report count, mean, standard deviation, quartiles, extrema, configured bounds, and boundary frequencies.

## Figures

With the default two models and PNG output, the standard run produces 18 figures:

- signed AIC and BIC difference distributions;
- Q-vs-PVL NLL, AIC, and BIC scatterplots;
- model-win counts;
- criterion-difference confidence intervals;
- PVL-Delta win-rate confidence intervals;
- source-study preference rates;
- boundary-fit rates;
- uniform-choice improvement distribution and paired scatter;
- one fitted-parameter distribution for each of the six model parameters.

## Text report

`analysis_report.txt` records validation success, aggregate fit statistics, model-win rates, inferential results, source-study coverage, boundary diagnostics, parameter summaries, and generated artifact paths.
