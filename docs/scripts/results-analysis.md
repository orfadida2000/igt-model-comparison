# Final result-analysis script

Module:

```text
scripts.results_analysis
```

This is the command-line entry point for the final result-analysis pipeline implemented by the reusable `igt.analysis` subpackage. For the analysis methodology, generated tables and figures, and inferential procedures, see [Final result analysis](../workflow/analysis.md) and [Statistical inference](../statistical-inference.md).

## Usage

```bash
uv run python -m scripts.results_analysis \
  --fit-results path/to/model_fits_corrected_617_subjects_<timestamp>.csv \
  --comparison-results path/to/model_comparison_corrected_617_subjects_<timestamp>.csv \
  --summary-results path/to/model_summary_corrected_617_subjects_<timestamp>.csv
```

The three result-table arguments are required and must belong to the same completed result set.

## Required arguments

| Argument | Purpose |
|---|---|
| `--fit-results` | Complete participant-by-model fit-results CSV. |
| `--comparison-results` | Complete model-comparison CSV corresponding to the fit-results file. |
| `--summary-results` | Model-summary CSV corresponding to the same result set. |

## Analysis and output options

| Argument | Purpose | Default |
|---|---|---:|
| `--output-dir` | Root directory for timestamped analysis output. | `assets/results/analysis` |
| `--logging-dir` | Root directory for analysis log files. | `assets/logs/analysis` |
| `--figure-formats` | One or more Matplotlib output formats. | `png` |
| `--figure-dpi` | Figure resolution in dots per inch. | `300` |
| `--histogram-bins` | Positive integer or NumPy histogram strategy such as `auto`. | `auto` |
| `--confidence-level` | Confidence level used for bootstrap and exact binomial intervals. | `0.95` |
| `--bootstrap-resamples` | Number of BCa bootstrap resamples for criterion-difference inference. | `10000` |
| `--log-level` | Root logger level; a negative value disables logging. | project default |

The project currently uses a fixed shared seed, so the bootstrap seed is taken from project configuration rather than exposed as a CLI argument.

## Outputs

A standard run creates a timestamped directory containing:

- seven derived CSV tables;
- 18 figures with the default two-model PNG configuration;
- `analysis_report.txt`;
- logging artifacts when logging is enabled.

See [Final result analysis](../workflow/analysis.md) for the full artifact list and [Result files](../results.md) for output organization.

## Related documentation

- [Final result analysis](../workflow/analysis.md)
- [Statistical inference](../statistical-inference.md)
- [Final results](../final-results.md)
- [Analysis API](../api/analysis.md)
- [Entry points and scripts API](../api/entry-points.md)
