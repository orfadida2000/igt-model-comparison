# Final included results

The repository includes the original 617-subject model-fitting run, a targeted
correction of selected PVL-Delta fits, and the downstream analysis based on the
corrected result set.

## Original model-fitting run

```text
assets/results/igt_model_comparison/2026-08-03_03-28-05/
```

This directory contains the original per-subject Q-learning and PVL-Delta fits
for the full dataset of 617 participants.

The original fitting results were used as the starting point for the final
analysis. Diagnostic checks identified eight PVL-Delta fits for which the fitted
negative log-likelihood was worse than the corresponding nested Q-learning
solution. Those fits were subsequently refitted by the correction workflow
described below.

The original results are retained in the repository so that the initial fitting
output, the identified correction targets, and the final corrected results remain
traceable.

## Corrected PVL-Delta result set

```text
assets/results/corrected_pvl_delta_fits/2026-08-12_07-49-35/
```

The corresponding correction report documents eight targeted PVL-Delta fits:

- 5 participants with 100 trials;
- 3 participants with 150 trials.

Each targeted participant was refitted using the normal PVL-Delta starting
points together with a starting point corresponding to its fitted Q-learning
solution.

All eight corrected fits obtained a meaningfully lower NLL than their original
PVL-Delta fit, none remained worse than the corresponding Q-learning fit, and
all non-targeted rows passed the integrity checks as unchanged.

The corrected result set, rather than the uncorrected original PVL-Delta fits,
is used for the final model-comparison analysis.

## Included analysis run

```text
assets/results/analysis/2026-08-16_06-11-54/
```

Both models converged for all 617 participants.

| Criterion | Q-learning wins | PVL-Delta wins | PVL-Delta win rate |
|---|---:|---:|---:|
| AIC | 107 | 510 | 82.66% |
| BIC | 190 | 427 | 69.21% |

There were no exact AIC or BIC ties.

## Criterion-difference estimates

Signed differences are Q-learning minus PVL-Delta, so positive values favor
PVL-Delta.

| Criterion | Mean difference | 95% BCa CI | Median difference | 95% BCa CI |
|---|---:|---|---:|---|
| AIC | 27.2510 | [24.7403, 30.4383] | 16.2498 | [13.5006, 18.8821] |
| BIC | 21.9144 | [19.4027, 25.0939] | 11.0395 | [8.2902, 13.5572] |

Both mean and median confidence intervals lie entirely above zero.

## Paired and win-rate tests

| Criterion | Wilcoxon statistic | Wilcoxon two-sided p | Exact-binomial two-sided p |
|---|---:|---:|---:|
| AIC | 9,504 | 1.26 × 10⁻⁸³ | 7.86 × 10⁻⁶⁴ |
| BIC | 30,440 | 1.38 × 10⁻⁴⁸ | 6.45 × 10⁻²² |

PVL-Delta non-tied win-rate confidence intervals were:

- AIC: 82.66%, 95% exact CI [79.44%, 85.56%];
- BIC: 69.21%, 95% exact CI [65.40%, 72.83%].

## Q-learning inverse-temperature sensitivity and final bound

The final Q-learning inverse-temperature bound was chosen through a targeted sensitivity analysis rather than treated as a psychological constant. The initial fitting configuration used an upper bound of 20. Converged Q-learning fits with `inverse_temperature >= 19.5` were selected, yielding 154 participants, and that cohort was refit under wider ceilings of 50 and 100.

The wider-bound fits showed that 20 was restrictive for a substantial subset of participants, so the final primary analysis used `beta_max = 100`. In the final 617-participant result set, 149 Q-learning fits (24.15%) reached the upper inverse-temperature bound of 100. This is retained as a boundary diagnostic rather than automatically classified as fit failure.

The numerical size of inverse temperature depends on the scale of the learned values. The value 100 should therefore be interpreted as the final optimization/search ceiling used by this project, not as a universal psychological maximum.

The current sensitivity script remains available as a reusable diagnostic, but its present predefined rerun conditions are not intended to recreate the historical 20 -> 50 -> 100 sweep verbatim. See [Q-learning inverse-temperature sensitivity](workflow/sensitivity.md).

## Boundary diagnostics

Boundary solutions remain common and are treated as diagnostics rather than
automatic fit failures:

- Q-learning: 422/617 fits (68.40%) had at least one parameter at a bound;
- PVL-Delta: 393/617 fits (63.70%) had at least one parameter at a bound.

The generated `parameter_summary.csv` and parameter-distribution figures provide
the parameter-specific breakdown.

## Interpretation

The final corrected results consistently favor PVL-Delta at the
participant-population level, with stronger preference under AIC and a still
substantial preference under BIC. BIC is more conservative because it penalizes
PVL-Delta's additional parameters more strongly.

The source-study summaries show that preference strength varies across studies.
The inferential analysis nevertheless treats participants, not studies, as
independent units and does not fit a hierarchical study model.

## Final written report

The completed academic report corresponding to these results is stored under:

```text
report/report.pdf
```

The full LaTeX source is kept in the same `report/` directory. The report uses the corrected result set and final analysis summarized on this page, while moving detailed fitting settings, sensitivity checks, optimization-correction details, and secondary diagnostics into appendices.

- [Final report documentation](report.md)
- [Open the final PDF on GitHub](https://github.com/orfadida2000/igt-model-comparison/blob/main/report/report.pdf)
- [Browse the report source on GitHub](https://github.com/orfadida2000/igt-model-comparison/tree/main/report)
