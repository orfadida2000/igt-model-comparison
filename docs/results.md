# Result files and schemas

## Primary fit table

`model_fits_*.csv` contains one row per participant/model fit. Core columns include:

```text
model
n_trials
subject_id
source_study
negative_log_likelihood
log_likelihood
aic
bic
uniform_choice_nll
nll_improvement_over_uniform
uniform_choice_fit
n_parameters_at_lower_bound
n_parameters_at_upper_bound
n_parameters_at_any_bound
converged
optimizer_message
n_function_evaluations
n_iterations
n_starts
```

Model parameter columns are added to the same row according to the fitted model.

## Comparison table

`model_comparison_*.csv` preserves the fit-result rows and adds:

```text
comparison_eligible
delta_aic
delta_bic
best_aic
best_bic
```

`delta_aic` and `delta_bic` are criterion differences from the within-subject minimum, so zero indicates a best-fitting row for that criterion.

## Summary table

`model_summary_*.csv` contains one row per model with:

- fit count;
- convergence count/rate;
- number of eligible comparisons;
- mean NLL/AIC/BIC among eligible comparisons;
- AIC and BIC win counts.

## Corrected results

The correction workflow writes the same three table schemas with `corrected` in the filenames and a separate text audit report.

## Analysis outputs

The standalone analysis creates:

```text
<timestamp>/
├── analysis_report.txt
├── figures/
└── tables/
```

See [Final result analysis](workflow/analysis.md) for the seven derived tables and 18 default figures.
