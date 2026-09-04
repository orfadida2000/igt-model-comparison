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
│   ├── signed_aic_difference_distribution.png
│   ├── signed_aic_difference_distribution.pdf
│   ├── ...
│   └── parameters/
│       ├── q_learning/
│       │   ├── <parameter>_distribution.png
│       │   └── <parameter>_distribution.pdf
│       └── pvl_delta/
│           ├── <parameter>_distribution.png
│           └── <parameter>_distribution.pdf
└── tables/
    ├── subject_level_model_comparison.csv
    ├── subject_level_model_comparison.tex
    ├── study_model_preference.csv
    ├── study_model_preference.tex
    ├── ...
    ├── model_win_inference.csv
    └── model_win_inference.tex
```

The analysis produces seven logical derived tables and 18 logical figures with the default two-model configuration.

Each logical table is exported as:

- `.csv` for canonical machine-readable analysis output;
- `.tex` for direct reuse in LaTeX documents.

Each logical figure is exported by default as:

- `.png` for inspection and general-purpose use;
- `.pdf` for vector report-quality use.

Different representations of the same artifact are stored side by side in the same semantic directory rather than in separate format-specific directory trees.

See [Final result analysis](workflow/analysis.md) for the complete analysis methodology and artifact list.

## Relationship to the final written report

The analysis outputs above are research artifacts produced by Python. They are distinct from the final academic report under `report/`. The LaTeX report incorporates selected tables and figures derived from the final corrected analysis, while `analysis_report.txt` remains a machine/workflow-oriented text summary rather than the course deliverable.

The compiled course report is `report/report.pdf`. See [Final results](final-results.md) for the exact result set represented in that document.
