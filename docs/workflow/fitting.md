# Fitting and comparison

## Entry point

```bash
uv run python -m igt.main
```

The primary workflow constructs one [`QLearningModel`][igt.models.q_learning.QLearningModel] and one [`PVLDeltaModel`][igt.models.pvl_delta.PVLDeltaModel], loads the IGT data, fits each model to every selected subject, and writes three result tables.

## Optimization

Every local fit uses SciPy's **L-BFGS-B** optimizer with model-specific parameter bounds. The default maximum is 1,000 optimizer iterations per start.

For each model/subject fit:

1. model-specific starting points are generated;
2. optional additional warm starts are appended when provided;
3. L-BFGS-B is run independently from each start;
4. among finite successful runs, the result with the lowest NLL is retained;
5. if every finite run is unsuccessful, the finite run with the lowest NLL is retained for diagnostics and the fit remains marked nonconverged;
6. fit statistics and diagnostics are computed.

## Serial and multiprocessing execution

`--workers` is normalized as follows:

- `0` → serial fitting in the main process;
- positive integer → that many worker processes;
- negative value → process-pool default, effectively allowing all available CPU cores.

Parallel execution uses a spawn multiprocessing context and reproduces the application's logging state inside workers. When normal root handlers exist, worker log records are forwarded through a manager-backed queue to a `QueueListener` in the parent process.

## Fit statistics

For a fitted subject/model combination, the pipeline records:

- negative log-likelihood (NLL);
- log-likelihood;
- AIC;
- BIC;
- convergence flag and optimizer message;
- function evaluations and iterations;
- number of starts;
- uniform-choice NLL;
- improvement over uniform choice;
- uniform-choice-fit flag;
- counts of parameters at lower, upper, or any configured bound;
- fitted model parameters.

For \(k\) fitted parameters, \(n\) trials, and log-likelihood \(LL\):

\[
AIC = 2k - 2LL
\]

\[
BIC = k\log(n) - 2LL
\]

Lower NLL, AIC, and BIC are better.

## Comparison eligibility

A subject is eligible for within-subject information-criterion comparison only when:

- every model row for that subject converged;
- AIC and BIC are finite;
- each model appears exactly once;
- at least two distinct models are present.

Eligible rows receive `delta_aic` and `delta_bic`, defined relative to the minimum criterion value within that subject. `best_aic` and `best_bic` indicate the model or models attaining zero delta.

## Output files

A primary run writes a timestamped directory containing:

```text
model_fits_<n>_subjects_<timestamp>.csv
model_comparison_<n>_subjects_<timestamp>.csv
model_summary_<n>_subjects_<timestamp>.csv
```

When `--analyze` is supplied, the standard analysis artifacts are additionally written under an `analysis/` subdirectory of the same fitting run.
