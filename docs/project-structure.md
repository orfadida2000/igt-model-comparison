# Project structure

The repository separates reusable implementation code from executable research workflows.

```text
igt/
├── analysis/        # Final-result validation, derived tables, inference, plots, report
├── cli_parsing/     # Shared declarative CLI specification and type-filter system
├── constants/       # Configuration, model, path, fitting, and schema constants
├── execution/       # Subject task construction, optimization, multiprocessing, pipeline
├── models/          # Base model interface, Q-learning, PVL-Delta, subject data
├── notify/          # Optional FormSubmit notifications
├── comparison.py    # AIC/BIC comparison and model-level summary
├── initialization.py # Grid/Sobol start generation and local-minimum selection
├── logging.py       # Application logging configuration
├── main.py          # Primary fitting/comparison command-line workflow
├── rdata_preprocessing.py
├── subject_selection.py
├── typing.py
└── utils/
```

The sibling `scripts/` directory contains executable, project-specific secondary workflows:

```text
scripts/
├── correct_pvl_delta_fits.py
├── q_inverse_temperature_sensitivity.py
└── results_analysis.py
```

Because `scripts/` is used as an implicit namespace package, these are invoked with `python -m scripts.<module>` from the repository root.

See [Scripts](scripts/index.md) for the user-facing CLI reference for each secondary workflow.

## Architectural boundaries

### `igt.models`

Owns computational assumptions: parameter order, bounds, likelihood evaluation, and model-specific starting-point generation.

### `igt.execution`

Owns the mechanics of fitting: subject tasks, L-BFGS-B optimization, warm starts, multiprocessing, convergence diagnostics, and conversion to flat result rows.

### `igt.comparison`

Adds model-comparison eligibility, AIC/BIC deltas, and winner flags, then summarizes valid comparisons by model.

### `igt.subject_selection`

Owns reusable participant-selection logic used by secondary workflows. It validates participant keys, convergence, model coverage, and NLL-based or parameter-based selection conditions.

### `igt.analysis`

Is a **subpackage of `igt`**, not a standalone package. It assumes completed result CSVs already exist. The command-line executable lives in `scripts/results_analysis.py`, while reusable analysis logic remains in `igt.analysis`.

### `igt.cli_parsing`

Provides the shared `ArgSpec` abstraction and registered type-filter system used by the main application and secondary scripts.

## Timestamped outputs

Each top-level result-producing workflow appends a timestamp directory using:

```text
%Y-%m-%d_%H-%M-%S
```

This keeps independent runs reproducible and prevents later executions from overwriting earlier artifacts.
