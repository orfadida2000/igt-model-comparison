# IGT Model Comparison

This documentation describes the final state of the **IGT Model Comparison** project: a participant-level computational comparison of Q-learning and PVL-Delta on Iowa Gambling Task data.

## Project scope

The implementation covers the complete workflow:

1. load and validate the bundled IGT RData dataset;
2. convert each participant to validated trial-level model input;
3. fit Q-learning and PVL-Delta with model-specific multi-start initialization;
4. compute likelihood, AIC, BIC, convergence, uniform-choice, and boundary diagnostics;
5. correct rare PVL-Delta local-optimization failures using Q-learning-equivalent warm starts;
6. test Q-learning inverse-temperature sensitivity for capped solutions;
7. validate final result tables and generate descriptive and inferential analyses;
8. export timestamped CSV, LaTeX-table, text, log, PNG, and PDF figure artifacts.

## Final written report

The completed course report is maintained separately from the MkDocs site under `report/`. The LaTeX source, report-specific figures and tables, bibliography, and compiled PDF are kept together so the academic deliverable can be read or rebuilt without changing the Python package.

- Repository source directory: `report/`
- Compiled deliverable: `report/report.pdf`
- [Open the final PDF on GitHub](https://github.com/orfadida2000/igt-model-comparison/blob/main/report/report.pdf)

The Python analysis pipeline does **not** generate the written report itself. It generates the validated numerical, tabular, graphical, and text artifacts that support the report.

## Main entry points

| Workflow | Command |
|---|---|
| Fit and compare both models | `uv run python -m igt.main` |
| Correct selected PVL-Delta fits | `uv run python -m scripts.correct_pvl_delta_fits ...` |
| Q inverse-temperature sensitivity | `uv run python -m scripts.q_inverse_temperature_sensitivity ...` |
| Analyze final result tables | `uv run python -m scripts.results_analysis ...` |

## Where to go next

- [Getting started](getting-started.md) — environment setup and first run.
- [Project structure](project-structure.md) — package and workflow architecture.
- [Data](data.md) — input data and participant identifiers.
- [Models](models/index.md) — model equations and parameterization.
- [Scripts](scripts/index.md) — command-line interfaces for the secondary workflows.
- [Fitting and comparison](workflow/fitting.md) — primary optimization pipeline.
- [PVL-Delta correction](workflow/correction.md) — nested-model warm-start correction.
- [Sensitivity analysis](workflow/sensitivity.md) — inverse-temperature ceiling checks.
- [Result analysis](workflow/analysis.md) — validation, derived tables, figures, inference, and the analysis text report.
- [Statistical inference](statistical-inference.md) — bootstrap, Wilcoxon, and exact binomial analyses.
- [Final results](final-results.md) — results included in the repository and their relation to the final written report.
- [Final report](report.md) — final PDF, LaTeX source layout, build instructions, and relation to the final analysis.
- [API reference](api/index.md) — generated documentation for implementation modules.
