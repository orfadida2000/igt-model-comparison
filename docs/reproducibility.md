# Reproducibility

## Fixed random seeds

The project-wide fixed seed is 42.

It is used for the default scrambled Sobol initialization of PVL-Delta and, through the analysis script's current fixed-seed configuration, for BCa bootstrap resampling.

## Timestamped artifacts

Primary fitting, correction, sensitivity, and standalone analysis runs create timestamped subdirectories using:

```text
YYYY-MM-DD_HH-MM-SS
```

This preserves previous runs rather than overwriting them.

## Model initialization

- Q-learning starts are deterministic for fixed model configuration and subject data because they are selected from a deterministic objective grid.
- PVL-Delta starts are reproducible under the fixed Sobol seed.
- The correction script adds deterministic Q-equivalent warm starts derived from previously fitted Q-learning parameters.

## Parallel execution

Subject fits are independent. Parallel execution preserves the original subject-task ordering when collecting results, so changing the number of workers should not intentionally change result-table ordering.

## Documentation build

The MkDocs configuration uses Google-style docstrings through `mkdocstrings`. Build with:

```bash
uv run mkdocs build --strict
```

## Final written report

The repository keeps the final LaTeX source and compiled PDF under `report/`. The report is based on the corrected 617-participant result set and the final downstream analysis documented in [Final results](final-results.md).

There are two distinct levels of reproducibility:

1. **Document build** — compile the checked-in LaTeX source, figures, and tables to reproduce `report/report.pdf`:

   ```bash
   cd report
   latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
   ```

2. **Scientific regeneration** — rerun the primary fits, targeted PVL-Delta correction, and final analysis from the project data to regenerate the numerical and graphical evidence underlying the report. Those workflows are documented in [Getting started](getting-started.md), [PVL-Delta correction](workflow/correction.md), and [Final result analysis](workflow/analysis.md).

The compiled PDF is kept in the repository because it is a final project deliverable, not merely an intermediate LaTeX build artifact. Auxiliary LaTeX build files are not part of the scientific record.
