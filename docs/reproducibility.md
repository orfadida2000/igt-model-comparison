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
