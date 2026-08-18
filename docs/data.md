# Data

## Input dataset

The bundled original dataset is:

```text
assets/data/IGTdata.rdata
```

The project also includes a processed long-format table:

```text
assets/data/processed/igt_long.csv
```

The configured dataset contains **617 available subjects**.

## Participant identity

A participant is identified throughout the pipeline by the composite key:

```text
(n_trials, subject_id)
```

The source study is stored separately in `source_study`. Model-level result rows extend the participant key with the `model` column.

These shared names are defined in `igt.constants.schema` and reused throughout fitting, correction, selection, and analysis.

## Long-format trial representation

The preprocessing layer validates the original RData objects and produces per-trial subject data containing:

- deck choice;
- win amount;
- loss amount;
- source-study metadata;
- subject and trial-count identifiers.

The model-facing [`SubjectData`][igt.models.typing.SubjectData] object stores choice, win, and loss arrays, validates their lengths and values, and exposes net outcomes as `wins + losses`.

## Payoff scaling

Both computational models divide net monetary outcomes by the shared `PAYOFF_SCALE` value of 100 before learning. This keeps learned values and sensitivity parameters on a numerically manageable scale while preserving the ordering and sign of objective outcomes.
