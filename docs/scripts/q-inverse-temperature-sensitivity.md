# Q inverse-temperature sensitivity script

Module:

```text
scripts.q_inverse_temperature_sensitivity
```

This script runs the Q-learning inverse-temperature sensitivity workflow. For why the analysis is performed and how selected fits are interpreted, see [Q-learning inverse-temperature sensitivity](../workflow/sensitivity.md).

## Usage

```bash
uv run python -m scripts.q_inverse_temperature_sensitivity \
  path/to/model_fits_617_subjects_<timestamp>.csv
```

The input fit-results CSV is positional.

## Arguments

| Argument | Purpose |
|---|---|
| `fit-results-path` | Input fit-results CSV used to select converged Q-learning fits for sensitivity reruns. |
| `--selection-threshold` | Inverse-temperature threshold for selecting Q-learning fits. Fits at or above the threshold are selected. |
| `--rdata-path` | IGT RData source used to reconstruct selected subjects' trial data. |
| `--workers` | Worker-process count. `0` uses serial execution; a negative value requests all available CPU cores. |
| `--output-dir` | Root directory for sensitivity-result artifacts under the current project configuration. |
| `--logging-dir` | Root directory for sensitivity-workflow log files under the current project configuration. |
| `--log-level` | Root logger level; a negative value disables logging. |
| `--no-progress` | Disable fitting progress-bar output. |

Some CLI options are conditionally exposed by shared configuration constants. Use `--help` to inspect the exact interface for the current configuration.

## Default output location

With the current configuration, sensitivity artifacts are written beneath:

```text
assets/results/q_inverse_temperature_sensitivity/
```

Each run receives its own timestamped subdirectory.

## Related documentation

- [Q-learning inverse-temperature sensitivity](../workflow/sensitivity.md)
- [Q-learning model](../models/q-learning.md)
- [Entry points and scripts API](../api/entry-points.md)
