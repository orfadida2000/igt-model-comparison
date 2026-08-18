# Q-learning inverse-temperature sensitivity

## Purpose

Q-learning inverse temperature can accumulate at an imposed upper bound. The sensitivity workflow isolates converged subjects whose fitted inverse temperature reaches a specified threshold and reruns Q-learning under the project's predefined inverse-temperature-maximum and start-count configurations.

This provides a diagnostic check on whether a finite inverse-temperature ceiling is materially influencing those fits.

## Command

```bash
uv run python -m scripts.q_inverse_temperature_sensitivity \
  path/to/model_fits_617_subjects_<timestamp>.csv
```

For the complete command-line interface, see [Q inverse-temperature sensitivity script](../scripts/q-inverse-temperature-sensitivity.md).

Important options include:

```text
--selection-threshold
--rdata-path
--workers
--output-dir
--logging-dir
--log-level
--no-progress
```

## Subject selection

Selection uses the fitted `inverse_temperature` from the Q-learning row, requires convergence by default, and returns the canonical participant key columns `(n_trials, subject_id)`.

## Outputs

The script saves:

- the selected participant keys;
- one Q-learning fit table for each configured sensitivity condition;
- an application log when logging is enabled.

Every run is placed in its own timestamped subdirectory under `assets/results/q_inverse_temperature_sensitivity/` by default.
