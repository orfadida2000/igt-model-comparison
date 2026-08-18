# PVL-Delta fit correction script

Module:

```text
scripts.correct_pvl_delta_fits
```

This is the command-line entry point for the targeted PVL-Delta correction workflow. For the methodological rationale, Q-learning-equivalent parameter mapping, replacement rules, and audit checks, see [PVL-Delta correction workflow](../workflow/correction.md).

## Usage

```bash
uv run python -m scripts.correct_pvl_delta_fits \
  path/to/model_fits_617_subjects_<timestamp>.csv
```

The input fit-results CSV is positional.

## Arguments

| Argument | Purpose |
|---|---|
| `fit-results-path` | Input fit-results CSV used for subject selection, Q-learning warm-start construction, and corrected output generation. |
| `--atol-per-trial` | Per-trial absolute NLL tolerance used when deciding whether Q-learning is meaningfully better than PVL-Delta. Default: `1e-8`. |
| `--rdata-path` | IGT RData source used to reconstruct selected subjects' trial data. |
| `--workers` | Worker-process count. `0` uses serial execution; a negative value requests all available CPU cores. |
| `--output-dir` | Root directory for corrected result artifacts under the current project configuration. |
| `--logging-dir` | Root directory for correction-workflow log files under the current project configuration. |
| `--log-level` | Root logger level; a negative value disables logging. |
| `--no-progress` | Disable fitting progress-bar output. |

Some CLI options are conditionally exposed by shared configuration constants. The command's `--help` output is authoritative for the current configuration.

## Default output location

With the current configuration, corrected artifacts are written beneath:

```text
assets/results/corrected_pvl_delta_fits/
```

Each run receives its own timestamped subdirectory.

The workflow writes corrected fit, comparison, and summary CSVs together with a correction audit report. See [Result files](../results.md) for the repository-wide output conventions.

## Related documentation

- [PVL-Delta correction workflow](../workflow/correction.md)
- [Final results](../final-results.md)
- [Entry points and scripts API](../api/entry-points.md)
