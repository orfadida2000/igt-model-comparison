# PVL-Delta correction workflow

## Purpose

PVL-Delta contains the objective-outcome Q-learning solution under the implemented parameter mapping. Therefore, after sufficiently good optimization, the best PVL-Delta NLL should not be meaningfully worse than the corresponding Q-learning NLL.

The correction workflow targets subjects where the initial fits violate that expectation.

## Command

```bash
uv run python -m scripts.correct_pvl_delta_fits \
  path/to/model_fits_617_subjects_<timestamp>.csv
```

For the complete command-line interface, see [PVL-Delta fit correction script](../scripts/correct-pvl-delta-fits.md).

Important optional arguments include:

```text
--atol-per-trial
--rdata-path
--workers
--output-dir
--logging-dir
--log-level
--no-progress
```

The default NLL absolute tolerance is `1e-8` **per trial**. The subject-specific tolerance is therefore:

\[
\mathrm{atol}_{subject} = n_{trials}\times 10^{-8}
\]

## Selection

The script uses the shared subject-selection layer to identify fully converged subjects for whom Q-learning is uniquely NLL-best beyond the configured tolerance.

## Q-equivalent PVL-Delta warm start

For each selected subject, the fitted Q-learning parameters are mapped into PVL-Delta as:

```text
PVL learning_rate        = Q learning_rate
PVL outcome_sensitivity  = 1
PVL loss_aversion        = 1
PVL response_consistency = log(Q inverse_temperature + 1) / log(3)
```

That mapped vector is supplied in addition to the ordinary 32 PVL-Delta Sobol starts, so corrected targeted fits normally report 33 starts.

## Replacement and integrity checks

Only the targeted PVL-Delta fit rows are replaced. The workflow verifies that:

- table shapes are preserved;
- columns and their order are preserved;
- participant-model row identities and order are preserved;
- all non-targeted rows remain unchanged;
- targeted results correspond exactly to the selected subjects;
- corrected NLLs improve meaningfully when expected;
- corrected PVL-Delta fits satisfy the nesting comparison against Q-learning.

## Outputs

The timestamped correction directory contains:

```text
model_fits_corrected_<n>_subjects_<timestamp>.csv
model_comparison_corrected_<n>_subjects_<timestamp>.csv
model_summary_corrected_<n>_subjects_<timestamp>.csv
model_correction_report_<n>_subjects_<timestamp>.txt
```

The final included correction run targeted eight subjects and resolved all eight without altering non-targeted rows. See [Final results](../final-results.md).
