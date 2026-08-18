# Scripts

The `scripts/` directory contains the project's user-facing secondary command-line workflows. These modules are siblings of the `igt` package and are run from the repository root with Python's `-m` module form.

```text
scripts/
├── correct_pvl_delta_fits.py
├── q_inverse_temperature_sensitivity.py
└── results_analysis.py
```

Because `scripts/` is an implicit namespace package, it does not require an `__init__.py` file.

## Available scripts

| Script | Purpose | Related workflow |
|---|---|---|
| `scripts.correct_pvl_delta_fits` | Refit selected PVL-Delta solutions using a Q-learning-equivalent warm start and regenerate corrected result tables. | [PVL-Delta correction](../workflow/correction.md) |
| `scripts.q_inverse_temperature_sensitivity` | Refit selected Q-learning subjects under predefined inverse-temperature sensitivity configurations. | [Q inverse-temperature sensitivity](../workflow/sensitivity.md) |
| `scripts.results_analysis` | Validate a completed result triplet and generate the final descriptive and inferential analysis artifacts. | [Final result analysis](../workflow/analysis.md) |

## Invocation convention

Run scripts from the repository root, for example:

```bash
uv run python -m scripts.results_analysis --help
```

The script pages in this section document the command-line interface. The corresponding **Workflow** pages explain the scientific or methodological purpose of each workflow, and [API Reference → Entry points and scripts](../api/entry-points.md) documents the underlying Python modules and callables.
