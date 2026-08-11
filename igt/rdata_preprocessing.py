"""Preprocessing utilities for the Steingroever et al. IGT dataset.

This module reads the original `IGTdata.rdata` file directly with
`pyreadr` and converts the contained wide-format objects into one clean
long-format pandas DataFrame suitable for model fitting.
"""

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
import pyreadr

from igt.constants.path import DATA_DIR
from igt.constants.schema import (
    N_TRIALS_COLUMN,
    PARTICIPANT_KEY_COLUMNS,
    SOURCE_STUDY_COLUMN,
    SUBJECT_ID_COLUMN,
)
from igt.typing import LineEnding, StrPathLike
from igt.utils.io import normalize_path, write_csv

DECK_LABELS: Mapping[int, str] = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
}

EXPECTED_OBJECT_PREFIXES: tuple[str, ...] = (
    "choice",
    "wi",
    "lo",
    "index",
)

OBJECT_NAME_RE = re.compile(
    r"^(choice|wi|lo|index)_(\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IGTBundle:
    """In-memory data objects for one trial-count group."""

    n_trials: int
    choice: pd.DataFrame
    wi: pd.DataFrame
    lo: pd.DataFrame
    index: pd.DataFrame


def load_rdata_objects(rdata_path: StrPathLike) -> dict[str, pd.DataFrame]:
    """Load all pandas DataFrames from the IGT RData file."""

    rdata_path = normalize_path(rdata_path)

    if not rdata_path.exists():
        raise FileNotFoundError(f"RData file does not exist: {rdata_path}")

    if not rdata_path.is_file():
        raise ValueError(f"Expected a file, got: {rdata_path}")

    raw_objects = pyreadr.read_r(rdata_path)
    objects: dict[str, pd.DataFrame] = {}

    for object_name, value in raw_objects.items():
        if value is None:
            continue

        if not isinstance(value, pd.DataFrame):
            raise TypeError(
                f"Expected object {object_name!r} to be a pandas DataFrame, "
                f"got {type(value).__name__}"
            )

        objects[str(object_name)] = value.copy()

    if not objects:
        raise ValueError(f"No pandas DataFrame objects found in: {rdata_path}")

    return objects


def collect_bundles(
    objects: Mapping[str, pd.DataFrame],
) -> dict[int, IGTBundle]:
    """Group loaded R objects into complete trial-count bundles."""

    grouped: dict[int, dict[str, pd.DataFrame]] = {}

    for object_name, dataframe in objects.items():
        match = OBJECT_NAME_RE.fullmatch(object_name)

        if match is None:
            continue

        prefix = match.group(1).lower()
        n_trials = int(match.group(2))
        grouped.setdefault(n_trials, {})[prefix] = dataframe

    if not grouped:
        raise ValueError(
            "No supported IGT objects were found. Expected names such as "
            "'choice_100', 'wi_100', 'lo_100', and 'index_100'."
        )

    bundles: dict[int, IGTBundle] = {}

    for n_trials, objects_by_prefix in sorted(grouped.items()):
        missing = sorted(set(EXPECTED_OBJECT_PREFIXES) - set(objects_by_prefix))

        if missing:
            missing_names = ", ".join(f"{prefix}_{n_trials}" for prefix in missing)
            raise ValueError(
                f"Incomplete object bundle for n_trials={n_trials}. Missing: {missing_names}"
            )

        bundles[n_trials] = IGTBundle(
            n_trials=n_trials,
            choice=objects_by_prefix["choice"],
            wi=objects_by_prefix["wi"],
            lo=objects_by_prefix["lo"],
            index=objects_by_prefix["index"],
        )

    return bundles


def validate_bundle(bundle: IGTBundle) -> None:
    """Validate the structure and alignment of one IGT bundle."""

    choice_shape = bundle.choice.shape

    if bundle.wi.shape != choice_shape:
        raise ValueError(
            f"Shape mismatch for n_trials={bundle.n_trials}: "
            f"choice={choice_shape}, wi={bundle.wi.shape}"
        )

    if bundle.lo.shape != choice_shape:
        raise ValueError(
            f"Shape mismatch for n_trials={bundle.n_trials}: "
            f"choice={choice_shape}, lo={bundle.lo.shape}"
        )

    if choice_shape[1] != bundle.n_trials:
        raise ValueError(
            f"Trial-count mismatch for n_trials={bundle.n_trials}: "
            f"choice matrix has {choice_shape[1]} columns"
        )

    if len(bundle.index) != choice_shape[0]:
        raise ValueError(
            f"Subject-count mismatch for n_trials={bundle.n_trials}: "
            f"choice has {choice_shape[0]} rows, "
            f"index has {len(bundle.index)} rows"
        )

    required_index_columns = {"Subj", "Study"}
    missing_index_columns = required_index_columns - set(bundle.index.columns)

    if missing_index_columns:
        missing_text = ", ".join(sorted(missing_index_columns))
        raise ValueError(
            f"Index object for n_trials={bundle.n_trials} is missing "
            f"required columns: {missing_text}"
        )

    choice_array = bundle.choice.to_numpy(dtype=np.float64, copy=False)

    if not np.isfinite(choice_array).all():
        raise ValueError(
            f"Choice matrix for n_trials={bundle.n_trials} contains missing or non-finite values"
        )

    integer_choices = choice_array.astype(np.int64)

    if not np.array_equal(choice_array, integer_choices):
        raise ValueError(
            f"Choice matrix for n_trials={bundle.n_trials} contains non-integer values"
        )

    valid_choice_codes = np.array(tuple(DECK_LABELS), dtype=np.int64)
    valid_choice_mask = np.isin(integer_choices, valid_choice_codes)

    if not valid_choice_mask.all():
        invalid_values = np.unique(integer_choices[~valid_choice_mask])
        raise ValueError(
            f"Choice matrix for n_trials={bundle.n_trials} contains "
            f"invalid deck codes: {invalid_values.tolist()}"
        )

    wi_array = bundle.wi.to_numpy(dtype=np.float64, copy=False)
    lo_array = bundle.lo.to_numpy(dtype=np.float64, copy=False)

    if not np.isfinite(wi_array).all():
        raise ValueError(
            f"Win matrix for n_trials={bundle.n_trials} contains missing or non-finite values"
        )

    if not np.isfinite(lo_array).all():
        raise ValueError(
            f"Loss matrix for n_trials={bundle.n_trials} contains missing or non-finite values"
        )

    subject_ids = pd.to_numeric(
        bundle.index["Subj"],
        errors="raise",
    ).astype(np.int64)

    if subject_ids.duplicated().any():
        duplicate_ids = (
            subject_ids[subject_ids.duplicated(keep=False)].sort_values().unique().tolist()
        )
        raise ValueError(f"Duplicate subject IDs for n_trials={bundle.n_trials}: {duplicate_ids}")

    study_names = bundle.index["Study"].astype("string").str.strip()

    if study_names.isna().any() or study_names.eq("").any():
        raise ValueError(f"Missing study names for n_trials={bundle.n_trials}")


def bundle_to_long_table(bundle: IGTBundle) -> pd.DataFrame:
    """Convert one validated IGT bundle to long trial-level format."""

    validate_bundle(bundle)

    choice_array = bundle.choice.to_numpy(dtype=np.int64, copy=False)
    wi_array = bundle.wi.to_numpy(dtype=np.float64, copy=False)
    lo_array = bundle.lo.to_numpy(dtype=np.float64, copy=False)

    subject_ids = (
        pd.to_numeric(bundle.index["Subj"], errors="raise").astype(np.int64).to_numpy(copy=False)
    )

    study_names = bundle.index["Study"].astype("string").str.strip().to_numpy(dtype=str, copy=False)

    n_subjects, n_trials = choice_array.shape

    subject_id_column = np.repeat(subject_ids, n_trials)
    n_trials_column = np.full(
        n_subjects * n_trials,
        bundle.n_trials,
        dtype=np.int64,
    )
    source_study_column = np.repeat(study_names, n_trials)
    trial_column = np.tile(
        np.arange(1, n_trials + 1, dtype=np.int64),
        n_subjects,
    )

    choice_column = choice_array.reshape(-1)
    win_column = wi_array.reshape(-1)
    loss_column = lo_array.reshape(-1)

    deck_column = np.array(
        [DECK_LABELS[int(choice)] for choice in choice_column],
        dtype=str,
    )

    return pd.DataFrame(
        {
            SUBJECT_ID_COLUMN: subject_id_column,
            N_TRIALS_COLUMN: n_trials_column,
            SOURCE_STUDY_COLUMN: source_study_column,
            "trial": trial_column,
            "choice": choice_column,
            "deck": deck_column,
            "win": win_column,
            "loss": loss_column,
            "net_outcome": win_column + loss_column,
        }
    )


def load_igt_long_table(rdata_path: StrPathLike) -> pd.DataFrame:
    """Load the RData file and return one combined long-format table."""

    objects = load_rdata_objects(rdata_path)
    bundles = collect_bundles(objects)

    tables = [bundle_to_long_table(bundle) for bundle in bundles.values()]

    data = pd.concat(tables, ignore_index=True)

    return data.sort_values(
        by=[*PARTICIPANT_KEY_COLUMNS, "trial"],
        kind="mergesort",
        ignore_index=True,
    )


def save_igt_long_table(
    rdata_path: StrPathLike,
    output_csv: StrPathLike,
) -> pd.DataFrame:
    """Load, preprocess, and save the complete IGT long-format table."""

    data = load_igt_long_table(rdata_path)
    write_csv(
        data,
        output_csv,
        index=False,
        newline=LineEnding.LF,
    )

    return data


def iter_subject_trials(
    data: pd.DataFrame,
    n_subjects: int | None = None,
) -> Iterator[tuple[tuple[int, int], pd.DataFrame]]:
    """Yield each subject's trials in chronological order."""
    if n_subjects is not None:
        if n_subjects < 0:
            raise ValueError("n_subjects must be greater than or equal to zero.")
        if n_subjects == 0:
            return  # No subjects requested, yield nothing.

    required_columns = {*PARTICIPANT_KEY_COLUMNS, "trial"}
    missing = required_columns - set(data.columns)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")

    grouped = data.groupby(
        [*PARTICIPANT_KEY_COLUMNS],
        sort=True,
    )

    i_subject = 1

    for raw_key, subject_trials in grouped:
        n_trials_raw, subject_id_raw = cast(
            tuple[int | np.integer, int | np.integer],
            raw_key,
        )

        subject_key = (
            int(n_trials_raw),
            int(subject_id_raw),
        )

        ordered_trials = subject_trials.sort_values(
            by="trial",
            kind="mergesort",
            ignore_index=True,
        )

        yield subject_key, ordered_trials

        if n_subjects is not None and i_subject >= n_subjects:
            break
        i_subject += 1


def main() -> None:
    """Run preprocessing with the default project paths."""

    rdata_path = DATA_DIR / "IGTdata.rdata"
    output_csv = DATA_DIR / "processed" / "igt_long.csv"

    data = save_igt_long_table(
        rdata_path=rdata_path,
        output_csv=output_csv,
    )

    subjects = data[[*PARTICIPANT_KEY_COLUMNS]].drop_duplicates()  # pyright: ignore[reportArgumentType]

    total_subjects = len(subjects)
    total_studies = subjects[SOURCE_STUDY_COLUMN].nunique()

    subjects_n_per_study = (
        subjects.groupby(SOURCE_STUDY_COLUMN, sort=False).size().rename("n_subjects")
    )

    print(f"Saved: {output_csv}")
    print(f"Trial Rows: {len(data)}")
    print(f"Total Subjects: {total_subjects}")
    print(f"Total Studies: {total_studies}")

    print("\nSubjects per study:")
    print(subjects_n_per_study)

    print("\nFirst rows:")
    print(data.head())


if __name__ == "__main__":
    main()
