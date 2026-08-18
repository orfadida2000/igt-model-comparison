"""Targeted correction workflow for PVL-Delta fits that violate model nesting numerically.

The script identifies participants for whom Q-learning is meaningfully NLL-better,
maps each fitted Q-learning solution to a theoretically equivalent PVL-Delta warm
start, refits only those PVL-Delta rows, replaces the corrected results, and writes an
audit report plus corrected fit/comparison/summary tables.
"""

import argparse
import logging
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from igt.cli_parsing.parser import get_parser
from igt.cli_parsing.type_filters.factory.path import (
    get_type_filters_for_existing_file_with_extensions_path,
)
from igt.cli_parsing.type_filters.presets import (
    NumericArgTypeProvider,
    PathArgTypeProvider,
    StringArgTypeProvider,
)
from igt.cli_parsing.typing import (
    ArgAction,
    ArgSpec,
)
from igt.comparison import (
    add_model_comparison_columns,
    summarize_model_comparison,
)
from igt.constants.config import (
    DEFAULT_N_PVL_STARTS,
    DEFAULT_N_WORKERS,
    DEFAULT_NOTIFY_FORMSUBMIT_ID,
    DEFAULT_ROOT_LOG_LEVEL,
    FILENAME_DATETIME_FMT,
    FIXED_SEED,
    USE_FIXED_NOTIFY_FORMSUBMIT_ID,
    USE_RDATA_PARENT_DIR_FOR_LOGGING,
    USE_RDATA_PARENT_DIR_FOR_OUTPUT,
)
from igt.constants.fitting import DEFAULT_MAX_ITERATIONS
from igt.constants.models import (
    INVERSE_TEMPERATURE_PARAMETER_NAME,
    LEARNING_RATE_PARAMETER_NAME,
    MAX_LEARNING_RATE,
    MIN_INVERSE_TEMPERATURE,
    MIN_LEARNING_RATE,
    RESPONSE_CONSISTENCY_PARAMETER_NAME,
)
from igt.constants.path import IGT_DATASET_PATH, LOGS_DIR, RESULTS_DIR
from igt.constants.schema import (
    CONVERGED_COLUMN,
    MODEL_COLUMN,
    N_TRIALS_COLUMN,
    NLL_COLUMN,
    PARTICIPANT_KEY_COLUMNS,
)
from igt.execution.pipeline import (
    FittingPipelineConfig,
    run_fitting_pipeline,
)
from igt.execution.typing import SubjectModelWarmStartsProvider
from igt.logging import (
    application_logging_cleanup,
    configure_application_logging,
)
from igt.models import PVLDeltaModel, QLearningModel
from igt.notify.formsubmit import (
    error_email_notifier,
    send_formsubmit_email_script_success_notification,
)
from igt.subject_selection import (
    _validate_nonnegative_finite_float,
    normalize_subject_key_columns,
    select_subjects_with_target_is_uniquely_nll_best_model,
)
from igt.typing import Float2DArray, StrPathLike
from igt.utils.io import normalize_path, read_csv
from igt.utils.tabular import normalize_nonempty_string_series

DEFAULT_SELECTION_ATOL: Final[float] = 1e-8

PVL_OUTCOME_SENSITIVITY: Final[float] = 1.0
PVL_LOSS_AVERSION: Final[float] = 1.0

LOGGER_NAME: Final[str] = "scripts.correct_pvl_delta_fits"


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for targeted PVL-Delta fit correction.

    The parser uses the project's declarative argument/type-filter infrastructure and
    conditionally exposes seed and FormSubmit options according to the corresponding
    fixed-value configuration flags.

    Returns:
        Raw argparse namespace containing the correction-workflow options.
    """

    arg_specs: list[ArgSpec] = [
        ArgSpec(
            name_or_flags="fit-results-path",
            type_filters=get_type_filters_for_existing_file_with_extensions_path(".csv"),
            help="Path to the input CSV file containing previous model fitting results used for subject selection, warm starts construction, and corrected output generation.",
            extra_options={
                "metavar": "FILE_PATH",
            },
        ),
        ArgSpec(
            name_or_flags=("--atol-per-trial",),
            type_filters=(NumericArgTypeProvider.NON_NEGATIVE_FINITE_FLOAT,),
            default=str(DEFAULT_SELECTION_ATOL),
            help=(
                "Absolute tolerance for considering two negative log-likelihood (NLL) values as equal for 1 trial when selecting subjects for PVL-Delta refitting. "
                "The effective absolute tolerance for a subject with N trials is N * atol_per_trial; "
                "must be a non-negative finite float (default: %(default)s)"
            ),
            extra_options={
                "metavar": "TOLERANCE",
            },
        ),
        ArgSpec(
            name_or_flags=("--rdata-path",),
            type_filters=get_type_filters_for_existing_file_with_extensions_path(
                extensions=(".rdata", ".rda")
            ),
            default=str(IGT_DATASET_PATH),
            help="Path to the input IGTdata.rdata file (default: %(default)s)",
            extra_options={
                "metavar": "FILE_PATH",
            },
        ),
        ArgSpec(
            name_or_flags=("--workers",),
            type_filters=(NumericArgTypeProvider.INTEGER,),
            default=str(DEFAULT_N_WORKERS),
            help=(
                "Number of worker processes to use for the refitting process; "
                "use 0 for serial execution and negative value for all available CPU cores (default: %(default)s)"
            ),
            extra_options={
                "metavar": "N_WORKERS",
                "dest": "n_workers",
            },
        ),
    ]

    if not USE_RDATA_PARENT_DIR_FOR_OUTPUT:
        arg_specs.append(
            ArgSpec(
                name_or_flags=("--output-dir",),
                type_filters=(PathArgTypeProvider.DIR_PATH,),
                default=str(RESULTS_DIR / "corrected_pvl_delta_fits"),
                help="Directory to save the results files (default: %(default)s)",
                extra_options={
                    "metavar": "OUTPUT_DIR",
                },
            )
        )

    if not USE_RDATA_PARENT_DIR_FOR_LOGGING:
        arg_specs.append(
            ArgSpec(
                name_or_flags=("--logging-dir",),
                type_filters=(PathArgTypeProvider.DIR_PATH,),
                default=str(LOGS_DIR / "corrected_pvl_delta_fits"),
                help="Directory to save the log files (default: %(default)s)",
                extra_options={
                    "metavar": "LOGGING_DIR",
                },
            )
        )

    arg_specs.append(
        ArgSpec(
            name_or_flags=("--log-level",),
            type_filters=(NumericArgTypeProvider.INTEGER,),
            default=str(DEFAULT_ROOT_LOG_LEVEL),
            help="Logging level for the root logger; use negative value to disable logging (default: %(default)s)",
            extra_options={
                "metavar": "ROOT_LOG_LEVEL",
            },
        )
    )

    if not USE_FIXED_NOTIFY_FORMSUBMIT_ID:
        arg_specs.append(
            ArgSpec(
                name_or_flags=("--notify-formsubmit-id",),
                type_filters=(StringArgTypeProvider.ALPHANUMERIC_STRING,),
                default=DEFAULT_NOTIFY_FORMSUBMIT_ID,
                help="FormSubmit ID to send email notifications upon script completion (default: %(default)s)",
                extra_options={
                    "metavar": "ID",
                },
            )
        )

    arg_specs.append(
        ArgSpec(
            name_or_flags=("--no-progress",),
            action=ArgAction.STORE_TRUE,
            help="Disable progress bar display during the fitting process",
        )
    )

    (
        parser,
        resolved_info,
    ) = get_parser(
        arg_specs,
        description=(
            "Select subjects for whom Q-learning has the lowest (best) NLL, "
            "refit their PVL-Delta model using an additional "
            "Q-learning-equivalent warm start, replace those rows in the "
            "complete fit-results table, and regenerate the model "
            "comparison and summary tables."
        ),
        extra_options={},
    )

    return parser.parse_args()


def _normalize_args(
    args: argparse.Namespace,
) -> argparse.Namespace:
    """Resolve raw correction-script arguments into normalized runtime values.

    Defaults and conditional fixed settings are applied, path-like values are normalized,
    and the resulting values are copied into a fresh namespace used by the workflow.

    Args:
        args: Raw namespace returned by `_parse_args`.

    Returns:
        Namespace containing normalized paths, execution settings, tolerances, seed, and
        optional notification configuration.
    """

    normalized_args: dict[str, Any] = {
        "fit_results_path": args.fit_results_path,
        "atol_per_trial": args.atol_per_trial,
        "rdata_path": args.rdata_path,
        "n_workers": None if args.n_workers < 0 else (1 if args.n_workers == 0 else args.n_workers),
        "output_dir": (
            args.output_dir
            if not USE_RDATA_PARENT_DIR_FOR_OUTPUT
            else args.rdata_path.parent / "output_files"
        ),
        "logging_dir": (
            args.logging_dir
            if not USE_RDATA_PARENT_DIR_FOR_LOGGING
            else args.rdata_path.parent / "logs"
        ),
        "logging_disabled": args.log_level < 0,
        "log_level": None if args.log_level < 0 else args.log_level,
        "notify_formsubmit_id": (
            DEFAULT_NOTIFY_FORMSUBMIT_ID
            if USE_FIXED_NOTIFY_FORMSUBMIT_ID
            else args.notify_formsubmit_id
        ),
        "no_progress": args.no_progress,
    }

    return argparse.Namespace(**normalized_args)


def _build_q_equivalent_pvl_warm_starts_provider(
    fit_results: DataFrame,
    subject_keys: DataFrame,
    *,
    pvl_delta_model: PVLDeltaModel,
) -> SubjectModelWarmStartsProvider:
    """Build a provider of Q-learning-equivalent PVL-Delta warm starts.

    For a Q-learning learning rate `a` and inverse temperature `beta`,
    the corresponding PVL-Delta starting point is:

    `(a, 1, 1, log_3(beta + 1))`.

    Args:
        fit_results: Original complete model-fit results.
        subject_keys: Participant keys selected for PVL-Delta refitting.
        pvl_delta_model: PVL-Delta model that will receive the warm starts.

    Returns:
        A callback that supplies one two-dimensional warm-start array for
        each selected PVL-Delta subject.

    Raises:
        TypeError: If either table is not a pandas DataFrame.
        ValueError: If required columns are missing, Q-learning rows are
            duplicated or absent, parameter values are invalid, or a
            mapped warm start lies outside the PVL-Delta bounds.
    """

    if not isinstance(fit_results, pd.DataFrame):
        raise TypeError(
            f"fit_results must be a pandas DataFrame, got {type(fit_results).__name__}."
        )

    if not isinstance(subject_keys, pd.DataFrame):
        raise TypeError(
            f"subject_keys must be a pandas DataFrame, got {type(subject_keys).__name__}."
        )

    key_columns = list(PARTICIPANT_KEY_COLUMNS)

    required_columns = {
        *key_columns,
        MODEL_COLUMN,
        LEARNING_RATE_PARAMETER_NAME,
        INVERSE_TEMPERATURE_PARAMETER_NAME,
    }

    missing_columns = required_columns - set(fit_results.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fit-results table is missing columns: {missing_text}")

    normalized_models = fit_results[MODEL_COLUMN].astype("string").str.strip()

    q_results = fit_results.loc[
        normalized_models.eq(QLearningModel.get_name()),
        [
            *key_columns,
            LEARNING_RATE_PARAMETER_NAME,
            INVERSE_TEMPERATURE_PARAMETER_NAME,
        ],
    ].copy()

    normalized_keys = normalize_subject_key_columns(q_results)

    for column_name in key_columns:
        q_results[column_name] = normalized_keys[column_name]

    duplicate_mask = q_results.duplicated(
        subset=key_columns,
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            "The fit-results table contains "
            f"{duplicate_count} duplicate Q-learning rows for the same "
            "participant keys."
        )

    selected_q_results = subject_keys.merge(
        q_results,
        on=key_columns,
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    missing_q_mask = selected_q_results["_merge"].ne("both")

    if missing_q_mask.any():
        missing_keys = selected_q_results.loc[
            missing_q_mask,
            key_columns,
        ]

        raise ValueError(
            "Selected subjects are missing Q-learning fit results:\n"
            f"{missing_keys.to_string(index=False)}"
        )

    selected_q_results = selected_q_results.drop(columns="_merge")

    try:
        q_learning_rates = pd.to_numeric(
            selected_q_results[LEARNING_RATE_PARAMETER_NAME],
            errors="raise",
        ).to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        q_inverse_temperatures = pd.to_numeric(
            selected_q_results[INVERSE_TEMPERATURE_PARAMETER_NAME],
            errors="raise",
        ).to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Q-learning parameter columns contain values that cannot be interpreted as numbers."
        ) from error

    if not np.isfinite(q_learning_rates).all():
        raise ValueError("Q-learning learning_rate contains missing or non-finite values.")

    if not np.isfinite(q_inverse_temperatures).all():
        raise ValueError("Q-learning inverse_temperature contains missing or non-finite values.")

    if np.any(q_learning_rates < MIN_LEARNING_RATE) or np.any(q_learning_rates > MAX_LEARNING_RATE):
        raise ValueError("Q-learning learning_rate values must be within [0, 1].")

    if np.any(q_inverse_temperatures < MIN_INVERSE_TEMPERATURE):
        raise ValueError("Q-learning inverse_temperature values must be nonnegative.")

    pvl_response_consistencies = np.log1p(q_inverse_temperatures) / np.log(3.0)

    if not np.isfinite(pvl_response_consistencies).all():
        raise ValueError("The mapped PVL-Delta response-consistency values are not finite.")

    selected_q_results[RESPONSE_CONSISTENCY_PARAMETER_NAME] = pvl_response_consistencies

    warm_starts_by_subject: dict[
        tuple[int, int],
        Float2DArray,
    ] = {}

    warm_start_columns = [
        *key_columns,
        LEARNING_RATE_PARAMETER_NAME,
        RESPONSE_CONSISTENCY_PARAMETER_NAME,
    ]

    for (
        n_trials,
        subject_id,
        learning_rate,
        response_consistency,
    ) in selected_q_results.loc[
        :,
        warm_start_columns,
    ].itertuples(
        index=False,
        name=None,
    ):
        subject_key = (
            int(n_trials),
            int(subject_id),
        )

        warm_start = np.array(
            [
                [
                    float(learning_rate),
                    PVL_OUTCOME_SENSITIVITY,
                    PVL_LOSS_AVERSION,
                    float(response_consistency),
                ]
            ],
            dtype=np.float64,
        )

        if not pvl_delta_model.parameters_within_bounds(warm_start[0]):
            raise ValueError(
                "The mapped PVL-Delta warm start for participant key "
                f"{subject_key} is outside the model bounds: "
                f"{warm_start[0].tolist()}"
            )

        warm_starts_by_subject[subject_key] = warm_start

    pvl_delta_model_name = pvl_delta_model.name

    def provide_warm_starts(
        model_name: str,
        n_trials: int,
        subject_id: int,
    ) -> Float2DArray | None:
        """Return the mapped warm start for one subject and model.

        Args:
            model_name: Name of the model being fitted.
            n_trials: Number of trials completed by the subject.
            subject_id: Subject identifier.

        Returns:
            One PVL-Delta warm starting point, or `None` when a
            different model is being fitted or a warm start is not available for the subject.
        """

        if model_name != pvl_delta_model_name:
            return None

        subject_key = (
            n_trials,
            subject_id,
        )

        warm_start = warm_starts_by_subject.get(subject_key, None)

        if warm_start is not None:
            warm_start = warm_start.copy()

        return warm_start

    return provide_warm_starts


def _validate_targeted_pvl_results(
    targeted_results: DataFrame,
    subject_keys: DataFrame,
    *,
    pvl_delta_model_name: str,
) -> None:
    """Validate the targeted PVL-Delta rerun results.

    Args:
        targeted_results: Newly generated PVL-Delta fit rows.
        subject_keys: Participant keys requested for refitting.
        pvl_delta_model_name: Expected model name in every result row.

    Raises:
        TypeError: If either argument is not a pandas DataFrame.
        ValueError: If required columns are missing, result rows are
            duplicated, unexpected models are present, or the result keys
            differ from the requested keys.
    """

    if not isinstance(targeted_results, pd.DataFrame):
        raise TypeError(
            f"targeted_results must be a pandas DataFrame, got {type(targeted_results).__name__}."
        )

    if not isinstance(subject_keys, pd.DataFrame):
        raise TypeError(
            f"subject_keys must be a pandas DataFrame, got {type(subject_keys).__name__}."
        )

    key_columns = list(PARTICIPANT_KEY_COLUMNS)

    required_columns = {
        *key_columns,
        MODEL_COLUMN,
    }

    missing_columns = required_columns - set(targeted_results.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Targeted fit-results table is missing columns: {missing_text}")

    normalized_models = targeted_results[MODEL_COLUMN].astype("string").str.strip()

    if not normalized_models.eq(pvl_delta_model_name).all():
        unexpected_models = sorted(
            normalized_models.loc[~normalized_models.eq(pvl_delta_model_name)]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            f"The targeted fit-results table contains unexpected models: {unexpected_models}"
        )

    targeted_index = pd.MultiIndex.from_frame(targeted_results.loc[:, key_columns])
    requested_index = pd.MultiIndex.from_frame(subject_keys.loc[:, key_columns])

    if not targeted_index.is_unique:
        raise ValueError("The targeted fit-results table contains duplicate participant rows.")

    missing_index = requested_index.difference(targeted_index)
    unexpected_index = targeted_index.difference(requested_index)

    if not missing_index.empty:
        raise ValueError(
            "The targeted fit-results table is missing requested "
            f"participant keys: {missing_index.tolist()}"
        )

    if not unexpected_index.empty:
        raise ValueError(
            "The targeted fit-results table contains unexpected "
            f"participant keys: {unexpected_index.tolist()}"
        )


def _replace_model_fit_rows(
    original_fit_results: DataFrame,
    replacement_fit_results: DataFrame,
) -> DataFrame:
    """Partially update selected model-fit rows.

    Rows are matched using the participant-key columns together with the
    model name. Core fit-result columns must be present in both tables.
    Every column supplied by the replacement table must also exist in the
    original table.

    Only non-key columns present in the replacement table are updated.
    Columns omitted from the replacement table retain their original values.

    Args:
        original_fit_results: Complete original model-fit results.
        replacement_fit_results: Newly fitted rows containing values that
            should update corresponding rows in the original table.

    Returns:
        A copy of the original table with matching rows partially updated.
        The original row order and column order are preserved.

    Raises:
        TypeError: If either argument is not a pandas DataFrame.
        ValueError: If required columns are missing, column names or
            participant-model keys are duplicated, replacement columns do
            not exist in the original table, or replacement rows do not
            exist in the original table.
    """

    if not isinstance(original_fit_results, pd.DataFrame):
        raise TypeError(
            "original_fit_results must be a pandas DataFrame, got "
            f"{type(original_fit_results).__name__}."
        )

    if not isinstance(replacement_fit_results, pd.DataFrame):
        raise TypeError(
            "replacement_fit_results must be a pandas DataFrame, got "
            f"{type(replacement_fit_results).__name__}."
        )

    if original_fit_results.columns.has_duplicates:
        duplicate_columns = original_fit_results.columns[
            original_fit_results.columns.duplicated(keep=False)
        ].tolist()

        raise ValueError(
            f"The original fit-results table contains duplicate column names: {duplicate_columns}"
        )

    if replacement_fit_results.columns.has_duplicates:
        duplicate_columns = replacement_fit_results.columns[
            replacement_fit_results.columns.duplicated(keep=False)
        ].tolist()

        raise ValueError(
            "The replacement fit-results table contains duplicate column names: "
            f"{duplicate_columns}"
        )

    key_columns = [
        *PARTICIPANT_KEY_COLUMNS,
        MODEL_COLUMN,
    ]

    required_fit_columns = {
        *key_columns,
        NLL_COLUMN,
        "log_likelihood",
        "aic",
        "bic",
        CONVERGED_COLUMN,
    }

    missing_original_columns = required_fit_columns - set(original_fit_results.columns)

    if missing_original_columns:
        missing_text = ", ".join(sorted(missing_original_columns))
        raise ValueError(
            f"The original fit-results table is missing required columns: {missing_text}"
        )

    missing_replacement_columns = required_fit_columns - set(replacement_fit_results.columns)

    if missing_replacement_columns:
        missing_text = ", ".join(sorted(missing_replacement_columns))
        raise ValueError(
            f"The replacement fit-results table is missing required columns: {missing_text}"
        )

    unexpected_replacement_columns = set(replacement_fit_results.columns) - set(
        original_fit_results.columns
    )

    if unexpected_replacement_columns:
        unexpected_text = ", ".join(sorted(unexpected_replacement_columns))
        raise ValueError(
            "The replacement fit-results table contains columns that are "
            "not present in the original table: "
            f"{unexpected_text}"
        )

    original = original_fit_results.copy()
    replacements = replacement_fit_results.copy()

    normalized_original_keys = normalize_subject_key_columns(original)

    for column_name in PARTICIPANT_KEY_COLUMNS:
        original[column_name] = normalized_original_keys[column_name]

    normalized_replacement_keys = normalize_subject_key_columns(replacements)

    for column_name in PARTICIPANT_KEY_COLUMNS:
        replacements[column_name] = normalized_replacement_keys[column_name]

    for table_name, table in (
        ("original", original),
        ("replacement", replacements),
    ):
        normalized_models = table[MODEL_COLUMN].astype("string").str.strip()

        if normalized_models.isna().any():
            raise ValueError(f"The {table_name} fit-results table contains missing model names.")

        if normalized_models.eq("").any():
            raise ValueError(f"The {table_name} fit-results table contains empty model names.")

        table[MODEL_COLUMN] = normalized_models

    original_duplicate_mask = original.duplicated(
        subset=key_columns,
        keep=False,
    )

    if original_duplicate_mask.any():
        duplicate_keys = original.loc[
            original_duplicate_mask,
            key_columns,
        ]

        raise ValueError(
            "The original fit-results table contains duplicate "
            "participant-model keys:\n"
            f"{duplicate_keys.to_string(index=False)}"
        )

    replacement_duplicate_mask = replacements.duplicated(
        subset=key_columns,
        keep=False,
    )

    if replacement_duplicate_mask.any():
        duplicate_keys = replacements.loc[
            replacement_duplicate_mask,
            key_columns,
        ]

        raise ValueError(
            "The replacement fit-results table contains duplicate "
            "participant-model keys:\n"
            f"{duplicate_keys.to_string(index=False)}"
        )

    original_indexed = original.set_index(
        key_columns,
        drop=False,
    )

    replacement_indexed = replacements.set_index(
        key_columns,
        drop=False,
    )

    replacement_index = replacement_indexed.index

    missing_row_mask = ~replacement_index.isin(original_indexed.index)

    if missing_row_mask.any():
        missing_rows = replacements.loc[
            missing_row_mask,
            key_columns,
        ]

        raise ValueError(
            "The replacement fit-results table contains rows that do not "
            "exist in the original table:\n"
            f"{missing_rows.to_string(index=False)}"
        )

    replacement_value_columns = [
        column_name for column_name in replacements.columns if column_name not in key_columns
    ]

    corrected_indexed = original_indexed.copy()

    corrected_indexed.loc[
        replacement_index,
        replacement_value_columns,
    ] = replacement_indexed.loc[
        replacement_index,
        replacement_value_columns,
    ].to_numpy()

    corrected = corrected_indexed.reset_index(drop=True)

    return corrected.loc[:, original_fit_results.columns]


def _setup() -> tuple[
    argparse.Namespace,
    argparse.Namespace,
    str,
    DataFrame,
    DataFrame,
    PVLDeltaModel,
    SubjectModelWarmStartsProvider,
    Path | None,
]:
    """Prepare one timestamped targeted PVL-Delta correction run.

    The setup stage parses and normalizes arguments, reads the original fit table,
    selects participants exhibiting the target NLL relation, constructs the PVL-Delta
    model and Q-equivalent warm-start provider, creates the output/logging context, and
    returns the objects required by the execution stage.

    Returns:
        Raw and normalized arguments, run timestamp, original fit table, selected
        participant keys, configured PVL-Delta model, subject-specific warm-start
        provider, and optional log-file path.
    """

    start_datetime_str = datetime.now().strftime(FILENAME_DATETIME_FMT)
    args = _parse_args()
    normalized_args = _normalize_args(args)

    normalized_args.output_dir = Path(normalized_args.output_dir) / start_datetime_str

    original_fit_results = read_csv(
        normalized_args.fit_results_path,
        table_name="fit-results",
    )

    subject_keys = select_subjects_with_target_is_uniquely_nll_best_model(
        original_fit_results,
        target_model=QLearningModel,
        atol_per_trial=normalized_args.atol_per_trial,
        fully_converged=True,
    )

    pvl_delta_model = PVLDeltaModel(
        n_starts=DEFAULT_N_PVL_STARTS,
        rng=FIXED_SEED,
    )

    warm_starts_provider = _build_q_equivalent_pvl_warm_starts_provider(
        original_fit_results,
        subject_keys,
        pvl_delta_model=pvl_delta_model,
    )

    logging_path = configure_application_logging(
        disabled=normalized_args.logging_disabled,
        root_logger_level=normalized_args.log_level,
        log_file_path=(
            normalized_args.logging_dir
            / (f"correct_pvl_delta_fits_{len(subject_keys)}_subjects_{start_datetime_str}.log")
        ),
    )

    return (
        args,
        normalized_args,
        start_datetime_str,
        original_fit_results,
        subject_keys,
        pvl_delta_model,
        warm_starts_provider,
        logging_path,
    )


def _compare_original_and_corrected_fit_results(
    original_fit_results: DataFrame,
    corrected_fit_results: DataFrame,
    subject_keys: DataFrame,
    *,
    report_path: StrPathLike,
    selection_atol_per_trial: float,
    logger: logging.Logger | str = LOGGER_NAME,
) -> None:
    """Validate corrected fit results and write a correction audit report.

    The function verifies that the original and corrected tables have the
    same structure and row identities, and that only the PVL-Delta rows of
    the selected subjects changed.

    The report compares the original and corrected PVL-Delta parameters and
    negative log-likelihood for every selected subject. It also reports
    whether the corrected PVL-Delta fit reaches the corresponding Q-learning
    negative log-likelihood.

    A corrected fit can be classified as:
        1. `IMPROVED` when its NLL is lower by a magnitude larger than `n_trials * selection_atol_per_trial`.
        2. `UNCHANGED` when the absolute NLL difference is less than or equal to `n_trials * selection_atol_per_trial`.
        3. `WORSE` when its NLL is higher by a magnitude larger than `n_trials * selection_atol_per_trial`.

    Args:
        original_fit_results: Complete fit-results table before correction.
        corrected_fit_results: Complete fit-results table after correction.
        subject_keys: Participant keys selected for PVL-Delta refitting.
        report_path: Path at which to write the text audit report.
        selection_atol_per_trial: The absolute tolerance per trial used for selecting subjects for PVL-Delta refitting.
        logger: Logger instance or logger name.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If schemas, row identities, target rows, or unchanged
            rows fail validation, or if required values are invalid (e.g. `selection_atol_per_trial` is not a non-negative number).
    """

    if not isinstance(original_fit_results, pd.DataFrame):
        raise TypeError(
            "original_fit_results must be a pandas DataFrame, got "
            f"{type(original_fit_results).__name__}."
        )

    if not isinstance(corrected_fit_results, pd.DataFrame):
        raise TypeError(
            "corrected_fit_results must be a pandas DataFrame, got "
            f"{type(corrected_fit_results).__name__}."
        )

    if not isinstance(subject_keys, pd.DataFrame):
        raise TypeError(
            f"subject_keys must be a pandas DataFrame, got {type(subject_keys).__name__}."
        )

    report_path = normalize_path(report_path, parameter_name="report_path")

    parsed_atol_per_trial = _validate_nonnegative_finite_float(
        selection_atol_per_trial, parameter_name="selection_atol_per_trial"
    )

    if isinstance(logger, str):
        report_logger = logging.getLogger(logger)
    elif isinstance(logger, logging.Logger):
        report_logger = logger
    else:
        raise TypeError(
            f"logger must be a logging.Logger or logger name, got {type(logger).__name__}."
        )

    if original_fit_results.columns.has_duplicates:
        duplicate_columns = original_fit_results.columns[
            original_fit_results.columns.duplicated(keep=False)
        ].tolist()

        raise ValueError(
            f"The original fit-results table contains duplicate column names: {duplicate_columns}"
        )

    if corrected_fit_results.columns.has_duplicates:
        duplicate_columns = corrected_fit_results.columns[
            corrected_fit_results.columns.duplicated(keep=False)
        ].tolist()

        raise ValueError(
            f"The corrected fit-results table contains duplicate column names: {duplicate_columns}"
        )

    if original_fit_results.shape != corrected_fit_results.shape:
        raise ValueError(
            "Original and corrected fit-results tables have different shapes: "
            f"original={original_fit_results.shape}, "
            f"corrected={corrected_fit_results.shape}."
        )

    original_columns = list(original_fit_results.columns)
    corrected_columns = list(corrected_fit_results.columns)

    if original_columns != corrected_columns:
        missing_corrected_columns = sorted(set(original_columns) - set(corrected_columns))
        unexpected_corrected_columns = sorted(set(corrected_columns) - set(original_columns))

        raise ValueError(
            "Original and corrected fit-results columns differ. "
            f"Missing corrected columns: {missing_corrected_columns}. "
            f"Unexpected corrected columns: {unexpected_corrected_columns}. "
            "The column order must also be identical."
        )

    participant_key_columns = list(PARTICIPANT_KEY_COLUMNS)  # includes the 'n_trials' column

    row_key_columns = [*participant_key_columns, MODEL_COLUMN]

    pvl_parameter_columns = [*PVLDeltaModel.get_parameter_names()]

    required_columns = {
        *row_key_columns,
        *pvl_parameter_columns,
        NLL_COLUMN,
        CONVERGED_COLUMN,
    }

    missing_required_columns = required_columns - set(original_fit_results.columns)

    if missing_required_columns:
        missing_text = ", ".join(sorted(missing_required_columns))
        raise ValueError(f"The fit-results tables are missing required columns: {missing_text}")

    normalized_subject_keys = normalize_subject_key_columns(subject_keys)

    duplicate_subject_mask = normalized_subject_keys.duplicated(
        subset=participant_key_columns,
        keep=False,
    )

    if duplicate_subject_mask.any():
        duplicate_keys = normalized_subject_keys.loc[
            duplicate_subject_mask,
            participant_key_columns,
        ]

        raise ValueError(
            "The selected subject-key table contains duplicate keys:\n"
            f"{duplicate_keys.to_string(index=False)}"
        )

    original = original_fit_results.copy()
    corrected = corrected_fit_results.copy()

    normalized_original_keys = normalize_subject_key_columns(original)
    normalized_corrected_keys = normalize_subject_key_columns(corrected)

    for column_name in participant_key_columns:
        original[column_name] = normalized_original_keys[column_name]
        corrected[column_name] = normalized_corrected_keys[column_name]

    for table_name, table in (
        ("original", original),
        ("corrected", corrected),
    ):
        normalized_models = normalize_nonempty_string_series(
            table[MODEL_COLUMN],
            column_name=MODEL_COLUMN,
        )

        table[MODEL_COLUMN] = normalized_models

        duplicate_row_mask = table.duplicated(
            subset=row_key_columns,
            keep=False,
        )

        if duplicate_row_mask.any():
            duplicate_rows = table.loc[
                duplicate_row_mask,
                row_key_columns,
            ]

            raise ValueError(
                f"The {table_name} fit-results table contains duplicate "
                "participant-model keys:\n"
                f"{duplicate_rows.to_string(index=False)}"
            )

    try:
        pd.testing.assert_frame_equal(
            original.loc[:, row_key_columns].reset_index(drop=True),
            corrected.loc[:, row_key_columns].reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "Original and corrected fit-results tables do not contain the "
            "same participant-model rows in the same order."
        ) from error

    pvl_delta_model_name = PVLDeltaModel.get_name()
    q_learning_model_name = QLearningModel.get_name()

    def validate_selected_model_rows(
        table: DataFrame,
        *,
        table_name: str,
        model_name: str,
    ) -> None:
        """Require one selected participant row for a specified model.

        Args:
            table: Original or corrected fit-results table.
            table_name: Human-readable table label used in diagnostics.
            model_name: Canonical model whose selected rows are required.

        Raises:
            ValueError: If any selected participant lacks exactly one row for the requested
                model in the supplied table.
        """

        model_subject_keys = table.loc[
            table[MODEL_COLUMN].eq(model_name),
            participant_key_columns,
        ]

        selected_model_rows = normalized_subject_keys.merge(
            model_subject_keys,
            on=participant_key_columns,
            how="left",
            indicator=True,
            validate="one_to_one",
        )

        missing_mask = selected_model_rows["_merge"].ne("both")

        if missing_mask.any():
            missing_keys = selected_model_rows.loc[
                missing_mask,
                participant_key_columns,
            ]

            raise ValueError(
                f"Selected subjects are missing {model_name} rows in the "
                f"{table_name} fit-results table:\n"
                f"{missing_keys.to_string(index=False)}"
            )

    for table_name, table in (
        ("original", original),
        ("corrected", corrected),
    ):
        validate_selected_model_rows(
            table,
            table_name=table_name,
            model_name=pvl_delta_model_name,
        )
        validate_selected_model_rows(
            table,
            table_name=table_name,
            model_name=q_learning_model_name,
        )

    selected_subject_index = pd.MultiIndex.from_frame(
        normalized_subject_keys.loc[:, participant_key_columns]
    )

    original_subject_index = pd.MultiIndex.from_frame(original.loc[:, participant_key_columns])

    targeted_pvl_mask = original[MODEL_COLUMN].eq(
        pvl_delta_model_name
    ).to_numpy() & original_subject_index.isin(selected_subject_index)

    expected_target_count = len(normalized_subject_keys)
    actual_target_count = int(targeted_pvl_mask.sum())

    if actual_target_count != expected_target_count:
        raise ValueError(
            "The number of targeted PVL-Delta rows is incorrect: "
            f"expected={expected_target_count}, actual={actual_target_count}."
        )

    original_unchanged_rows = original_fit_results.loc[~targeted_pvl_mask].reset_index(drop=True)

    corrected_unchanged_rows = corrected_fit_results.loc[~targeted_pvl_mask].reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            original_unchanged_rows,
            corrected_unchanged_rows,
            check_dtype=False,
            check_exact=True,
            check_categorical=False,
        )
    except AssertionError as error:
        raise ValueError("Rows outside the targeted PVL-Delta fits were modified.") from error

    def format_report_value(value: Any) -> str:
        """Format one scalar value for the plain-text correction report.

        Missing values are represented as `"NA"`, floating-point values use a compact
        12-significant-digit format, integer-like values are rendered without a decimal
        point, and all remaining values use their string representation.

        Args:
            value: Scalar diagnostic value to format.

        Returns:
            Human-readable report representation.
        """

        try:
            if pd.isna(value):
                return "NA"
        except (TypeError, ValueError):
            pass

        if isinstance(value, (float, np.floating)):
            return f"{float(value):.12g}"

        if isinstance(value, (int, np.integer)):
            return str(int(value))

        return str(value)

    status_counts: dict[str, list[int]] = {
        "IMPROVED": [],
        "UNCHANGED": [],
        "WORSE": [],
        "NESTING_FAILURE": [],
    }

    subject_report_blocks: list[str] = []

    diagnostic_columns = [
        column_name
        for column_name in (
            CONVERGED_COLUMN,
            "message",
            "nfev",
            "nit",
            "n_starts",
        )
        if column_name in original.columns
    ]

    def get_model_row(
        table: DataFrame,
        *,
        table_name: str,
        model_name: str,
        subject_key_values: tuple[int, ...],
    ) -> Series:
        """Return the unique fit-result row for one model and participant key.

        Args:
            table: Original or corrected fit-results table.
            table_name: Human-readable table label used in diagnostics.
            model_name: Canonical model to select.
            subject_key_values: Values aligned with `PARTICIPANT_KEY_COLUMNS` for one
                participant.

        Returns:
            The single matching model-fit row.

        Raises:
            ValueError: If the model/participant key selects zero or multiple rows.
        """

        row_mask = table[MODEL_COLUMN].eq(model_name)

        for column_name, value in zip(
            participant_key_columns,
            subject_key_values,
            strict=True,
        ):
            row_mask = row_mask & table[column_name].eq(value)

        matching_rows = table.loc[row_mask]

        if len(matching_rows) != 1:
            subject_description = ", ".join(
                f"{column_name}={value}"
                for column_name, value in zip(
                    participant_key_columns,
                    subject_key_values,
                    strict=True,
                )
            )

            raise ValueError(
                f"Expected exactly one {model_name} row for "
                f"{subject_description} in the {table_name} table, "
                f"but found {len(matching_rows)}."
            )

        return matching_rows.iloc[0]

    for subject_key_values in normalized_subject_keys.loc[
        :,
        participant_key_columns,
    ].itertuples(
        index=False,
        name=None,
    ):
        normalized_key_values = tuple(int(value) for value in subject_key_values)

        original_pvl_row = get_model_row(
            original,
            table_name="original",
            model_name=pvl_delta_model_name,
            subject_key_values=normalized_key_values,
        )

        corrected_pvl_row = get_model_row(
            corrected,
            table_name="corrected",
            model_name=pvl_delta_model_name,
            subject_key_values=normalized_key_values,
        )

        q_learning_row = get_model_row(
            original,
            table_name="original",
            model_name=q_learning_model_name,
            subject_key_values=normalized_key_values,
        )

        subject_description = (
            "("
            + ", ".join(
                f"{column_name}={value}"
                for column_name, value in zip(
                    participant_key_columns,
                    normalized_key_values,
                    strict=True,
                )
            )
            + ")"
        )

        # could be taken also from `corrected_pvl_row` or `q_learning_row` (already validated to be identical)
        n_trials = int(original_pvl_row[N_TRIALS_COLUMN])
        effective_atol = n_trials * parsed_atol_per_trial

        original_pvl_nll = _validate_nonnegative_finite_float(
            original_pvl_row[NLL_COLUMN],
            parameter_name=(f"Original PVL-Delta NLL for {subject_description}"),
        )

        corrected_pvl_nll = _validate_nonnegative_finite_float(
            corrected_pvl_row[NLL_COLUMN],
            parameter_name=(f"Corrected PVL-Delta NLL for {subject_description}"),
        )

        q_learning_nll = _validate_nonnegative_finite_float(
            q_learning_row[NLL_COLUMN],
            parameter_name=(f"Q-learning NLL for {subject_description}"),
        )

        pvl_nll_difference = original_pvl_nll - corrected_pvl_nll

        if pvl_nll_difference > effective_atol:
            status = "IMPROVED"
        elif -pvl_nll_difference > effective_atol:
            status = "WORSE"
        else:
            # !(nll_difference > effective_atol) -> nll_difference <= effective_atol
            # !(-nll_difference > effective_atol) -> nll_difference >= -effective_atol
            # nll_difference <= effective_atol && nll_difference >= -effective_atol -> |nll_difference| <= effective_atol
            status = "UNCHANGED"

        status_counts[status].append(n_trials)

        corrected_models_nll_difference = corrected_pvl_nll - q_learning_nll

        nesting_condition_satisfied = not (corrected_models_nll_difference > effective_atol)

        if not nesting_condition_satisfied:
            status_counts["NESTING_FAILURE"].append(n_trials)

        flags: list[str] = []

        if status == "UNCHANGED":
            flags.append("NO_MEANINGFUL_NLL_IMPROVEMENT")
        elif status == "WORSE":
            flags.append("CORRECTED_NLL_IS_WORSE")

        if not nesting_condition_satisfied:
            flags.append("CORRECTED_PVL_NLL_REMAINS_ABOVE_Q_NLL")

        is_there_neg_diff = pvl_nll_difference < 0 or corrected_models_nll_difference < 0
        subject_lines = [
            f"Participant {subject_description}",
            f"  Status                         : {status}",
            (
                f"  PVL nesting condition satisfied: {'Yes' if nesting_condition_satisfied else 'No'}"
            ),
            (f"  Flags                          : {', '.join(flags) if flags else 'N/A'}"),
            "",
            "  Negative log-likelihood comparison",
            f"    • Q-learning                        : {' ' * is_there_neg_diff}{q_learning_nll:.12g}",  # NLL is non-negative
            f"    • Original PVL-Delta                : {' ' * is_there_neg_diff}{original_pvl_nll:.12g}",  # NLL is non-negative
            f"    • Corrected PVL-Delta               : {' ' * is_there_neg_diff}{corrected_pvl_nll:.12g}",  # NLL is non-negative
            f"    • Corrected PVL-Delta - Original PVL: {' ' if is_there_neg_diff and pvl_nll_difference >= 0 else ''}{pvl_nll_difference:.12g}",
            f"    • Corrected PVL-Delta - Q-learning  : {' ' if is_there_neg_diff and corrected_models_nll_difference >= 0 else ''}{corrected_models_nll_difference:.12g}",
            "",
            "  PVL-Delta parameters comparison",
        ]

        for parameter_name in pvl_parameter_columns:
            original_value = format_report_value(original_pvl_row[parameter_name])
            corrected_value = format_report_value(corrected_pvl_row[parameter_name])

            subject_lines.extend(
                [
                    f"    • {parameter_name}",
                    f"        original : {original_value}",
                    f"        corrected: {corrected_value}",
                ]
            )

        subject_lines.extend(
            [
                "",
                "  Optimization diagnostics comparison",
            ]
        )

        for diagnostic_name in diagnostic_columns:
            original_value = format_report_value(original_pvl_row[diagnostic_name])
            corrected_value = format_report_value(corrected_pvl_row[diagnostic_name])

            subject_lines.extend(
                [
                    f"    • {diagnostic_name}",
                    f"        original : {original_value}",
                    f"        corrected: {corrected_value}",
                ]
            )

        subject_report_blocks.append("\n".join(subject_lines))

    improved_count_per_n_trials = Counter(status_counts["IMPROVED"])
    unchanged_count_per_n_trials = Counter(status_counts["UNCHANGED"])
    worse_count_per_n_trials = Counter(status_counts["WORSE"])
    non_improved_count_per_n_trials = unchanged_count_per_n_trials + worse_count_per_n_trials
    total_count_per_n_trials = improved_count_per_n_trials + non_improved_count_per_n_trials

    nesting_failure_count_per_n_trials = Counter(status_counts["NESTING_FAILURE"])

    sorted_n_trials = sorted(total_count_per_n_trials.keys())
    max_n_trials_width = len(str(sorted_n_trials[-1]))

    def format_report_lines_of_count_per_n_trials(
        status_name: str,
        count_per_n_trials: Counter[int],
        *,
        indent_size: int = 2,
        base_indent_level: int = 0,
    ) -> list[str]:
        """Format per-trial-count audit counts for the correction report.

        Args:
            status_name: Label describing the counted correction status.
            count_per_n_trials: Counts keyed by subject trial count.
            indent_size: Number of spaces in one indentation level.
            base_indent_level: Initial indentation level for generated lines.

        Returns:
            Report lines describing the total and the breakdown by trial count.
        """
        lines: list[str] = [f"{' ' * (base_indent_level * indent_size)}{status_name} subjects"]

        max_n_trials_width = max((len(str(n_trials)) for n_trials in count_per_n_trials), default=0)

        total_count = count_per_n_trials.total()
        max_count_width = len(str(total_count))

        for n_trials, count in sorted(count_per_n_trials.items()):
            lines.append(
                f"{' ' * ((base_indent_level + 1) * indent_size)}• {n_trials:>{max_n_trials_width}} trials: {count:>{max_count_width}} subjects"
            )

        # len("trials") = len("Total") + 1
        right_pad_for_total = max_n_trials_width + 2
        lines.append(
            f"{' ' * ((base_indent_level + 1) * indent_size)}• Total{' ' * right_pad_for_total}: {total_count:>{max_count_width}} subjects"
        )

        return lines

    report_lines = [
        "PVL-Delta Correction Report",
        "===========================",
        "",
        *format_report_lines_of_count_per_n_trials(
            "Target", total_count_per_n_trials, base_indent_level=0
        ),
        "",
        *format_report_lines_of_count_per_n_trials(
            "Improved", improved_count_per_n_trials, base_indent_level=0
        ),
        "",
        *format_report_lines_of_count_per_n_trials(
            "Unchanged", unchanged_count_per_n_trials, base_indent_level=0
        ),
        "",
        *format_report_lines_of_count_per_n_trials(
            "Worsen", worse_count_per_n_trials, base_indent_level=0
        ),
        "",
        *format_report_lines_of_count_per_n_trials(
            "Not improved", non_improved_count_per_n_trials, base_indent_level=0
        ),
        "",
        *format_report_lines_of_count_per_n_trials(
            "Nesting failure", nesting_failure_count_per_n_trials, base_indent_level=0
        ),
        "",
        "NLL absolute comparison tolerance",
        f"  • per trial{' ' * (len('subject with ') + max_n_trials_width + 2)}: {parsed_atol_per_trial:.12g}",
        *[
            f"  • for subject with {n_trials:>{max_n_trials_width}} trials: {n_trials * parsed_atol_per_trial:.12g}"
            for n_trials in sorted_n_trials
        ],
        "",
        "Integrity checks",
        "  • Table shapes match                              : Yes",
        "  • Column names and order match                    : Yes",
        "  • Participant-model row identities and order match: Yes",
        "  • All non-targeted rows are unchanged             : Yes",
        "",
        "Subject-level results",
        "---------------------",
        "",
        "\n\n".join(subject_report_blocks),
    ]

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        "\n".join(report_lines).rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )

    report_logger.info(
        "Saved PVL-Delta correction report written to: %s",
        report_path,
    )

    non_improved_count = non_improved_count_per_n_trials.total()
    nesting_failure_count = nesting_failure_count_per_n_trials.total()

    if non_improved_count > 0:
        report_logger.warning(
            "%d of %d refitted PVL-Delta subjects did not obtain a meaningfully lower NLL.",
            non_improved_count,
            len(normalized_subject_keys),
        )

    if nesting_failure_count > 0:
        report_logger.warning(
            "%d of %d corrected PVL-Delta fits still have a higher NLL "
            "than the corresponding Q-learning fit.",
            nesting_failure_count,
            len(normalized_subject_keys),
        )

    if non_improved_count == 0 and nesting_failure_count == 0:
        report_logger.info(
            "All %d refitted PVL-Delta subjects obtained a meaningfully "
            "lower NLL and no corrected PVL-Delta fit had a higher NLL "
            "than the corresponding Q-learning fit.",
            len(normalized_subject_keys),
        )


def _run(
    *,
    normalized_args: argparse.Namespace,
    start_datetime_str: str,
    original_fit_results: DataFrame,
    subject_keys: DataFrame,
    pvl_delta_model: PVLDeltaModel,
    warm_starts_provider: SubjectModelWarmStartsProvider,
    logger: logging.Logger | str = LOGGER_NAME,
) -> list[Path]:
    """Run the targeted refit and save three corrected result tables.

    Args:
        normalized_args: Normalized command-line arguments namespace.
        start_datetime_str: Timestamp used in output filenames.
        original_fit_results: Original complete model-fit table.
        subject_keys: Participant keys selected for refitting.
        pvl_delta_model: PVL-Delta model to refit.
        warm_starts_provider: Provider of Q-learning-equivalent warm
            starting points.
        logger: Logger instance or logger name.

    Returns:
        Paths of the corrected model-fit, model-comparison, model-summary CSV files,
        and the comparison report of the original and corrected fit-results tables.
    """

    logger = logging.getLogger(logger) if isinstance(logger, str) else logger

    logger.info(
        "Running PVL-Delta refits for %d subjects with %d regular "
        "starting points plus one Q-learning-equivalent warm start.",
        len(subject_keys),
        DEFAULT_N_PVL_STARTS,
    )

    targeted_pvl_results = run_fitting_pipeline(
        FittingPipelineConfig(
            rdata_path=normalized_args.rdata_path,
            models=(pvl_delta_model,),
            max_iterations=DEFAULT_MAX_ITERATIONS,
            n_workers=normalized_args.n_workers,
            show_progress=not normalized_args.no_progress,
            n_subjects=None,
            subject_keys=subject_keys,
            subject_model_warm_starts_provider=warm_starts_provider,
        )
    )

    _validate_targeted_pvl_results(
        targeted_pvl_results,
        subject_keys,
        pvl_delta_model_name=pvl_delta_model.name,
    )

    logger.info(
        "Completed %d targeted PVL-Delta refits.",
        len(targeted_pvl_results),
    )

    corrected_fit_results = _replace_model_fit_rows(
        original_fit_results,
        targeted_pvl_results,
    )

    comparison_table = add_model_comparison_columns(corrected_fit_results)
    summary_table = summarize_model_comparison(corrected_fit_results)

    actual_n_subjects = int(
        corrected_fit_results.loc[:, list(PARTICIPANT_KEY_COLUMNS)].drop_duplicates().shape[0]
    )

    filename_suffix = f"{actual_n_subjects}_subjects_{start_datetime_str}"

    fits_path = normalized_args.output_dir / (f"model_fits_corrected_{filename_suffix}.csv")
    comparison_path = normalized_args.output_dir / (
        f"model_comparison_corrected_{filename_suffix}.csv"
    )
    summary_path = normalized_args.output_dir / (f"model_summary_corrected_{filename_suffix}.csv")
    report_path = normalized_args.output_dir / (f"model_correction_report_{filename_suffix}.txt")

    corrected_fit_results.to_csv(
        fits_path,
        index=False,
        lineterminator="\n",
    )
    comparison_table.to_csv(
        comparison_path,
        index=False,
        lineterminator="\n",
    )
    summary_table.to_csv(
        summary_path,
        index=False,
        lineterminator="\n",
    )

    logger.info(
        "Saved corrected complete model fits to: %s",
        fits_path,
    )
    logger.info(
        "Saved regenerated model comparison to: %s",
        comparison_path,
    )
    logger.info(
        "Saved regenerated model summary to: %s",
        summary_path,
    )

    _compare_original_and_corrected_fit_results(
        original_fit_results=original_fit_results,
        corrected_fit_results=corrected_fit_results,
        subject_keys=subject_keys,
        report_path=report_path,
        selection_atol_per_trial=normalized_args.atol_per_trial,
        logger=logger,
    )

    return [
        fits_path,
        comparison_path,
        summary_path,
        report_path,
    ]


def _cleanup(
    *,
    logger: logging.Logger | str = LOGGER_NAME,
    output_dir: Path | None = None,
) -> None:
    """Clean up application logging after the correction workflow.

    Args:
        logger: Logger instance or logger name.
        output_dir: Optional run-output directory to remove when it is empty.
    """

    logger = logging.getLogger(logger) if isinstance(logger, str) else logger

    if output_dir is not None and output_dir.is_dir():
        logger.info("Cleaning up output directory: %s", output_dir)

        if not any(output_dir.iterdir()):
            logger.info("Output directory is empty, attempting to remove it: %s", output_dir)
            try:
                output_dir.rmdir()
                logger.info("Removed empty output directory: %s", output_dir)
            except Exception as e:
                logger.warning(
                    "Failed to remove output directory %s: %s",
                    output_dir,
                    str(e),
                )
        else:
            logger.info("Output directory is not empty, skipping cleanup: %s", output_dir)

    logger.info("Cleaning up application logging...")
    application_logging_cleanup()


def main() -> None:
    """Run the complete targeted PVL-Delta correction workflow.

    The entry point resolves command-line inputs, selects apparent nesting violations,
    builds Q-equivalent PVL warm starts, refits the targeted participants, regenerates
    corrected fit/comparison/summary outputs, writes the correction audit report, sends
    optional notifications, and cleans up logging.

    Raises:
        Exception: Propagates setup, fitting, validation, output, or notification errors
            after the failure-notification context has had an opportunity to report them.
    """

    start_counter = time.perf_counter()

    (
        args,
        normalized_args,
        start_datetime_str,
        original_fit_results,
        subject_keys,
        pvl_delta_model,
        warm_starts_provider,
        logging_path,
    ) = _setup()

    notify_formsubmit_id: str | None = normalized_args.notify_formsubmit_id
    logger = logging.getLogger(LOGGER_NAME)

    try:
        with error_email_notifier(
            formsubmit_id=notify_formsubmit_id,
            script_name=Path(__file__).name,
            start_counter=start_counter,
        ):
            logger.info(msg="Starting targeted PVL-Delta fit correction...")

            logger.debug("Parsed command-line arguments: %s", args)
            logger.debug("Normalized command-line arguments: %s", normalized_args)

            logger.info(
                "Selected %d converged subjects for whom Q-learning beats "
                "the best competing model by more than atol=(N * %g), where N denotes the number of trials for each subject respectively.",
                len(subject_keys),
                normalized_args.atol_per_trial,
            )

            if subject_keys.empty:
                raise ValueError("No subjects satisfied the Q-learning NLL-selection criterion.")

            normalized_args.output_dir.mkdir(parents=True, exist_ok=True)

            result_files = _run(
                normalized_args=normalized_args,
                start_datetime_str=start_datetime_str,
                original_fit_results=original_fit_results,
                subject_keys=subject_keys,
                pvl_delta_model=pvl_delta_model,
                warm_starts_provider=warm_starts_provider,
                logger=logger,
            )

            if logging_path is not None:
                logger.info(
                    "Saved log file to: %s",
                    logging_path,
                )

                result_files.append(Path(logging_path))

            end_counter = time.perf_counter()
            elapsed_time = end_counter - start_counter
            elapsed_time_obj = timedelta(seconds=round(elapsed_time))

            logger.info(
                "Completed targeted PVL-Delta fit correction in %s.",
                elapsed_time_obj,
            )

            if notify_formsubmit_id is not None:
                logger.info(
                    "Sending FormSubmit notification to %r with result files %r as one zip attachment.",
                    notify_formsubmit_id,
                    [path.name for path in result_files],
                )

            if notify_formsubmit_id is not None:
                zip_filename = "corrected_pvl_delta_fit_output_files.zip"

                result_file_lines = "\n".join(f"  - {path.name}" for path in result_files)

                email_message = f"""
The targeted PVL-Delta fit correction has completed successfully.

The selected PVL-Delta rows were replaced in a copy of the complete model-fit table. The model-comparison and model-summary tables were then regenerated from that corrected table.

The following files are attached in a zip file named {zip_filename!r}:
{result_file_lines}
                """.strip()

                send_formsubmit_email_script_success_notification(
                    formsubmit_id=notify_formsubmit_id,
                    message=email_message,
                    duration_seconds=elapsed_time,
                    script_name=Path(__file__).name,
                    file_paths=result_files,
                    zip_filename=zip_filename,
                )
    finally:
        logger.info("Performing cleanup...")
        _cleanup(
            logger=logger,
            output_dir=normalized_args.output_dir,
        )


if __name__ == "__main__":
    main()
