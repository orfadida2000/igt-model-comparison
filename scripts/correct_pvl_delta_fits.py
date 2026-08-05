"""Correct selected PVL-Delta fits using Q-learning-equivalent warm starts."""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

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
)
from igt.constants.fitting import DEFAULT_MAX_ITERATIONS
from igt.constants.models import MAX_LEARNING_RATE, MIN_INVERSE_TEMPERATURE, MIN_LEARNING_RATE
from igt.constants.path import IGT_DATASET_PATH, LOGS_DIR, RESULTS_DIR
from igt.constants.schema import PARTICIPANT_KEY_COLUMNS
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
    normalize_subject_key_columns,
    read_fit_results_csv,
    select_model_lowest_nll_subject_keys,
)
from igt.typing import Float2DArray

NLL_SELECTION_EPSILON: Final[float] = 1e-8

PVL_OUTCOME_SENSITIVITY: Final[float] = 1.0
PVL_LOSS_AVERSION: Final[float] = 1.0

LOGGER_NAME: Final[str] = "scripts.correct_pvl_delta_fits"


def _existing_file_path(value: str) -> Path:
    """Parse an existing file path from a command-line argument.

    Args:
        value: Raw command-line path value.

    Returns:
        The parsed path.

    Raises:
        argparse.ArgumentTypeError: If the path does not identify an
            existing file.
    """

    path = Path(value)

    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")

    return path


def _directory_path(value: str) -> Path:
    """Parse a directory path from a command-line argument.

    The directory is not created by this function.

    Args:
        value: Raw command-line path value.

    Returns:
        The parsed path.

    Raises:
        argparse.ArgumentTypeError: If the path exists but is not a
            directory.
    """

    path = Path(value)

    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"Path exists but is not a directory: {path}")

    return path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Select subjects for whom Q-learning has the lowest NLL, "
            "refit their PVL-Delta model using an additional "
            "Q-learning-equivalent warm start, replace those rows in the "
            "complete fit-results table, and regenerate the model "
            "comparison and summary tables."
        )
    )

    parser.add_argument(
        "fit_results_path",
        type=_existing_file_path,
        help=(
            "Original complete model-fits CSV used to select subjects, "
            "construct warm starts, and create the corrected output."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=NLL_SELECTION_EPSILON,
        help=(
            "Minimum NLL advantage required for Q-learning over the best "
            f"competing model (default: {NLL_SELECTION_EPSILON:g})."
        ),
    )
    parser.add_argument(
        "--rdata-path",
        type=_existing_file_path,
        default=IGT_DATASET_PATH,
        help=(f"Input IGT RData file (default: {IGT_DATASET_PATH})."),
    )
    parser.add_argument(
        "--output-dir",
        type=_directory_path,
        default=RESULTS_DIR / "corrected_pvl_delta_fits",
        help=(
            "Corrected-results output directory "
            f"(default: {RESULTS_DIR / 'corrected_pvl_delta_fits'})."
        ),
    )
    parser.add_argument(
        "--logging-dir",
        type=_directory_path,
        default=LOGS_DIR / "corrected_pvl_delta_fits",
        help=(f"Log directory (default: {LOGS_DIR / 'corrected_pvl_delta_fits'})."),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_N_WORKERS,
        help=(
            "Worker processes; use 0 for serial execution and a negative "
            "value for all available CPU cores."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=DEFAULT_ROOT_LOG_LEVEL,
        help=("Root logging level; use a negative value to disable logging."),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the subject progress bar.",
    )

    return parser.parse_args()


def _normalize_args(args: argparse.Namespace) -> dict[str, Any]:
    normalized_args: dict[str, Any] = {
        "fit_results_path": args.fit_results_path,
        "epsilon": args.epsilon if args.epsilon >= 0 else NLL_SELECTION_EPSILON,
        "rdata_path": args.rdata_path,
        "n_workers": (None if args.workers < 0 else (1 if args.workers == 0 else args.workers)),
        "output_dir": args.output_dir or args.rdata_path.parent / "output_files",
        "logging_dir": args.logging_dir or args.rdata_path.parent / "logs",
        "logging_disabled": bool(args.log_level < 0),
        "log_level": None if args.log_level < 0 else args.log_level,
        "no_progress": args.no_progress,
        "notify_formsubmit_id": DEFAULT_NOTIFY_FORMSUBMIT_ID,
    }

    return normalized_args


def _build_q_equivalent_pvl_warm_starts_provider(
    fit_results: DataFrame,
    subject_keys: DataFrame,
    *,
    pvl_delta_model: PVLDeltaModel,
) -> SubjectModelWarmStartsProvider:
    """Build a provider of Q-learning-equivalent PVL-Delta warm starts.

    For a Q-learning learning rate ``a`` and inverse temperature ``beta``,
    the corresponding PVL-Delta starting point is:

    ``(a, 1, 1, log_3(beta + 1))``.

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
        "model",
        "learning_rate",
        "inverse_temperature",
    }

    missing_columns = required_columns - set(fit_results.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fit-results table is missing columns: {missing_text}")

    normalized_models = fit_results["model"].astype("string").str.strip()

    q_results = fit_results.loc[
        normalized_models.eq(QLearningModel.get_name()),
        [
            *key_columns,
            "learning_rate",
            "inverse_temperature",
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
            selected_q_results["learning_rate"],
            errors="raise",
        ).to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        q_inverse_temperatures = pd.to_numeric(
            selected_q_results["inverse_temperature"],
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

    selected_q_results["pvl_response_consistency"] = pvl_response_consistencies

    warm_starts_by_subject: dict[
        tuple[int, int],
        Float2DArray,
    ] = {}

    warm_start_columns = [
        *key_columns,
        "learning_rate",
        "pvl_response_consistency",
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
            One PVL-Delta warm starting point, or ``None`` when a
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
        "model",
    }

    missing_columns = required_columns - set(targeted_results.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Targeted fit-results table is missing columns: {missing_text}")

    normalized_models = targeted_results["model"].astype("string").str.strip()

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
        "model",
    ]

    required_fit_columns = {
        *key_columns,
        "negative_log_likelihood",
        "log_likelihood",
        "aic",
        "bic",
        "converged",
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
        normalized_models = table["model"].astype("string").str.strip()

        if normalized_models.isna().any():
            raise ValueError(f"The {table_name} fit-results table contains missing model names.")

        if normalized_models.eq("").any():
            raise ValueError(f"The {table_name} fit-results table contains empty model names.")

        table["model"] = normalized_models

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
    dict[str, Any],
    str,
    DataFrame,
    DataFrame,
    PVLDeltaModel,
    SubjectModelWarmStartsProvider,
    Path | None,
]:
    """Resolve inputs, select subjects, and configure logging.

    Returns:
        The parsed arguments, output timestamp, original fit-results
        table, selected participant keys, PVL-Delta model, warm-start
        provider, and optional log-file path.
    """

    start_datetime_str = datetime.now().strftime(FILENAME_DATETIME_FMT)
    args = _parse_args()
    normalized_args = _normalize_args(args)

    normalized_args["output_dir"] = Path(normalized_args["output_dir"]) / start_datetime_str

    original_fit_results = read_fit_results_csv(normalized_args["fit_results_path"])

    subject_keys = select_model_lowest_nll_subject_keys(
        original_fit_results,
        model=QLearningModel,
        epsilon=normalized_args["epsilon"],
        require_convergence=True,
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
        disabled=normalized_args["logging_disabled"],
        root_level=normalized_args["log_level"],
        log_file_path=(
            normalized_args["logging_dir"]
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
    report_path: Path,
    nll_tolerance: float = 1e-8,
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

    A corrected fit is classified as:

    - ``IMPROVED`` when its NLL is lower by more than ``nll_tolerance``.
    - ``UNCHANGED`` when the absolute NLL difference is within the tolerance.
    - ``WORSE`` when its NLL is higher by more than the tolerance.

    Args:
        original_fit_results: Complete fit-results table before correction.
        corrected_fit_results: Complete fit-results table after correction.
        subject_keys: Participant keys selected for PVL-Delta refitting.
        report_path: Path at which to write the text audit report.
        nll_tolerance: Nonnegative tolerance used when comparing NLL values.
        logger: Logger instance or logger name.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If schemas, row identities, target rows, or unchanged
            rows fail validation, or if required values are invalid.
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

    if not isinstance(report_path, Path):
        raise TypeError(f"report_path must be a pathlib.Path, got {type(report_path).__name__}.")

    if isinstance(nll_tolerance, bool):
        raise TypeError("nll_tolerance must be a real number, not a Boolean value.")

    try:
        parsed_nll_tolerance = float(nll_tolerance)
    except (TypeError, ValueError) as error:
        raise TypeError("nll_tolerance must be a real number.") from error

    if not np.isfinite(parsed_nll_tolerance) or parsed_nll_tolerance < 0.0:
        raise ValueError("nll_tolerance must be finite and nonnegative.")

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

    participant_key_columns = list(PARTICIPANT_KEY_COLUMNS)

    row_key_columns = [
        *participant_key_columns,
        "model",
    ]

    pvl_parameter_columns = [
        "learning_rate",
        "outcome_sensitivity",
        "loss_aversion",
        "response_consistency",
    ]

    required_columns = {
        *row_key_columns,
        *pvl_parameter_columns,
        "negative_log_likelihood",
        "converged",
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
        normalized_models = table["model"].astype("string").str.strip()

        if normalized_models.isna().any():
            raise ValueError(f"The {table_name} fit-results table contains missing model names.")

        if normalized_models.eq("").any():
            raise ValueError(f"The {table_name} fit-results table contains empty model names.")

        table["model"] = normalized_models

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
        """Validate that every selected subject has one row for a model."""

        model_subject_keys = table.loc[
            table["model"].eq(model_name),
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

    targeted_pvl_mask = original["model"].eq(
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

    def finite_float(
        value: Any,
        *,
        value_name: str,
    ) -> float:
        """Convert one report value to a finite float."""

        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{value_name} must be numeric, not Boolean.")

        try:
            parsed_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{value_name} cannot be interpreted as a number.") from error

        if not np.isfinite(parsed_value):
            raise ValueError(f"{value_name} must be finite.")

        return parsed_value

    def format_report_value(value: Any) -> str:
        """Format a scalar value for the text report."""

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

    status_counts = {
        "IMPROVED": 0,
        "UNCHANGED": 0,
        "WORSE": 0,
    }

    nesting_failure_count = 0
    subject_report_blocks: list[str] = []

    diagnostic_columns = [
        column_name
        for column_name in (
            "converged",
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
        """Return exactly one model-fit row for a participant."""

        row_mask = table["model"].eq(model_name)

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

        subject_description = ", ".join(
            f"{column_name}={value}"
            for column_name, value in zip(
                participant_key_columns,
                normalized_key_values,
                strict=True,
            )
        )

        original_pvl_nll = finite_float(
            original_pvl_row["negative_log_likelihood"],
            value_name=(f"Original PVL-Delta NLL for {subject_description}"),
        )

        corrected_pvl_nll = finite_float(
            corrected_pvl_row["negative_log_likelihood"],
            value_name=(f"Corrected PVL-Delta NLL for {subject_description}"),
        )

        q_learning_nll = finite_float(
            q_learning_row["negative_log_likelihood"],
            value_name=(f"Q-learning NLL for {subject_description}"),
        )

        nll_improvement = original_pvl_nll - corrected_pvl_nll

        if nll_improvement > parsed_nll_tolerance:
            status = "IMPROVED"
        elif nll_improvement < -parsed_nll_tolerance:
            status = "WORSE"
        else:
            status = "UNCHANGED"

        status_counts[status] += 1

        corrected_minus_q_nll = corrected_pvl_nll - q_learning_nll

        nesting_condition_satisfied = corrected_pvl_nll <= q_learning_nll + parsed_nll_tolerance

        if not nesting_condition_satisfied:
            nesting_failure_count += 1

        flags: list[str] = []

        if status == "UNCHANGED":
            flags.append("NO_MEANINGFUL_NLL_IMPROVEMENT")
        elif status == "WORSE":
            flags.append("CORRECTED_NLL_IS_WORSE")

        if not nesting_condition_satisfied:
            flags.append("CORRECTED_PVL_NLL_REMAINS_ABOVE_Q_NLL")

        subject_lines = [
            f"Participant: {subject_description}",
            f"Status: {status}",
            (f"PVL nesting condition satisfied: {'Yes' if nesting_condition_satisfied else 'No'}"),
            (f"Flags: {', '.join(flags) if flags else 'None'}"),
            "",
            "Negative log-likelihood:",
            f"  Q-learning:          {q_learning_nll:.12g}",
            f"  Original PVL-Delta:  {original_pvl_nll:.12g}",
            f"  Corrected PVL-Delta: {corrected_pvl_nll:.12g}",
            f"  Improvement:         {nll_improvement:.12g}",
            f"  Corrected PVL - Q:   {corrected_minus_q_nll:.12g}",
            "",
            "PVL-Delta parameters:",
        ]

        for parameter_name in pvl_parameter_columns:
            original_value = format_report_value(original_pvl_row[parameter_name])
            corrected_value = format_report_value(corrected_pvl_row[parameter_name])

            subject_lines.extend(
                [
                    f"  {parameter_name}:",
                    f"    original:  {original_value}",
                    f"    corrected: {corrected_value}",
                ]
            )

        subject_lines.extend(
            [
                "",
                "Optimization diagnostics:",
            ]
        )

        for diagnostic_name in diagnostic_columns:
            original_value = format_report_value(original_pvl_row[diagnostic_name])
            corrected_value = format_report_value(corrected_pvl_row[diagnostic_name])

            subject_lines.extend(
                [
                    f"  {diagnostic_name}:",
                    f"    original:  {original_value}",
                    f"    corrected: {corrected_value}",
                ]
            )

        subject_report_blocks.append("\n".join(subject_lines))

    non_improved_count = status_counts["UNCHANGED"] + status_counts["WORSE"]

    report_lines = [
        "PVL-Delta Correction Report",
        "===========================",
        "",
        f"Targeted subjects: {len(normalized_subject_keys)}",
        f"Improved: {status_counts['IMPROVED']}",
        f"Unchanged: {status_counts['UNCHANGED']}",
        f"Worse: {status_counts['WORSE']}",
        f"Not improved: {non_improved_count}",
        (f"Corrected PVL NLL still above Q-learning NLL: {nesting_failure_count}"),
        f"NLL comparison tolerance: {parsed_nll_tolerance:.12g}",
        "",
        "Integrity checks:",
        "  Table shapes match: Yes",
        "  Column names and order match: Yes",
        "  Participant-model row identities and order match: Yes",
        "  All non-targeted rows are unchanged: Yes",
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
    normalized_args: dict[str, Any],
    start_datetime_str: str,
    original_fit_results: DataFrame,
    subject_keys: DataFrame,
    pvl_delta_model: PVLDeltaModel,
    warm_starts_provider: SubjectModelWarmStartsProvider,
    logger: logging.Logger | str = LOGGER_NAME,
) -> list[Path]:
    """Run the targeted refit and save three corrected result tables.

    Args:
        normalized_args: Normalized command-line arguments.
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
            rdata_path=normalized_args["rdata_path"],
            models=(pvl_delta_model,),
            max_iterations=DEFAULT_MAX_ITERATIONS,
            n_workers=normalized_args["n_workers"],
            show_progress=not normalized_args["no_progress"],
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

    fits_path = normalized_args["output_dir"] / (f"model_fits_corrected_{filename_suffix}.csv")
    comparison_path = normalized_args["output_dir"] / (
        f"model_comparison_corrected_{filename_suffix}.csv"
    )
    summary_path = normalized_args["output_dir"] / (
        f"model_summary_corrected_{filename_suffix}.csv"
    )
    report_path = normalized_args["output_dir"] / (f"model_correction_report_{filename_suffix}.txt")

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
        nll_tolerance=normalized_args["epsilon"],
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
) -> None:
    """Perform cleanup after the script has finished.

    Args:
        logger: Logger instance or logger name.
    """

    logger = logging.getLogger(logger) if isinstance(logger, str) else logger

    logger.info("Cleaning up application logging...")
    application_logging_cleanup()


def main() -> None:
    """Correct selected PVL-Delta fits and regenerate all result tables."""

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

    notify_formsubmit_id: str | None = normalized_args["notify_formsubmit_id"]

    with error_email_notifier(
        formsubmit_id=notify_formsubmit_id,
        script_name=Path(__file__).name,
        start_counter=start_counter,
    ):
        logger = logging.getLogger(LOGGER_NAME)
        logger.info("Starting targeted PVL-Delta fit correction...")

        logger.debug("Parsed command-line arguments: %r", vars(args))
        logger.debug("Normalized command-line arguments: %r", normalized_args)

        logger.info(
            "Selected %d converged subjects for whom Q-learning beats "
            "the best competing model by more than epsilon=%g.",
            len(subject_keys),
            normalized_args["epsilon"],
        )

        if subject_keys.empty:
            raise ValueError("No subjects satisfied the Q-learning NLL-selection criterion.")

        normalized_args["output_dir"].mkdir(parents=True, exist_ok=True)

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

        logger.info("Performing cleanup...")
        _cleanup(logger=logger)

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


if __name__ == "__main__":
    main()
