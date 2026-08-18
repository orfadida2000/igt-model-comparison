"""Subject-level orchestration for serial and multiprocessing model fitting.

The module converts long-format trial rows into validated `SubjectData`, constructs
serializable subject tasks, dispatches every requested model fit, and preserves task
ordering across worker processes. Statistical comparison of completed fits is handled
separately by `igt.comparison`.
"""

import logging
import multiprocessing as mp
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from logging.handlers import QueueHandler, QueueListener
from multiprocessing.context import BaseContext
from queue import Queue

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from igt.constants.schema import SOURCE_STUDY_COLUMN
from igt.models.base import ComputationalModel
from igt.models.typing import SubjectData
from igt.rdata_preprocessing import iter_subject_trials

from .fitting import fit_model
from .typing import (
    LoggingContext,
    ModelFitResult,
    ModelWarmStarts,
    SubjectFitTask,
    SubjectModelWarmStartsProvider,
)

_REQUIRED_TRIAL_COLUMNS = {
    SOURCE_STUDY_COLUMN,
    "choice",
    "win",
    "loss",
}


_WORKER_MODELS: tuple[ComputationalModel, ...] | None = None
_WORKER_OPTIMIZER_OPTIONS: dict[str, object] | None = None
_WORKER_LOGGER: logging.Logger | None = None


def _compute_subject_data_from_trials(subject_trials: pd.DataFrame) -> SubjectData:
    """Convert one participant's trial rows to validated model input arrays.

    The function requires source-study, choice, win, and loss columns. Choice and
    outcome columns are copied into NumPy arrays and passed to
    [`SubjectData`][igt.models.typing.SubjectData], which performs the remaining
    shape and value validation.

    Args:
        subject_trials: Long-format trial rows for one participant, in trial order.

    Returns:
        Validated choices, wins, and losses for model likelihood evaluation.

    Raises:
        ValueError: If a required trial column is missing or `SubjectData`
            validation rejects the extracted values.
    """

    missing_columns = _REQUIRED_TRIAL_COLUMNS - set(subject_trials.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required subject columns: {missing_text}")

    return SubjectData(
        choices=subject_trials["choice"].to_numpy(
            dtype=np.int64,
            copy=True,
        ),
        wins=subject_trials["win"].to_numpy(
            dtype=np.float64,
            copy=True,
        ),
        losses=subject_trials["loss"].to_numpy(
            dtype=np.float64,
            copy=True,
        ),
    )


def _get_subject_study(subject_trials: pd.DataFrame) -> str:
    """Extract and validate the unique source-study label for one participant.

    Args:
        subject_trials: Long-format trial rows for one participant.

    Returns:
        The participant's stripped, nonempty source-study name.

    Raises:
        ValueError: If the rows contain zero or multiple nonmissing source-study
            values, or if the unique normalized value is empty.
    """

    studies = subject_trials[SOURCE_STUDY_COLUMN].astype("string").str.strip().dropna().unique()

    if len(studies) != 1:
        raise ValueError(
            f"Every subject must belong to exactly one source study; found {len(studies)} values."
        )

    study = str(studies[0])

    if not study:
        raise ValueError("source_study must not be empty.")

    return study


def _validate_models(
    models: Sequence[ComputationalModel],
) -> tuple[ComputationalModel, ...]:
    """Validate the model collection used for participant fitting.

    Args:
        models: Models that will be fitted to every selected participant.

    Returns:
        The supplied models materialized as an immutable tuple.

    Raises:
        ValueError: If no model is supplied or two supplied models share the same
            canonical model name.
    """

    model_tuple = tuple(models)

    if not model_tuple:
        raise ValueError("At least one model is required.")

    model_names = [model.name for model in model_tuple]

    if len(set(model_names)) != len(model_names):
        raise ValueError("Model names must be unique.")

    return model_tuple


def _build_subject_tasks(
    data: pd.DataFrame,
    models: Sequence[ComputationalModel],
    n_subjects: int | None = None,
    subject_model_warm_starts_provider: SubjectModelWarmStartsProvider | None = None,
) -> list[SubjectFitTask]:
    """Build serializable fitting tasks from the long-format IGT table.

    Participant groups are obtained from
    [`iter_subject_trials`][igt.rdata_preprocessing.iter_subject_trials]. For each
    participant, the function validates the source study and trial arrays and, when
    a provider is supplied, attaches any model-specific warm-start matrices.

    Args:
        data: Long-format IGT trial table.
        models: Models for which optional warm starts should be requested.
        n_subjects: Optional maximum number of participant groups to include.
        subject_model_warm_starts_provider: Optional callback returning additional
            starting points for a model and participant key.

    Returns:
        One `SubjectFitTask` per selected participant, in dataset iteration order.

    Raises:
        ValueError: If participant rows are invalid or no participant task can be
            built when at least one participant was requested.
    """

    tasks: list[SubjectFitTask] = []

    for (n_trials, subject_id), subject_trials in iter_subject_trials(data, n_subjects=n_subjects):
        subject_model_warm_starts: list[ModelWarmStarts] = []

        if subject_model_warm_starts_provider is not None:
            for model in models:
                starting_points = subject_model_warm_starts_provider(
                    model.name,
                    n_trials,
                    subject_id,
                )

                if starting_points is not None:
                    subject_model_warm_starts.append(
                        ModelWarmStarts(
                            model_name=model.name,
                            starting_points=starting_points,
                        )
                    )

        subject_task = SubjectFitTask(
            n_trials=n_trials,
            subject_id=subject_id,
            source_study=_get_subject_study(subject_trials),
            data=_compute_subject_data_from_trials(subject_trials),
            model_warm_starts=tuple(subject_model_warm_starts),
        )

        tasks.append(subject_task)

    if not tasks and (n_subjects is None or n_subjects > 0):
        raise ValueError("The dataset does not contain any subjects.")

    return tasks


def _fit_models_for_subject(
    models: Sequence[ComputationalModel],
    *,
    task: SubjectFitTask,
    optimizer_options: Mapping[str, object] | None = None,
    logger: logging.Logger | None = None,
) -> list[ModelFitResult]:
    """Fit every requested model to one participant task.

    Warm starts attached to the task are matched by canonical model name and passed
    to [`fit_model`][igt.execution.fitting.fit_model]. Results are returned in the
    same order as the validated model sequence.

    Args:
        models: Models to fit to the participant.
        task: Serializable participant task containing data and optional warm starts.
        optimizer_options: Optional SciPy optimizer options forwarded to every fit.
        logger: Optional logger used for subject- and model-level diagnostics.

    Returns:
        One completed `ModelFitResult` for each model.
    """

    model_tuple = _validate_models(models)

    warm_starts_by_model = {
        item.model_name: item.starting_points for item in task.model_warm_starts
    }

    if logger is not None:
        logger.debug("Fitting the models for the subject's %r", task)

    model_fit_results: list[ModelFitResult] = []

    for model in model_tuple:
        warm_starting_points = warm_starts_by_model.get(model.name)

        if logger is not None:
            logger.debug(
                "Fitting model %r to subject's task %r with warm-starting points: %r",
                model.name,
                task,
                warm_starting_points,
            )

        model_fit_results.append(
            fit_model(
                model,
                task.data,
                n_trials=task.n_trials,
                subject_id=task.subject_id,
                source_study=task.source_study,
                optimizer_options=optimizer_options,
                warm_starting_points=warm_starting_points,
                logger=logger,
            )
        )

    if logger is not None:
        logger.debug(
            "Completed fitting all models for the subject's task %r",
            task,
        )

    return model_fit_results


def _worker_logger_initialization(
    logging_context: LoggingContext,
    proxy_queue: Queue[logging.LogRecord] | None,
    manager_logger_name: str | None,
) -> None:
    """Initialize logging inside a spawned worker process.

    The captured main-process logging state is applied first. When a proxy
    queue is supplied, a `QueueHandler` forwards worker records to the parent
    process. The function then creates the process-specific worker logger used
    by fitting helpers.

    Args:
        logging_context: Captured logging state from the parent process.
        proxy_queue: Optional multiprocessing queue that receives worker log
            records.
        manager_logger_name: Optional parent logger name used to namespace the
            worker logger.

    Notes:
        The fitting manager uses the `spawn` start method. Other process start
        methods are outside the assumptions of this initialization routine.
    """
    global _WORKER_LOGGER

    # 0. Apply the captured logging context to the current process
    logging_context.apply()

    if proxy_queue is not None:
        # 1. Create the QueueHandler with the proxy queue
        queue_handler = QueueHandler(proxy_queue)

        # 2. Get the root logger for this worker and add the QueueHandler to it
        root_logger = logging.getLogger()
        root_logger.addHandler(queue_handler)

    # 3. Create the specific child logger of the worker
    base_worker_logger_name = f"worker_{os.getpid()}"

    if not manager_logger_name:
        worker_logger_name = base_worker_logger_name
    else:
        worker_logger_name = f"{manager_logger_name}.{base_worker_logger_name}"

    worker_logger = logging.getLogger(worker_logger_name)

    # 4. Assign the child logger of the worker to global worker logger variable
    _WORKER_LOGGER = worker_logger


def _initialize_worker(
    models: tuple[ComputationalModel, ...],
    logging_context: LoggingContext,
    optimizer_options: dict[str, object] | None = None,
    proxy_queue: Queue[logging.LogRecord] | None = None,
    manager_logger_name: str | None = None,
) -> None:
    """Initialize process-global fitting state inside a spawned worker.

    The model tuple and optimizer options are stored in worker globals so individual
    submitted tasks remain small. Worker logging is initialized from the captured
    parent-process logging context.

    Args:
        models: Validated models shared by all tasks executed in the worker.
        logging_context: Parent-process logging state to reproduce in the worker.
        optimizer_options: Optional optimizer options shared by worker fits.
        proxy_queue: Optional queue used to forward worker log records to the parent.
        manager_logger_name: Optional parent logger name used to namespace the
            process-specific worker logger.
    """

    global _WORKER_MODELS
    global _WORKER_OPTIMIZER_OPTIONS
    global _WORKER_LOGGER

    _WORKER_MODELS = models
    _WORKER_OPTIMIZER_OPTIONS = optimizer_options

    _worker_logger_initialization(
        logging_context,
        proxy_queue,
        manager_logger_name,
    )


def _fit_models_for_subject_using_worker(task: SubjectFitTask) -> list[ModelFitResult]:
    """Fit one participant using process-global worker configuration.

    Args:
        task: Serializable participant-fitting task submitted to the process pool.

    Returns:
        One fit result per worker-configured model for the participant.

    Raises:
        RuntimeError: If the process-pool initializer did not populate the worker
            model tuple before the task is executed.
    """

    if _WORKER_MODELS is None:
        raise RuntimeError("Worker process was not initialized with models.")

    return _fit_models_for_subject(
        _WORKER_MODELS,
        task=task,
        optimizer_options=_WORKER_OPTIMIZER_OPTIONS,
        logger=_WORKER_LOGGER,
    )


def _fit_all_subjects_serial(
    tasks: Sequence[SubjectFitTask],
    models: tuple[ComputationalModel, ...],
    *,
    optimizer_options: Mapping[str, object] | None,
    show_progress: bool,
    manager_logger_name: str | None = None,
) -> list[ModelFitResult]:
    """Fit all participant tasks sequentially in the current process.

    Args:
        tasks: Participant tasks to execute in order.
        models: Validated models fitted to every task.
        optimizer_options: Optional optimizer options forwarded to each model fit.
        show_progress: Whether to display the subject-level progress bar.
        manager_logger_name: Optional logger name used for fitting diagnostics.

    Returns:
        Flattened model-fit results in participant-task order and model order.
    """
    manager_logger = logging.getLogger(manager_logger_name)

    manager_logger.info(
        "Fitting %d subjects sequentially (no workers) for the models: %r.",
        len(tasks),
        [model.name for model in models],
    )

    iterator = tqdm(
        tasks,
        total=len(tasks),
        desc="Fitting subjects",
        unit="subject",
        disable=not show_progress,
    )

    results: list[ModelFitResult] = []

    for task in iterator:
        results.extend(
            _fit_models_for_subject(
                models,
                task=task,
                optimizer_options=optimizer_options,
                logger=manager_logger,
            )
        )

    manager_logger.info("Completed fitting all subjects sequentially.")

    return results


def _multi_worker_subjects_fittig(
    tasks: Sequence[SubjectFitTask],
    models: tuple[ComputationalModel, ...],
    *,
    optimizer_options: dict[str, object] | None,
    show_progress: bool,
    n_workers: int | None,
    results_by_position: list[list[ModelFitResult] | None],
    logging_context: LoggingContext,
    mp_context: BaseContext | None = None,
    proxy_queue: Queue[logging.LogRecord] | None = None,
    manager_logger_name: str | None = None,
) -> None:
    """Submit participant tasks to a process pool and collect results by task position.

    Futures may complete in arbitrary order, so each result is written into the
    preallocated slot associated with its original task position. This preserves
    deterministic output ordering after parallel execution.

    Args:
        tasks: Participant tasks to submit.
        models: Validated models installed in each worker process.
        optimizer_options: Optional optimizer options shared by worker fits.
        show_progress: Whether to display completion progress.
        n_workers: Requested process-pool size, or `None` for the executor default.
        results_by_position: Preallocated result slots indexed by task position.
        logging_context: Parent-process logging state reproduced in workers.
        mp_context: Optional multiprocessing context used by the executor.
        proxy_queue: Optional queue forwarding worker log records to the parent.
        manager_logger_name: Optional logger name restored in worker processes.

    Raises:
        RuntimeError: If any submitted participant task raises while being fitted.
    """
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp_context,
        initializer=_initialize_worker,
        initargs=(
            models,
            logging_context,
            optimizer_options,
            proxy_queue,
            manager_logger_name,
        ),
    ) as executor:
        futures: dict[Future[list[ModelFitResult]], int] = {
            executor.submit(_fit_models_for_subject_using_worker, task): position
            for position, task in enumerate(tasks)
        }

        completed = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Fitting subjects",
            unit="subject",
            disable=not show_progress,
        )

        for future in completed:
            position = futures[future]
            task = tasks[position]

            try:
                results_by_position[position] = future.result()
            except Exception as error:
                raise RuntimeError(
                    "Failed to fit subject "
                    f"(n_trials={task.n_trials}, subject_id={task.subject_id})."
                ) from error


def _fit_all_subjects_parallel(
    tasks: Sequence[SubjectFitTask],
    models: tuple[ComputationalModel, ...],
    *,
    optimizer_options: Mapping[str, object] | None,
    show_progress: bool,
    n_workers: int | None,
    manager_logger_name: str | None = None,
) -> list[ModelFitResult]:
    """Fit participant tasks in spawned worker processes while preserving result order.

    The function captures the current logging configuration and uses the `spawn`
    multiprocessing context. When the root logger has active non-null handlers,
    worker records are forwarded through a multiprocessing queue and handled in the
    parent process.

    Args:
        tasks: Participant tasks to execute.
        models: Validated models fitted to every task.
        optimizer_options: Optional optimizer options forwarded to each model fit.
        show_progress: Whether to display subject-level progress.
        n_workers: Worker-process count, or `None` to use the executor default.
        manager_logger_name: Optional logger name used by the manager and workers.

    Returns:
        Flattened model-fit results in original participant-task order.

    Raises:
        RuntimeError: If a worker task fails or a completed task does not populate
            its expected result slot.
    """
    logging_context = LoggingContext()

    manager_logger = logging.getLogger(manager_logger_name)

    manager_logger.info(
        "Fitting %d subjects in parallel (%s) for the models: %r.",
        len(tasks),
        f"{n_workers} workers" if n_workers is not None else "all available CPU cores workers",
        [model.name for model in models],
    )

    root_logger_handlers = logging.getLogger().handlers

    options = dict(optimizer_options) if optimizer_options is not None else None
    results_by_position: list[list[ModelFitResult] | None] = [None] * len(tasks)

    mp_context = mp.get_context("spawn")
    manager_logger.debug("Using multiprocessing context: %r", mp_context.get_start_method())

    if not root_logger_handlers or all(
        isinstance(handler, logging.NullHandler) for handler in root_logger_handlers
    ):
        manager_logger.debug(
            "No root logger handlers that aren't NullHandler were found, so multiprocessing logging will be disabled (i.e. no proxy queue and no queue listener will be created)."
        )
        _multi_worker_subjects_fittig(
            tasks,
            models,
            show_progress=show_progress,
            n_workers=n_workers,
            optimizer_options=options,
            results_by_position=results_by_position,
            logging_context=logging_context,
            mp_context=mp_context,
            proxy_queue=None,
            manager_logger_name=manager_logger_name,
        )
    else:
        manager_logger.debug(
            "Root logger handlers that aren't NullHandler were found, so multiprocessing logging will be enabled (i.e. a proxy queue and a queue listener will be created)."
        )
        with mp_context.Manager() as manager:
            # Create the proxy queue
            proxy_queue: Queue[logging.LogRecord] = manager.Queue(-1)

            # Create the QueueListener (listens to proxy queue, writes terminal and file)
            listener = QueueListener(
                proxy_queue,
                *root_logger_handlers,
                respect_handler_level=True,
            )

            # Start the listener
            listener.start()

            try:
                # Context manager for Executor, passing the proxy queue in initargs
                _multi_worker_subjects_fittig(
                    tasks,
                    models,
                    show_progress=show_progress,
                    n_workers=n_workers,
                    optimizer_options=options,
                    results_by_position=results_by_position,
                    logging_context=logging_context,
                    mp_context=mp_context,
                    proxy_queue=proxy_queue,
                    manager_logger_name=manager_logger_name,
                )
            finally:
                # Stop listening
                listener.stop()

    results: list[ModelFitResult] = []

    for subject_results in results_by_position:
        if subject_results is None:
            raise RuntimeError("A subject fit completed without returning results.")

        results.extend(subject_results)

    manager_logger.info("Completed fitting all subjects in parallel.")

    return results


def fit_all_subjects(
    data: pd.DataFrame,
    models: Sequence[ComputationalModel],
    *,
    optimizer_options: Mapping[str, object] | None = None,
    show_progress: bool = True,
    n_workers: int | None = None,
    n_subjects: int | None = None,
    subject_model_warm_starts_provider: SubjectModelWarmStartsProvider | None = None,
) -> list[ModelFitResult]:
    """Fit every requested model to every selected participant.

    The function validates model identities, converts the long-format dataset to
    participant tasks, and dispatches those tasks either sequentially (`n_workers=1`)
    or through spawned worker processes. Optional warm starts are requested while
    tasks are built in the main process.

    Args:
        data: Complete long-format IGT trial table.
        models: Models to fit to each selected participant.
        optimizer_options: Optional optimizer options forwarded to every model fit.
        show_progress: Whether to display participant-level progress.
        n_workers: Number of worker processes. Use `1` for serial execution and
            `None` for the process-pool default.
        n_subjects: Optional maximum number of participants to fit.
        subject_model_warm_starts_provider: Optional callback supplying additional
            starting points for a model and participant key.

    Returns:
        Flattened per-participant, per-model fit results.

    Raises:
        TypeError: If `n_workers` is neither an integer nor `None`.
        ValueError: If `n_workers` is not positive, model validation fails, or the
            selected dataset cannot be converted into valid participant tasks.
        RuntimeError: If parallel fitting fails for any participant.
    """

    if n_workers is not None:
        if not isinstance(n_workers, int):
            raise TypeError(
                f"n_workers must be an integer or None, got {type(n_workers).__name__}."
            )

        if n_workers <= 0:
            raise ValueError("n_workers must be greater than zero.")

    manager_logger = logging.getLogger(__name__)

    manager_logger.debug("Validating the models.")
    model_tuple = _validate_models(models)

    manager_logger.debug("Optimizer options: %r", optimizer_options)

    for model in model_tuple:
        manager_logger.debug(
            "Model %r has %d parameters with bounds: %r",
            model.name,
            model.n_parameters,
            model._parameter_name_to_bound_map,
        )

    manager_logger.debug("Building subject tasks from the dataset.")
    tasks = _build_subject_tasks(
        data,
        models=model_tuple,
        n_subjects=n_subjects,
        subject_model_warm_starts_provider=subject_model_warm_starts_provider,
    )

    manager_logger.info("Starting to fit %d subjects", len(tasks))

    if n_workers is not None and n_workers == 1:
        manager_logger.info(
            "The number of workers is 1, so fitting will be done serially in the main process."
        )
        return _fit_all_subjects_serial(
            tasks,
            model_tuple,
            optimizer_options=optimizer_options,
            show_progress=show_progress,
            manager_logger_name=manager_logger.name,
        )

    manager_logger.info(
        "The number of workers is %s, so fitting will be done in parallel using multiple processes.",
        n_workers if n_workers is not None else "unspecified (all available CPU cores)",
    )
    return _fit_all_subjects_parallel(
        tasks,
        model_tuple,
        optimizer_options=optimizer_options,
        show_progress=show_progress,
        n_workers=n_workers,
        manager_logger_name=manager_logger.name,
    )
