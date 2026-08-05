"""Execute model fitting across IGT subjects.

This module owns dataset-to-subject conversion and subject-level scheduling.
The statistical comparison of completed fits belongs in ``comparison.py``.
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
    "source_study",
    "choice",
    "win",
    "loss",
}


_WORKER_MODELS: tuple[ComputationalModel, ...] | None = None
_WORKER_OPTIMIZER_OPTIONS: dict[str, object] | None = None
_WORKER_LOGGER: logging.Logger | None = None


def _compute_subject_data_from_trials(subject_trials: pd.DataFrame) -> SubjectData:
    """Convert one subject's long-format trial rows into model input arrays."""

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
    """Return and validate the unique source study for one subject."""

    studies = subject_trials["source_study"].astype("string").str.strip().dropna().unique()

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
    """Validate and freeze the supplied model sequence as a tuple."""

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
    """Create one serializable fitting task per subject."""

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
    """Fit every supplied model to one subject."""

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
    """
    Initialize the logger for a worker process.

    This function is called in the context of a worker process, and it sets up the logging configuration for that process.
    It first applies the captured logging context of the main process, then if `proxy_queue` is not None, it creates a QueueHandler that sends log records to the provided proxy queue and adds it to the root logger.
    Finally, it creates a specific child logger for the worker process and assigns it to the global _WORKER_LOGGER variable.

    Args:
        logging_context (LoggingContext): The logging context to apply.
        proxy_queue (Queue[logging.LogRecord] | None): A multiprocessing proxy queue that will receive log records from the worker process or None if no multiprocessing logging is desired.
        manager_logger_name (str | None): The name of the manager logger in which the worker logger will be nested. If None or empty, the worker logger will be a top-level logger.

    Note:
        This function assumes that the worker process was created using the start method 'spawn', other start methods may produce unexpected behavior.
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
    """Store read-only fitting configuration once in each worker process."""

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
    """Fit one subject inside a worker initialized by the process pool."""

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
    """Fit subject tasks sequentially."""
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
    """Fit independent subjects using separate Python processes."""
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
    """Fit every model to every subject, serially or in parallel.

    Args:
        data: Complete long-format IGT table.
        models: Models to fit to every subject.
        optimizer_options: Options passed to every L-BFGS-B run.
        show_progress: Whether to display a subject progress bar.
        n_workers: Number of worker processes. Use ``1`` for serial fitting.
        n_subjects: Number of subjects to fit. If None, fit all subjects.
        subject_model_warm_starts_provider: Optional callable that provides warm-start parameter values for a given model and subject.

    Returns:
        A list of model-fit results, one for each model and subject.
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
