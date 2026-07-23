"""Execute model fitting across IGT subjects.

This module owns dataset-to-subject conversion and subject-level scheduling.
The statistical comparison of completed fits belongs in ``comparison.py``.
"""

from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from igt.fitting import ModelFitResult, fit_model
from igt.models.base import ComputationalModel, SubjectData
from igt.rdata_preprocessing import iter_subject_trials

_REQUIRED_TRIAL_COLUMNS = {
    "source_study",
    "choice",
    "win",
    "loss",
}


@dataclass(frozen=True, slots=True)
class SubjectFitTask:
    """Serializable input required to fit every model to one subject."""

    n_trials: int
    subject_id: int
    source_study: str
    data: SubjectData


_WORKER_MODELS: tuple[ComputationalModel, ...] | None = None
_WORKER_OPTIMIZER_OPTIONS: dict[str, object] | None = None


def subject_data_from_trials(subject_trials: pd.DataFrame) -> SubjectData:
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


def _subject_study(subject_trials: pd.DataFrame) -> str:
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


def _build_subject_tasks(data: pd.DataFrame, n_subjects: int | None = None) -> list[SubjectFitTask]:
    """Create one serializable fitting task per subject."""

    tasks: list[SubjectFitTask] = []

    for (n_trials, subject_id), subject_trials in iter_subject_trials(data, n_subjects=n_subjects):
        tasks.append(
            SubjectFitTask(
                n_trials=n_trials,
                subject_id=subject_id,
                source_study=_subject_study(subject_trials),
                data=subject_data_from_trials(subject_trials),
            )
        )

    if not tasks:
        raise ValueError("The dataset does not contain any subjects.")

    return tasks


def fit_models_for_subject(
    models: Sequence[ComputationalModel],
    *,
    task: SubjectFitTask,
    optimizer_options: Mapping[str, object] | None = None,
) -> list[ModelFitResult]:
    """Fit every supplied model to one subject."""

    model_tuple = _validate_models(models)

    return [
        fit_model(
            model,
            task.data,
            n_trials=task.n_trials,
            subject_id=task.subject_id,
            source_study=task.source_study,
            optimizer_options=optimizer_options,
        )
        for model in model_tuple
    ]


def _initialize_worker(
    models: tuple[ComputationalModel, ...],
    optimizer_options: dict[str, object] | None,
) -> None:
    """Store read-only fitting configuration once in each worker process."""

    global _WORKER_MODELS
    global _WORKER_OPTIMIZER_OPTIONS

    _WORKER_MODELS = models
    _WORKER_OPTIMIZER_OPTIONS = optimizer_options


def _fit_subject_in_worker(task: SubjectFitTask) -> list[ModelFitResult]:
    """Fit one subject inside a worker initialized by the process pool."""

    if _WORKER_MODELS is None:
        raise RuntimeError("Worker process was not initialized with models.")

    return fit_models_for_subject(
        _WORKER_MODELS,
        task=task,
        optimizer_options=_WORKER_OPTIMIZER_OPTIONS,
    )


def _fit_all_subjects_serial(
    tasks: Sequence[SubjectFitTask],
    models: tuple[ComputationalModel, ...],
    *,
    optimizer_options: Mapping[str, object] | None,
    show_progress: bool,
) -> list[ModelFitResult]:
    """Fit subject tasks sequentially."""

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
            fit_models_for_subject(
                models,
                task=task,
                optimizer_options=optimizer_options,
            )
        )

    return results


def _fit_all_subjects_parallel(
    tasks: Sequence[SubjectFitTask],
    models: tuple[ComputationalModel, ...],
    *,
    optimizer_options: Mapping[str, object] | None,
    show_progress: bool,
    n_workers: int,
) -> list[ModelFitResult]:
    """Fit independent subjects using separate Python processes."""

    options = dict(optimizer_options) if optimizer_options is not None else None
    results_by_position: list[list[ModelFitResult] | None] = [None] * len(tasks)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_initialize_worker,
        initargs=(models, options),
    ) as executor:
        futures: dict[Future[list[ModelFitResult]], int] = {
            executor.submit(_fit_subject_in_worker, task): position
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

    results: list[ModelFitResult] = []

    for subject_results in results_by_position:
        if subject_results is None:
            raise RuntimeError("A subject fit completed without returning results.")

        results.extend(subject_results)

    return results


def fit_all_subjects(
    data: pd.DataFrame,
    models: Sequence[ComputationalModel],
    *,
    optimizer_options: Mapping[str, object] | None = None,
    show_progress: bool = True,
    n_workers: int = 1,
    n_subjects: int | None = None,
) -> list[ModelFitResult]:
    """Fit every model to every subject, serially or in parallel.

    Args:
        data: Complete long-format IGT table.
        models: Models to fit to every subject.
        optimizer_options: Options passed to every L-BFGS-B run.
        show_progress: Whether to display a subject progress bar.
        n_workers: Number of worker processes. Use ``1`` for serial fitting.
    """

    if not isinstance(n_workers, int):
        raise TypeError("n_workers must be an integer.")

    if n_workers <= 0:
        raise ValueError("n_workers must be greater than zero.")

    model_tuple = _validate_models(models)
    tasks = _build_subject_tasks(data, n_subjects=n_subjects)

    if n_workers == 1:
        return _fit_all_subjects_serial(
            tasks,
            model_tuple,
            optimizer_options=optimizer_options,
            show_progress=show_progress,
        )

    return _fit_all_subjects_parallel(
        tasks,
        model_tuple,
        optimizer_options=optimizer_options,
        show_progress=show_progress,
        n_workers=n_workers,
    )
