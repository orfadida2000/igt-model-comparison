"""Population-level inference for paired model-comparison results."""

from typing import Literal, Protocol, cast

import numpy as np
from pandas import DataFrame
from scipy.stats import binomtest, bootstrap, wilcoxon

from igt.typing import FloatArray

from .config import AnalysisConfig


class _WilcoxonResult(Protocol):
    statistic: float
    pvalue: float


_CRITERION_DIFFERENCE_COLUMNS = {
    "AIC": "aic_q_minus_pvl",
    "BIC": "bic_q_minus_pvl",
}


def _bootstrap_confidence_interval(
    values: FloatArray,
    *,
    statistic_name: Literal["mean", "median"],
    config: AnalysisConfig,
) -> tuple[float, float]:
    """Return a reproducible BCa bootstrap confidence interval."""

    if values.ndim != 1:
        raise ValueError("Bootstrap values must be one-dimensional.")

    if values.size < 2:
        raise ValueError("At least two observations are required for bootstrap inference.")

    if statistic_name == "mean":
        statistic = np.mean
    elif statistic_name == "median":
        statistic = np.median
    else:
        raise ValueError("statistic_name must be 'mean' or 'median'.")

    if np.all(values == values[0]):
        constant_value = float(values[0])
        return constant_value, constant_value

    result = bootstrap(
        (values,),
        statistic,
        vectorized=True,
        n_resamples=config.bootstrap_resamples,
        batch=min(config.bootstrap_resamples, 1_000),
        confidence_level=config.confidence_level,
        alternative="two-sided",
        method="BCa",
        rng=np.random.default_rng(config.bootstrap_seed),
    )
    lower = float(result.confidence_interval.low)
    upper = float(result.confidence_interval.high)

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Bootstrap confidence interval contains non-finite bounds.")

    return lower, upper


def build_criterion_inference_table(
    subject_comparison: DataFrame,
    config: AnalysisConfig,
) -> DataFrame:
    """Infer population-level AIC and BIC differences between the models.

    Criterion differences are defined as Q-learning minus PVL-Delta, so
    positive values favor PVL-Delta. Bootstrap confidence intervals quantify
    uncertainty in the mean and median differences. The Wilcoxon signed-rank
    test evaluates whether the paired difference distribution is centered at
    zero under its symmetry assumption.
    """

    records: list[dict[str, float | int | str]] = []

    for criterion, column_name in _CRITERION_DIFFERENCE_COLUMNS.items():
        values = subject_comparison[column_name].to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"No subject-level {criterion} differences are available.")

        if not np.isfinite(values).all():
            raise ValueError(f"Subject-level {criterion} differences contain non-finite values.")

        mean_ci_lower, mean_ci_upper = _bootstrap_confidence_interval(
            values,
            statistic_name="mean",
            config=config,
        )
        median_ci_lower, median_ci_upper = _bootstrap_confidence_interval(
            values,
            statistic_name="median",
            config=config,
        )
        nonzero_count = int(np.count_nonzero(values))

        if nonzero_count == 0:
            wilcoxon_statistic = 0.0
            wilcoxon_p_value = 1.0
        else:
            wilcoxon_result = cast(
                _WilcoxonResult,
                wilcoxon(
                    values,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                ),
            )
            wilcoxon_statistic = float(wilcoxon_result.statistic)
            wilcoxon_p_value = float(wilcoxon_result.pvalue)

        records.append(
            {
                "criterion": criterion,
                "difference_definition": f"{criterion}(Q-learning) - {criterion}(PVL-Delta)",
                "n_subjects": int(values.size),
                "mean_difference": float(np.mean(values)),
                "mean_ci_lower": mean_ci_lower,
                "mean_ci_upper": mean_ci_upper,
                "median_difference": float(np.median(values)),
                "median_ci_lower": median_ci_lower,
                "median_ci_upper": median_ci_upper,
                "confidence_level": config.confidence_level,
                "bootstrap_resamples": config.bootstrap_resamples,
                "bootstrap_seed": config.bootstrap_seed,
                "bootstrap_method": "BCa",
                "wilcoxon_statistic": wilcoxon_statistic,
                "wilcoxon_p_value": wilcoxon_p_value,
                "wilcoxon_alternative": "two-sided",
                "wilcoxon_zero_method": "wilcox",
                "n_nonzero_differences": nonzero_count,
            }
        )

    return DataFrame(records)


def build_model_win_inference_table(
    subject_comparison: DataFrame,
    config: AnalysisConfig,
) -> DataFrame:
    """Infer whether PVL-Delta wins more often than a 50/50 null expectation.

    Exact ties are excluded from the binomial test and from the reported
    non-tied PVL-Delta win rate. The confidence interval is the exact binomial
    interval returned by SciPy.
    """

    records: list[dict[str, float | int | str]] = []
    n_subjects = len(subject_comparison)

    if n_subjects == 0:
        raise ValueError("No eligible subjects are available for model-win inference.")

    for criterion in ("aic", "bic"):
        criterion_label = criterion.upper()
        q_best = subject_comparison[f"q_best_{criterion}"].to_numpy(dtype=np.bool_)
        pvl_best = subject_comparison[f"pvl_best_{criterion}"].to_numpy(dtype=np.bool_)
        pvl_win_mask = pvl_best & ~q_best
        q_win_mask = q_best & ~pvl_best
        tie_mask = pvl_best & q_best
        unresolved_mask = ~pvl_best & ~q_best

        if unresolved_mask.any():
            raise ValueError(
                f"{criterion_label} contains {int(unresolved_mask.sum())} subject(s) "
                "without a model winner or tie."
            )

        pvl_wins = int(pvl_win_mask.sum())
        q_wins = int(q_win_mask.sum())
        ties = int(tie_mask.sum())
        n_non_ties = pvl_wins + q_wins

        if n_non_ties == 0:
            pvl_win_rate = float("nan")
            ci_lower = float("nan")
            ci_upper = float("nan")
            binomial_p_value = 1.0
        else:
            binomial_result = binomtest(
                pvl_wins,
                n=n_non_ties,
                p=0.5,
                alternative="two-sided",
            )
            confidence_interval = binomial_result.proportion_ci(
                confidence_level=config.confidence_level,
                method="exact",
            )
            pvl_win_rate = pvl_wins / n_non_ties
            ci_lower = float(confidence_interval.low)
            ci_upper = float(confidence_interval.high)
            binomial_p_value = float(binomial_result.pvalue)

        records.append(
            {
                "criterion": criterion_label,
                "n_subjects": n_subjects,
                "pvl_wins": pvl_wins,
                "q_wins": q_wins,
                "ties": ties,
                "n_non_ties": n_non_ties,
                "pvl_win_rate_non_tied": pvl_win_rate,
                "pvl_win_rate_ci_lower": ci_lower,
                "pvl_win_rate_ci_upper": ci_upper,
                "confidence_level": config.confidence_level,
                "confidence_interval_method": "exact",
                "null_pvl_win_probability": 0.5,
                "binomial_p_value": binomial_p_value,
                "binomial_alternative": "two-sided",
            }
        )

    return DataFrame(records)
