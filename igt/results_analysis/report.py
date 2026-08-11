"""Text reporting for result-analysis outputs."""

import numpy as np
from pandas import DataFrame

from igt.constants.schema import MODEL_COLUMN
from igt.typing import LineEnding, StrPathLike
from igt.utils.io import normalize_path, write_text


def _model_display_name(model_name: str) -> str:
    """Return a readable model label."""
    model_name = model_name.strip()

    name_components = [c.strip() for c in model_name.split("_") if c.strip()]

    if len(name_components) == 0:
        return model_name
    elif len(name_components) == 1:
        return name_components[0].upper()
    elif len(name_components) == 2:
        return f"{name_components[0].upper()}-{name_components[1].title()}"
    else:
        return f"{name_components[0].upper()}-{name_components[1].title()} {' '.join(c.title() for c in name_components[2:])}"


def write_analysis_report(
    summary: DataFrame,
    model_win_table: DataFrame,
    study_preference: DataFrame,
    boundary_summary: DataFrame,
    parameter_summary: DataFrame,
    generated_figures: tuple[StrPathLike, ...],
    generated_tables: tuple[StrPathLike, ...],
    *,
    report_path: StrPathLike,
) -> None:
    """Write a compact audit and interpretation report."""

    lines = [
        "IGT Model-Results Analysis Report",
        "=================================",
        "",
        "Validated inputs",
        "----------------",
        "The fit, comparison, and summary tables passed schema, key,",
        "cross-table alignment, and summary-recomputation checks.",
        "",
        "Aggregate model results",
        "-----------------------",
    ]

    summary_model_names = summary[MODEL_COLUMN].astype("string").to_numpy(dtype=str)
    summary_n_fits = summary["n_fits"].to_numpy(dtype=np.int64)
    summary_n_converged = summary["n_converged"].to_numpy(dtype=np.int64)
    summary_convergence_rates = summary["convergence_rate"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    summary_mean_nll = summary["mean_negative_log_likelihood"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    summary_mean_aic = summary["mean_aic"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    summary_mean_bic = summary["mean_bic"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    summary_aic_wins = summary["aic_wins"].to_numpy(dtype=np.int64)
    summary_bic_wins = summary["bic_wins"].to_numpy(dtype=np.int64)

    for (
        model_name,
        n_fits,
        n_converged,
        convergence_rate,
        mean_nll,
        mean_aic,
        mean_bic,
        aic_wins,
        bic_wins,
    ) in zip(
        summary_model_names,
        summary_n_fits,
        summary_n_converged,
        summary_convergence_rates,
        summary_mean_nll,
        summary_mean_aic,
        summary_mean_bic,
        summary_aic_wins,
        summary_bic_wins,
        strict=True,
    ):
        lines.extend(
            [
                f"{_model_display_name(model_name)}:",
                f"  Fits: {n_fits}",
                f"  Converged: {n_converged} ({convergence_rate:.2%})",
                f"  Mean NLL: {mean_nll:.6f}",
                f"  Mean AIC: {mean_aic:.6f}",
                f"  Mean BIC: {mean_bic:.6f}",
                f"  AIC wins: {aic_wins}",
                f"  BIC wins: {bic_wins}",
                "",
            ]
        )

    lines.extend(
        [
            "Model-win rates",
            "---------------",
        ]
    )

    win_criteria = model_win_table["criterion"].astype("string").to_numpy(dtype=str)
    win_model_names = model_win_table[MODEL_COLUMN].astype("string").to_numpy(dtype=str)
    win_counts = model_win_table["wins"].to_numpy(dtype=np.int64)
    win_subject_counts = model_win_table["n_subjects"].to_numpy(dtype=np.int64)
    win_rates = model_win_table["win_rate"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    for criterion, model_name, wins, n_subjects, win_rate in zip(
        win_criteria,
        win_model_names,
        win_counts,
        win_subject_counts,
        win_rates,
        strict=True,
    ):
        lines.append(
            f"{criterion} — {_model_display_name(model_name)}: {wins}/{n_subjects} ({win_rate:.2%})"
        )

    lines.extend(
        [
            "",
            "Source-study coverage",
            "---------------------",
            f"Studies summarized: {len(study_preference)}",
            "",
            "Boundary diagnostics",
            "--------------------",
        ]
    )

    any_bound_rows = boundary_summary.loc[boundary_summary["category"].eq("at_least_one_any_bound")]
    boundary_model_names = any_bound_rows[MODEL_COLUMN].astype("string").to_numpy(dtype=str)
    boundary_counts = any_bound_rows["count"].to_numpy(dtype=np.int64)
    boundary_fit_counts = any_bound_rows["n_fits"].to_numpy(dtype=np.int64)
    boundary_rates = any_bound_rows["rate"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    for model_name, count, n_fits, rate in zip(
        boundary_model_names,
        boundary_counts,
        boundary_fit_counts,
        boundary_rates,
        strict=True,
    ):
        lines.append(
            f"{_model_display_name(model_name)}: "
            f"{count}/{n_fits} fits ({rate:.2%}) had at least one "
            "parameter at a bound."
        )

    lines.extend(
        [
            "",
            "Parameter distributions",
            "-----------------------",
        ]
    )

    parameter_model_names = parameter_summary[MODEL_COLUMN].astype("string").to_numpy(dtype=str)
    parameter_names = parameter_summary["parameter"].astype("string").to_numpy(dtype=str)
    parameter_medians = parameter_summary["median"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    parameter_q1 = parameter_summary["q1"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    parameter_q3 = parameter_summary["q3"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    parameter_lower_rates = parameter_summary["rate_at_lower_bound"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    parameter_upper_rates = parameter_summary["rate_at_upper_bound"].to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    for (
        model_name,
        parameter_name,
        median,
        q1,
        q3,
        lower_rate,
        upper_rate,
    ) in zip(
        parameter_model_names,
        parameter_names,
        parameter_medians,
        parameter_q1,
        parameter_q3,
        parameter_lower_rates,
        parameter_upper_rates,
        strict=True,
    ):
        lines.append(
            f"{_model_display_name(model_name)} / {parameter_name}: "
            f"median={median:.6g}, "
            f"IQR=[{q1:.6g}, {q3:.6g}], "
            f"lower-bound rate={lower_rate:.2%}, "
            f"upper-bound rate={upper_rate:.2%}"
        )

    lines.extend(
        [
            "",
            "Generated artifacts",
            "-------------------",
            f"Figures: {len(generated_figures)}",
        ]
    )
    lines.extend(f"  {normalize_path(path).as_posix()}" for path in generated_figures)
    lines.append(f"Tables: {len(generated_tables)}")
    lines.extend(f"  {normalize_path(path).as_posix()}" for path in generated_tables)

    write_text("\n".join(lines), report_path, newline=LineEnding.LF)
