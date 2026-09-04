"""End-to-end orchestration for post-fit result analysis.

The pipeline loads and validates result tables, derives descriptive and inferential
summaries, writes CSV and LaTeX table outputs, generates figures, and emits the
analysis report.
See [`generate_results_analysis`][igt.analysis.pipeline.generate_results_analysis].
"""

from dataclasses import dataclass
from pathlib import Path

from igt.typing import StrPathLike
from igt.utils.io import normalize_path, write_csv, write_latex_table

from .artifacts import GeneratedFigure, GeneratedTable
from .config import AnalysisConfig
from .inference import (
    build_criterion_inference_table,
    build_model_win_inference_table,
)
from .io import load_result_tables
from .plots import generate_all_figures
from .report import write_analysis_report
from .tables import (
    build_boundary_summary_table,
    build_model_win_table,
    build_parameter_summary_table,
    build_study_preference_table,
    build_subject_comparison_table,
)
from .validation import validate_result_tables


@dataclass(frozen=True, slots=True)
class AnalysisOutputs:
    """Artifacts produced by one complete result-analysis run.

    Attributes:
        output_directory: Root directory for the run.
        report_path: Generated plain-text analysis report.
        figures: Logical generated figures, each grouping all physical formats.
        tables: Logical generated tables, each grouping CSV and LaTeX outputs.
    """

    output_directory: Path
    report_path: Path
    figures: tuple[GeneratedFigure, ...]
    tables: tuple[GeneratedTable, ...]

    @property
    def figure_paths(self) -> tuple[Path, ...]:
        """Return every physical figure path across all logical figures."""

        return tuple(path for figure in self.figures for path in figure.paths)

    @property
    def table_paths(self) -> tuple[Path, ...]:
        """Return every physical table path across all logical tables."""

        return tuple(path for table in self.tables for path in table.paths)


def generate_results_analysis(
    fits_path: StrPathLike,
    comparison_path: StrPathLike,
    summary_path: StrPathLike,
    output_directory: StrPathLike,
    *,
    config: AnalysisConfig | None = None,
) -> AnalysisOutputs:
    """Validate final result CSVs and generate the complete analysis artifact set.

    The pipeline loads and cross-validates the fitting, comparison, and summary
    tables; derives descriptive and inferential tables; writes every derived table as
    both CSV and LaTeX; generates all standard figures in every configured format; and
    writes a compact text report.

    Args:
        fits_path: Complete per-subject, per-model fit-results CSV.
        comparison_path: Model-comparison CSV generated from the same fit table.
        summary_path: Aggregate model-summary CSV generated from the same fit table.
        output_directory: Root directory in which analysis artifacts are written.
        config: Optional analysis configuration. Defaults to
            [AnalysisConfig][igt.analysis.config.AnalysisConfig].

    Returns:
        Logical generated figures and tables plus the report and root output paths.

    Raises:
        ValueError: If the result tables are inconsistent or contain values that
            violate the expected final-result schema.
    """

    analysis_config = config if config is not None else AnalysisConfig()
    normalized_output_directory = normalize_path(
        output_directory,
        parameter_name="output_directory",
    )
    tables = validate_result_tables(
        load_result_tables(
            fits_path,
            comparison_path,
            summary_path,
        ),
        analysis_config,
    )

    figures_directory = normalized_output_directory / "figures"
    tables_directory = normalized_output_directory / "tables"
    subject_comparison = build_subject_comparison_table(tables.comparison)
    study_preference = build_study_preference_table(subject_comparison)
    boundary_summary = build_boundary_summary_table(tables.fits)
    parameter_summary = build_parameter_summary_table(
        tables.fits,
        analysis_config,
    )
    model_win_table = build_model_win_table(subject_comparison)
    criterion_inference = build_criterion_inference_table(
        subject_comparison,
        analysis_config,
    )
    model_win_inference = build_model_win_inference_table(
        subject_comparison,
        analysis_config,
    )

    table_outputs = {
        "subject_level_model_comparison": subject_comparison,
        "study_model_preference": study_preference,
        "boundary_summary": boundary_summary,
        "parameter_summary": parameter_summary,
        "model_win_summary": model_win_table,
        "criterion_difference_inference": criterion_inference,
        "model_win_inference": model_win_inference,
    }
    generated_tables: list[GeneratedTable] = []

    for table_name, data in table_outputs.items():
        output_stem = tables_directory / table_name
        csv_path = output_stem.with_suffix(".csv")
        latex_path = output_stem.with_suffix(".tex")
        write_csv(data, csv_path)
        write_latex_table(data, latex_path)
        generated_tables.append(
            GeneratedTable(
                output_stem=output_stem,
                paths=(csv_path, latex_path),
            )
        )

    generated_figures = generate_all_figures(
        tables.fits,
        subject_comparison,
        study_preference,
        boundary_summary,
        model_win_table,
        criterion_inference,
        model_win_inference,
        output_directory=figures_directory,
        config=analysis_config,
    )
    report_path = normalized_output_directory / "analysis_report.txt"
    write_analysis_report(
        tables.summary,
        model_win_table,
        study_preference,
        boundary_summary,
        parameter_summary,
        criterion_inference,
        model_win_inference,
        generated_figures,
        tuple(generated_tables),
        report_path=report_path,
    )

    return AnalysisOutputs(
        output_directory=normalized_output_directory,
        report_path=report_path,
        figures=generated_figures,
        tables=tuple(generated_tables),
    )
