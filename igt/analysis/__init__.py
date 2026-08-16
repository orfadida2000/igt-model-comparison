"""Analysis, statistical inference, and visualization of IGT model-fit results."""

from .config import AnalysisConfig
from .inference import (
    build_criterion_inference_table,
    build_model_win_inference_table,
)
from .io import ResultTables, load_result_tables
from .pipeline import AnalysisOutputs, generate_results_analysis
from .tables import (
    build_boundary_summary_table,
    build_model_win_table,
    build_parameter_summary_table,
    build_study_preference_table,
    build_subject_comparison_table,
)
from .validation import validate_result_tables

__all__ = [
    "AnalysisConfig",
    "AnalysisOutputs",
    "ResultTables",
    "build_criterion_inference_table",
    "build_boundary_summary_table",
    "build_model_win_inference_table",
    "build_model_win_table",
    "build_parameter_summary_table",
    "build_study_preference_table",
    "build_subject_comparison_table",
    "generate_results_analysis",
    "load_result_tables",
    "validate_result_tables",
]
