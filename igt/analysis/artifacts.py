"""Logical artifact containers produced by result analysis.

Each generated figure or table can be persisted in more than one file format. The
containers in this module keep those physical files grouped as one logical analysis
artifact.
"""

from dataclasses import dataclass
from pathlib import Path

from igt.utils.io import normalize_path


def _normalize_output_paths(
    paths: tuple[Path, ...],
    *,
    artifact_name: str,
) -> tuple[Path, ...]:
    """Validate and normalize the physical paths of one generated artifact."""

    if not isinstance(paths, tuple):
        raise TypeError(f"{artifact_name}.paths must be a tuple of pathlib.Path values.")

    if not paths:
        raise ValueError(f"{artifact_name}.paths must contain at least one path.")

    normalized_paths = tuple(
        normalize_path(path, parameter_name=f"{artifact_name}.paths")
        for path in paths
    )

    if len(set(normalized_paths)) != len(normalized_paths):
        raise ValueError(f"{artifact_name}.paths must be unique.")

    return normalized_paths


def _normalize_output_stem(
    output_stem: Path,
    *,
    artifact_name: str,
) -> Path:
    """Normalize one logical output stem and require it to have no suffix."""

    normalized_output_stem = normalize_path(
        output_stem,
        parameter_name=f"{artifact_name}.output_stem",
    )

    if normalized_output_stem.suffix:
        raise ValueError(f"{artifact_name}.output_stem must not have a file suffix.")

    return normalized_output_stem


def _validate_paths_match_stem(
    output_stem: Path,
    paths: tuple[Path, ...],
    *,
    artifact_name: str,
) -> None:
    """Require every physical path to represent one format of the logical stem."""

    for path in paths:
        if not path.suffix:
            raise ValueError(f"Every {artifact_name}.paths entry must have a file suffix.")

        if path.with_suffix("") != output_stem:
            raise ValueError(
                f"Every {artifact_name}.paths entry must match output_stem apart from "
                f"its file suffix. Stem: {output_stem}; path: {path}."
            )


@dataclass(frozen=True, slots=True)
class GeneratedFigure:
    """One logical figure and every file format written for it.

    Attributes:
        output_stem: Figure destination without a file-format suffix.
        paths: Physical figure files, in configured format order.
    """

    output_stem: Path
    paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        """Normalize the logical stem and generated file paths."""

        output_stem = _normalize_output_stem(
            self.output_stem,
            artifact_name="GeneratedFigure",
        )
        paths = _normalize_output_paths(
            self.paths,
            artifact_name="GeneratedFigure",
        )
        _validate_paths_match_stem(
            output_stem,
            paths,
            artifact_name="GeneratedFigure",
        )
        object.__setattr__(self, "output_stem", output_stem)
        object.__setattr__(self, "paths", paths)

    @property
    def name(self) -> str:
        """Return the final component of the logical figure stem."""

        return self.output_stem.name

    @property
    def formats(self) -> tuple[str, ...]:
        """Return generated file formats in physical-path order."""

        return tuple(path.suffix.lstrip(".").lower() for path in self.paths)


@dataclass(frozen=True, slots=True)
class GeneratedTable:
    """One logical table and every file format written for it.

    Attributes:
        output_stem: Table destination without a file-format suffix.
        paths: Physical table files, in generated format order.
    """

    output_stem: Path
    paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        """Normalize the logical stem and generated file paths."""

        output_stem = _normalize_output_stem(
            self.output_stem,
            artifact_name="GeneratedTable",
        )
        paths = _normalize_output_paths(
            self.paths,
            artifact_name="GeneratedTable",
        )
        _validate_paths_match_stem(
            output_stem,
            paths,
            artifact_name="GeneratedTable",
        )
        object.__setattr__(self, "output_stem", output_stem)
        object.__setattr__(self, "paths", paths)

    @property
    def name(self) -> str:
        """Return the final component of the logical table stem."""

        return self.output_stem.name

    @property
    def formats(self) -> tuple[str, ...]:
        """Return generated file formats in physical-path order."""

        return tuple(path.suffix.lstrip(".").lower() for path in self.paths)
