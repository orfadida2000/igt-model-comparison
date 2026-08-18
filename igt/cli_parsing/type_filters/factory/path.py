"""Factories for filesystem-path command-line validation.

The helpers normalize path-like input and build validators for existence, file versus
directory expectations, and allowed file extensions. Named path presets are defined
in `igt.cli_parsing.type_filters.presets.path`.
"""

import argparse
from collections.abc import Iterable
from pathlib import Path

from igt.cli_parsing.type_filters.core.definitions import (
    GenericTypeFilter,
    TypeFilter,
)
from igt.cli_parsing.type_filters.presets.path import PathArgTypeProvider
from igt.typing import StrPathLike


def _filter_file_path_by_extension(file_path: StrPathLike, extension_set: set[str]) -> StrPathLike:
    """Require a file path to use one of the allowed extensions.

    Args:
        file_path: Path to validate.
        extension_set: Normalized allowed file extensions.

    Returns:
        The unchanged path when its suffix is allowed.

    Raises:
        ArgumentTypeError: If the file suffix is not in the allowed set.
    """
    file_path = Path(file_path)

    file_extension = file_path.suffix.lower()

    if file_extension not in extension_set:
        raise argparse.ArgumentTypeError(
            f"Invalid file path {file_path!r},the file must have one of the following extensions: {tuple(extension_set)!r}, got {file_extension!r}"
        )

    return file_path


def get_type_filters_for_file_with_extensions_path(
    extensions: str | Iterable[str],
) -> tuple[
    PathArgTypeProvider | TypeFilter,
    *tuple[PathArgTypeProvider | GenericTypeFilter, ...],
]:
    """Build a type-filter chain for a file with an allowed extension.

    Extension strings are stripped, lowercased, and normalized to a leading dot before
    the suffix validator is created. Empty extension entries are ignored.

    Args:
        extensions: One extension string or an iterable of extension strings.

    Returns:
        A chain beginning with the project file-path parser followed by a suffix validator.

    Raises:
        TypeError: If `extensions` is neither a string nor iterable, or an iterable
            element is not a string.
        ValueError: If no nonempty extension remains after normalization.
    """

    if isinstance(extensions, str):
        extensions = (extensions,)
    elif not isinstance(extensions, Iterable):
        raise TypeError(f"Expected an iterable of extensions, got {type(extensions).__name__}")

    extension_lst: list[str] = []

    for ext in extensions:
        if not isinstance(ext, str):
            raise TypeError(f"Expected a string for extension, got {type(ext).__name__}")

        clean_ext = ext.strip().lower().lstrip(".")

        if clean_ext:
            extension_lst.append(f".{clean_ext}")

    if len(extension_lst) == 0:
        raise ValueError("At least one non-empty extension must be provided.")

    extension_set = set(extension_lst)

    return (
        PathArgTypeProvider.FILE_PATH,
        lambda file_path: _filter_file_path_by_extension(file_path, extension_set),
    )


def get_type_filters_for_existing_file_with_extensions_path(
    extensions: str | Iterable[str],
) -> tuple[PathArgTypeProvider | TypeFilter, *tuple[PathArgTypeProvider | GenericTypeFilter, ...]]:
    """Build a type-filter chain for a existing file with an allowed extension.

    Extension strings are stripped, lowercased, and normalized to a leading dot before
    the suffix validator is created. Empty extension entries are ignored.

    Args:
        extensions: One extension string or an iterable of extension strings.

    Returns:
        A chain beginning with the project existing-file path parser followed by a suffix validator.

    Raises:
        TypeError: If `extensions` is neither a string nor iterable, or an iterable
            element is not a string.
        ValueError: If no nonempty extension remains after normalization.
    """

    if isinstance(extensions, str):
        extensions = (extensions,)
    elif not isinstance(extensions, Iterable):
        raise TypeError(f"Expected an iterable of extensions, got {type(extensions).__name__}")

    extension_lst: list[str] = []

    for ext in extensions:
        if not isinstance(ext, str):
            raise TypeError(f"Expected a string for extension, got {type(ext).__name__}")

        clean_ext = ext.strip().lower().lstrip(".")

        if clean_ext:
            extension_lst.append(f".{clean_ext}")

    if len(extension_lst) == 0:
        raise ValueError("At least one non-empty extension must be provided.")

    extension_set = set(extension_lst)

    return (
        PathArgTypeProvider.EXISTING_FILE_PATH,
        lambda file_path: _filter_file_path_by_extension(file_path, extension_set),
    )
