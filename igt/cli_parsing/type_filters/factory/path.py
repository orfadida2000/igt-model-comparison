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
    """
    Returns a tuple of type filters for file paths with the given extension.

    Args:
        extensions: The file extensions to filter by (e.g., ['.csv', '.json']).

    Returns:
        A tuple containing the type filters for file paths with the specified suffix.
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
    """
    Returns a tuple of type filters for existing file paths with the given extension.

    Args:
        extensions: The file extensions to filter by (e.g., ['.csv', '.json']).

    Returns:
        A tuple containing the type filters for existing file paths with the specified suffix.
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
