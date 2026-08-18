"""Helpers for building argparse parsers from declarative argument specifications.

The public factory functions create project-standard parsers and add collections of
[`ArgSpec`][igt.cli_parsing.typing.ArgSpec] definitions while centralizing parser
construction error handling.
"""

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

from igt.cli_parsing.typing import ArgSpec, ResolvedArgInfo


def get_parser(
    arg_specs: Sequence[ArgSpec],
    *,
    description: str | None = None,
    extra_options: Mapping[str, Any] | None = None,
) -> tuple[argparse.ArgumentParser, list[ResolvedArgInfo]]:
    """Construct an argument parser from declarative argument specifications.

    Args:
        arg_specs: Argument specifications to add in order.
        description: Optional parser description.
        extra_options: Additional keyword options forwarded to
            `argparse.ArgumentParser`; any `description` entry is ignored in
            favor of the explicit argument.

    Returns:
        The configured parser and resolved metadata for each added argument.

    Raises:
        ValueError: If parser construction or addition of any argument fails.
    """

    resolved_arg_info_list: list[ResolvedArgInfo] = []

    if extra_options is None:
        extra_options = {}

    extra_options = dict(extra_options)
    extra_options.pop("description", None)

    try:
        parser = argparse.ArgumentParser(
            description=description,
            **extra_options,
        )
    except Exception as e:
        raise ValueError(f"Error while creating the ArgumentParser: {e}") from e

    try:
        for arg_spec in arg_specs:
            resolved_info = arg_spec.add_to_parser(parser)
            resolved_arg_info_list.append(resolved_info)
    except Exception as e:
        raise ValueError(f"Error while adding an argument to the parser: {e}") from e

    return parser, resolved_arg_info_list
