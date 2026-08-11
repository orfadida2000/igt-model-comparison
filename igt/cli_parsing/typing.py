import argparse
import keyword
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, cast

from igt.cli_parsing.types.numeric import ARG_TYPE_CALLABLE_MAP as NUMERIC_ARG_TYPE_CALLABLE_MAP
from igt.cli_parsing.types.numeric import NumericArgType
from igt.cli_parsing.types.path import ARG_TYPE_CALLABLE_MAP as PATH_ARG_TYPE_CALLABLE_MAP
from igt.cli_parsing.types.path import PathArgType
from igt.cli_parsing.types.string import ARG_TYPE_CALLABLE_MAP as STRING_ARG_TYPE_CALLABLE_MAP
from igt.cli_parsing.types.string import StringArgType
from igt.typing import NonEmptyMixedTuple

NOTSET: Final[object] = object()


class ArgAction(Enum):
    """Enumeration of argument actions for command-line argument parsing."""

    STORE = "store"
    STORE_TRUE = "store_true"
    STORE_FALSE = "store_false"
    STORE_CONST = "store_const"
    APPEND = "append"
    APPEND_CONST = "append_const"
    EXTEND = "extend"
    COUNT = "count"
    HELP = "help"
    VERSION = "version"


type ArgType = StringArgType | NumericArgType | PathArgType


def _is_arg_type(value: Any) -> bool:
    """Check if the given value is an instance of ArgType (StringArgType, NumericArgType, or PathArgType)."""

    return isinstance(value, (StringArgType, NumericArgType, PathArgType))


ARG_TYPE_CALLABLE_MAP: Final[
    Mapping[
        ArgType,
        Callable[[str], Any] | NonEmptyMixedTuple[Callable[[str], Any], Callable[[Any], Any]],
    ]
] = MappingProxyType(
    {
        **STRING_ARG_TYPE_CALLABLE_MAP,
        **NUMERIC_ARG_TYPE_CALLABLE_MAP,
        **PATH_ARG_TYPE_CALLABLE_MAP,
    }
)


@dataclass(kw_only=True, frozen=True)
class ResolvedArgInfo:
    """Resolved specification for a command-line argument."""

    effective_name_or_flags: tuple[str, *tuple[str, ...]] = field(
        metadata={
            "help": "Normalized name of the argument (to be used in the 'add_argument' method)."
        },
    )
    dest_identifier: str = field(
        metadata={
            "help": "Normalized identifier name of the argument which will be used in the parsed Namespace object.",
        },
    )
    is_positional: bool = field(
        metadata={
            "help": "Whether the argument is a positional argument (doesn't have leading hyphen).",
        },
    )


@dataclass(kw_only=True, frozen=True)
class ArgSpec:
    """Specification for a command-line argument."""

    _USED_ADD_ARG_PARAMS: ClassVar[tuple[str, ...]] = (
        "action",
        "default",
        "type",
        "choices",
        "required",
        "help",
    )

    name_or_flags: str | Sequence[str] = field(
        metadata={"help": "Name of the argument (used for the command-line parser)."},
    )
    _effective_name_or_flags: tuple[str] | None = field(
        init=False,
        default=None,
        metadata={
            "help": "Normalized name of the argument (to be used in the 'add_argument' method).",
        },
    )
    _dest_identifier: str | None = field(
        init=False,
        default=None,
        metadata={
            "help": "Normalized identifier name of the argument which will be used in the parsed Namespace object.",
        },
    )
    action: ArgAction = field(
        default=ArgAction.STORE,
        metadata={
            "help": "Action to be taken when the argument is encountered.",
        },
    )
    type_filters: (
        ArgType | Callable[[str], Any] | Sequence[ArgType | Callable[[Any], Any]] | None
    ) = field(
        default=None,
        metadata={
            "help": "Type filter(s) for the argument (used for validation and conversion). Can be an ArgType, a callable, or a sequence of ArgTypes and/or callables. (will be normalized to a sequence of callables for validation and conversion).",
        },
    )
    _argparse_type: Callable[[str], Any] | None = field(
        init=False,
        default=None,
        metadata={
            "help": "Normalized type callable of the argument (used for validation and conversion) for argparse.",
        },
    )
    required: bool | None = field(
        default=None,
        metadata={
            "help": "Whether the argument is required.",
        },
    )
    default: Any = field(
        default=NOTSET,
        metadata={
            "help": "Default value for the argument.",
        },
    )
    choices: Iterable[Any] | None = field(
        default=None,
        metadata={
            "help": "Valid choices for the argument.",
        },
    )
    help: str | None = field(
        default=None,
        metadata={
            "help": "Help text for the argument.",
        },
    )
    extra_options: Mapping[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Additional options for the argument (passed to 'add_argument').",
        },
    )

    def __post_init__(self) -> None:
        name_or_flags = self.name_or_flags

        if not isinstance(name_or_flags, (str, Sequence)):
            raise TypeError(
                f"'name_or_flags' must be a str or a sequence of str, got {type(name_or_flags).__name__}"
            )

        if not isinstance(name_or_flags, str):
            name_or_flags = tuple(name_or_flags)

            if len(name_or_flags) == 0:
                raise ValueError("'name_or_flags' sequence must not be empty")

            for name in name_or_flags:
                if not isinstance(name, str):
                    raise TypeError(
                        f"Each element of 'name_or_flags' sequence must be a str, got {type(name).__name__}"
                    )

        object.__setattr__(self, "name_or_flags", name_or_flags)

        if not isinstance(self.extra_options, Mapping):
            raise TypeError(
                f"'extra_options' must be a Mapping, got {type(self.extra_options).__name__}"
            )

        extra_options = dict(self.extra_options)

        for key in extra_options:
            if not isinstance(key, str):
                raise TypeError(
                    f"Each key in 'extra_options' must be a str, got {type(key).__name__}"
                )

        self._clean_extra_options(strict=False)

        if not isinstance(self.action, ArgAction):
            raise TypeError(f"'action' must be an ArgAction, got {type(self.action).__name__}")

        is_store_append_extend_action = self._has_store_append_extend_action()
        is_version_help_action = self._has_version_help_action()

        if self.action is ArgAction.VERSION and "version" not in self.extra_options:
            raise ValueError(
                "'version' must be specified in 'extra_options' when 'action' is ArgAction.VERSION"
            )

        if is_version_help_action and self.default is not NOTSET:
            raise ValueError(
                f"'default' must not be specified when 'action' is {self.action.value}"
            )

        if self.type_filters is not None:
            if not is_store_append_extend_action:
                raise ValueError(
                    f"'type_filters' must not be specified (not None) when 'action' isn't one of {(ArgAction.STORE.value, ArgAction.APPEND.value, ArgAction.EXTEND.value)}, got {self.action.value}"
                )

            if (
                not _is_arg_type(self.type_filters)
                and not isinstance(self.type_filters, Sequence)
                and not callable(self.type_filters)
            ):
                raise TypeError(
                    f"'type_filters' must be an ArgType, a callable, a non-empty sequence of ArgTypes and/or callables, or None, got {type(self.type_filters).__name__}"
                )

            if isinstance(self.type_filters, Sequence):
                if len(self.type_filters) == 0:
                    raise ValueError("'type_filters' sequence must not be empty")

                type_filters = []
                for filter in self.type_filters:
                    if not _is_arg_type(filter) and not callable(filter):
                        raise TypeError(
                            f"Each element of 'type_filters' sequence must be an ArgType or a callable, got {type(filter).__name__}"
                        )
                    type_filters.append(filter)

                type_filters = tuple(type_filters)
            else:
                type_filters = (self.type_filters,)

            type_filters_tup: NonEmptyMixedTuple[
                ArgType | Callable[[str], Any], ArgType | Callable[[Any], Any]
            ] = type_filters
            object.__setattr__(self, "type_filters", type_filters_tup)

        if self.required is not None and not isinstance(self.required, bool):
            raise TypeError(
                f"'required' must be a bool or None, got {type(self.required).__name__}"
            )

        if is_version_help_action:
            if self.required is True:
                # Accepts explicit False even though argparse doesn't allow giving required at all for version/help actions (will be omitted when adding to parser).
                raise ValueError(
                    f"'required' must not be True when 'action' is {self.action.value}"
                )

            object.__setattr__(
                self, "required", None
            )  # Set to None to omit it when adding to the parser.

        if self.choices is not None and not isinstance(self.choices, Iterable):
            raise TypeError(
                f"'choices' must be an Iterable or None, got {type(self.choices).__name__}"
            )

        choices = tuple(self.choices) if self.choices is not None else None
        if choices is not None and len(choices) == 0:
            raise ValueError("'choices' must not be an empty sequence")
        if choices is not None and not is_store_append_extend_action:
            raise ValueError(
                f"'choices' must not be specified when 'action' is {self.action.value}"
            )
        if choices is not None and self.default is not NOTSET and self.default not in choices:
            raise ValueError(
                f"'default' value {self.default} is not in the specified 'choices' values: {choices}"
            )
        object.__setattr__(self, "choices", choices)

        if self.help is not None and not isinstance(self.help, str):
            raise TypeError(f"'help' must be a str or None, got {type(self.help).__name__}")

        if is_store_append_extend_action:
            argparse_type = self._create_aggregated_argparse_type_filter()
            object.__setattr__(self, "_argparse_type", argparse_type)

    def _create_aggregated_argparse_type_filter(self) -> Callable[[str], Any]:
        """Final type callable for argparse, applying all type filters in order."""

        if (
            self.type_filters is None
            or not isinstance(self.type_filters, tuple)
            or len(self.type_filters) == 0
        ):
            raise RuntimeError(
                f"'type_filters' is not a non-empty tuple; This should not happen under normal circumstances unless the ArgSpec instance was not used in the intended way. Current value: {self.type_filters!r}"
            )

        type_filters = cast(
            NonEmptyMixedTuple[ArgType | Callable[[str], Any], ArgType | Callable[[Any], Any]],
            self.type_filters,
        )

        normalized_type_filters: list[Callable[[Any], Any]] = []

        for filter in type_filters:
            if _is_arg_type(filter):
                filter = cast(ArgType, filter)

                if filter not in ARG_TYPE_CALLABLE_MAP:
                    raise ValueError(f"Unsupported ArgType: {filter}")

                callable_filter = ARG_TYPE_CALLABLE_MAP[filter]
                if isinstance(callable_filter, tuple):
                    if len(callable_filter) == 0:
                        raise ValueError(
                            f"Callable filter for ArgType {filter} is an empty tuple; This should not happen under normal circumstances unless the `ARG_TYPE_CALLABLE_MAP` was modified after initialization."
                        )

                    callable_filters = cast(
                        NonEmptyMixedTuple[Callable[[str], Any], Callable[[Any], Any]],
                        callable_filter,
                    )
                else:
                    callable_filters = (callable_filter,)

                for f in callable_filters:
                    if not callable(f):
                        raise TypeError(
                            f"Each element of the callable filters for ArgType {filter} must be a callable, got {type(f).__name__}"
                        )

                normalized_type_filters.extend(callable_filters)
            elif callable(filter):
                normalized_type_filters.append(filter)
            else:
                raise TypeError(
                    f"Each element of 'type_filters' must be an ArgType or a callable, got {type(filter).__name__}"
                )

        def _aggregated_argparse_type(value: str) -> Any:
            current_value: Any = value

            for filter in normalized_type_filters:
                current_value = filter(current_value)

            return current_value

        return _aggregated_argparse_type

    def _resolve_effective_name_or_flags_and_dest_identifier(
        self, prefix_chars: str
    ) -> ResolvedArgInfo:

        if not isinstance(prefix_chars, str):
            raise TypeError(f"'prefix_chars' must be a str, got {type(prefix_chars).__name__}")

        if not prefix_chars:
            raise ValueError("'prefix_chars' must not be empty")

        is_store_append_extend_action = self._has_store_append_extend_action()

        match_whitespace = re.search(r"\s", prefix_chars)

        if match_whitespace:
            raise ValueError(
                f"'prefix_chars' must not contain whitespace characters, found {match_whitespace.group()!r} at index {match_whitespace.start()}"
            )

        names = (self.name_or_flags,) if isinstance(self.name_or_flags, str) else self.name_or_flags

        if len(names) == 0:
            raise ValueError("'name_or_flags' sequence must not be empty")

        names = [name.lstrip() for name in names]
        is_positional_arg = False
        non_positional_prefixes: list[str] = []
        clean_names: list[str] = []

        for name in names:
            non_positional_prefix = self._get_non_positional_prefix(name, prefix_chars)

            if non_positional_prefix is None:
                if len(names) > 1:
                    raise ValueError(
                        f"'name_or_flags' must be either a single positional argument, single non-positional argument, or a sequence of non-positional arguments (all starting with one of the prefix_chars={tuple(prefix_chars)!r}), got {self.name_or_flags!r}"
                    )

                is_positional_arg = True
                clean_name = name.rstrip().replace(" ", "_")
            else:
                non_positional_prefixes.append(non_positional_prefix)
                clean_name = name[len(non_positional_prefix) :].rstrip().replace(" ", "_")

            clean_names.append(clean_name)

        if is_positional_arg:
            clean_positional_name = clean_names[0]

            if self.required is False:
                # Accepts explicit True even though argparse doesn't allow giving required at all for positional arguments (will be omitted when adding to parser).
                raise ValueError(
                    f"'required' must not be False when 'name_or_flags' is a positional argument (doesn't have leading prefix_chars={tuple(prefix_chars)!r})"
                )

            object.__setattr__(
                self, "required", None
            )  # Set to None to omit it when adding to the parser.

            if not is_store_append_extend_action:
                raise ValueError(
                    f"'action' must be one of {(ArgAction.STORE.value, ArgAction.APPEND.value, ArgAction.EXTEND.value)} when 'name_or_flags' is a positional argument (doesn't have leading prefix_chars={tuple(prefix_chars)!r}), got {self.action.value}"
                )

            if "dest" in self.extra_options:
                raise ValueError(
                    f"'dest' must not be specified in 'extra_options' when 'name_or_flags' is a positional argument (doesn't have leading prefix_chars={tuple(prefix_chars)!r}), got dest={self.extra_options['dest']}"
                )

            # normalize clean_name from hyphens to underscores for parser_name so an argument like 'foo-bar' becomes 'foo_bar' in the parser (a valid Python identifier), which is the default behavior in argparse for non-positional arguments (not the case for positional arguments)
            effective_name: str = clean_positional_name.replace("-", "_")
            effective_name_or_flags = [effective_name]
            dest_identifier = effective_name
        else:
            if "dest" in self.extra_options:
                dest = self.extra_options["dest"]

                self._validate_python_identifier(dest, label="extra_options['dest']")

                dest = cast(str, dest)
                dest_identifier = dest
            else:
                dest_identifier = None

            effective_flags: list[str] = []

            for non_positional_prefix, clean_name in zip(
                non_positional_prefixes, clean_names, strict=True
            ):
                # normalize clean_name from underscores to hyphens for parser_name because argparse already normalizes hyphens to underscores for the identifier used in the namespace, and using hyphens with non-positional arguments is more common in CLI conventions (e.g., --foo-bar)
                effective_flag = f"{non_positional_prefix}{clean_name.replace('_', '-')}"

                if dest_identifier is not None and len(non_positional_prefix) > 1:
                    dest_identifier = effective_flag[len(non_positional_prefix) :].replace("-", "_")

                effective_flags.append(effective_flag)

            if dest_identifier is None:
                dest_identifier = effective_flags[0][len(non_positional_prefixes[0]) :].replace(
                    "-", "_"
                )

            effective_name_or_flags = effective_flags

        self._validate_python_identifier(dest_identifier, label="namespace destination")

        effective_name_or_flags = tuple(effective_name_or_flags)
        assert len(effective_name_or_flags) > 0  # for type checker

        object.__setattr__(self, "_effective_name_or_flags", effective_name_or_flags)
        object.__setattr__(self, "_dest_identifier", dest_identifier)

        return ResolvedArgInfo(
            effective_name_or_flags=effective_name_or_flags,
            dest_identifier=dest_identifier,
            is_positional=is_positional_arg,
        )

    def _clear_effective_name_or_flags_and_dest_identifier(self) -> None:
        """Clear the internal fields `_effective_name_or_flags` and `_dest_identifier`."""

        object.__setattr__(self, "_effective_name_or_flags", None)
        object.__setattr__(self, "_dest_identifier", None)

    def _clean_extra_options(self, strict: bool) -> None:

        extra_options = dict(self.extra_options)

        for param in self._USED_ADD_ARG_PARAMS:
            if param in extra_options:
                if strict:
                    raise ValueError(
                        f"'extra_options' must not contain a key that is already set as a separate attribute, got '{param}'"
                    )
                else:
                    logging.warning(
                        "%r is specified in 'extra_options', but it is already set as a separate attribute. The value in 'extra_options' will be ignored.",
                        param,
                    )

                    extra_options.pop(param)

        object.__setattr__(self, "extra_options", MappingProxyType(extra_options))

    @staticmethod
    def _validate_python_identifier(
        identifier: str,
        *,
        label: str | None = None,
    ) -> None:
        """Validate if the given name is a valid Python identifier and not a keyword."""

        if label is not None and not isinstance(label, str):
            raise TypeError(f"'label' must be a str or None, got {type(label).__name__}")

        if not isinstance(identifier, str):
            raise TypeError(f"'identifier' must be a str, got {type(identifier).__name__}")

        if not identifier.isidentifier():
            if label is None:
                raise ValueError(f"{identifier!r} is not a valid Python identifier")

            raise ValueError(f"{label} must be a valid Python identifier, got {identifier!r}")

        if keyword.iskeyword(identifier):
            if label is None:
                raise ValueError(f"{identifier!r} is a Python keyword")

            raise ValueError(f"{label} must not be a Python keyword, got {identifier!r}")

    @staticmethod
    def _get_non_positional_prefix(name: str, prefix_chars: str = "-") -> str | None:
        """Get the prefix characters for this argument if it is a non-positional argument (has leading hyphen), otherwise return None."""

        if not isinstance(prefix_chars, str):
            raise TypeError(f"'prefix_chars' must be a str, got {type(prefix_chars).__name__}")

        if not prefix_chars:
            raise ValueError("'prefix_chars' must not be empty")

        pattern = r"^[" + re.escape(prefix_chars) + r"]+"

        match = re.match(pattern, name)

        if match:
            return match.group()

        return None

    def is_positional(self, prefix_chars: str = "-") -> bool:
        """Check if this argument is a positional argument (doesn't have leading hyphen)."""

        effective_name_or_flags = self.get_effective_name_or_flags()

        # The first element of effective_name_or_flags will be used to determine if it's positional or not,
        # because if it's a positional argument, then there will only be one element in the tuple and if it's a non-positional argument, then all elements will be non-positional (starting with one of the prefix_chars).
        if self._get_non_positional_prefix(effective_name_or_flags[0], prefix_chars) is None:
            return True

        return False

    def _has_store_append_extend_action(self) -> bool:
        """Check if this argument has an action that stores, appends, or extends values."""

        return (
            self.action is ArgAction.STORE
            or self.action is ArgAction.APPEND
            or self.action is ArgAction.EXTEND
        )

    def _has_version_help_action(self) -> bool:
        """Check if this argument has an action that is version or help."""

        return self.action is ArgAction.VERSION or self.action is ArgAction.HELP

    def add_to_parser(self, parser: argparse.ArgumentParser) -> ResolvedArgInfo:
        """Add this argument specification to the given argument parser."""

        self._clear_effective_name_or_flags_and_dest_identifier()
        self._clean_extra_options(strict=True)

        is_store_append_extend_action = self._has_store_append_extend_action()

        is_version_help_action = self._has_version_help_action()

        try:
            # This also validates that if it's a positional argument, then action is STORE, APPEND, or EXTEND
            resolved_info = self._resolve_effective_name_or_flags_and_dest_identifier(
                parser.prefix_chars
            )

            effective_name_or_flags = resolved_info.effective_name_or_flags
            is_positional_arg = resolved_info.is_positional

            if is_version_help_action:
                # No need to check if positional, because if it is positional, then action must be STORE, APPEND, or EXTEND (already validated in `_resolve_parser_and_identifier_names`)
                parser.add_argument(
                    *effective_name_or_flags,
                    action=self.action.value,
                    help=self.help,
                    **self.extra_options,
                )
            elif is_store_append_extend_action:
                assert self._argparse_type is not None  # for type checker

                if is_positional_arg:
                    # 'required' must not be specified for positional arguments (already validated to be `True` in `_resolve_parser_and_identifier_names` as well)
                    if self.default is NOTSET:
                        parser.add_argument(
                            *effective_name_or_flags,
                            action=self.action.value,
                            type=self._argparse_type,
                            choices=self.choices,
                            help=self.help,
                            **self.extra_options,
                        )
                    else:
                        parser.add_argument(
                            *effective_name_or_flags,
                            action=self.action.value,
                            type=self._argparse_type,
                            default=self.default,
                            choices=self.choices,
                            help=self.help,
                            **self.extra_options,
                        )
                else:
                    def_req_opts = {
                        "default": self.default,
                        "required": NOTSET if self.required is None else self.required,
                    }
                    def_req_opts = {k: v for k, v in def_req_opts.items() if v is not NOTSET}

                    parser.add_argument(
                        *effective_name_or_flags,
                        action=self.action.value,
                        type=self._argparse_type,
                        **def_req_opts,
                        choices=self.choices,
                        help=self.help,
                        **self.extra_options,
                    )
            else:
                # No need to check if positional, because if it is positional, then action must be STORE, APPEND, or EXTEND (already validated in `_resolve_parser_and_identifier_names`)

                def_req_opts = {
                    "default": self.default,
                    "required": NOTSET if self.required is None else self.required,
                }
                def_req_opts = {k: v for k, v in def_req_opts.items() if v is not NOTSET}

                parser.add_argument(
                    *effective_name_or_flags,
                    action=self.action.value,
                    **def_req_opts,
                    help=self.help,
                    **self.extra_options,
                )
        finally:
            self._clear_effective_name_or_flags_and_dest_identifier()

        return resolved_info

    def get_effective_name_or_flags(self) -> tuple[str]:
        """Get the effective name or flags for this argument (to be used in the 'add_argument' method)."""

        if self._effective_name_or_flags is None:
            raise ValueError(
                "Effective name or flags are not set, call 'add_to_parser' first to set them"
            )

        if not isinstance(self._effective_name_or_flags, tuple):
            raise TypeError(
                f"Effective name or flags must be a tuple, got {type(self._effective_name_or_flags).__name__!r}; This should not happen under normal circumstances unless the ArgSpec instance was manually modified after initialization."
            )

        if len(self._effective_name_or_flags) == 0:
            raise RuntimeError(
                "Effective name or flags is an empty tuple; This should not happen under normal circumstances unless the ArgSpec instance was manually modified after initialization."
            )

        for value in self._effective_name_or_flags:
            if not isinstance(value, str):
                raise TypeError(
                    f"Each element of effective name or flags must be a str, got {type(value).__name__!r}; This should not happen under normal circumstances unless the ArgSpec instance was manually modified after initialization."
                )

        return self._effective_name_or_flags

    def get_namespace_identifier(self) -> str:
        """Get the identifier name for this argument in the parsed Namespace."""

        if self._dest_identifier is None:
            raise ValueError(
                "Namespace identifier is not set, call 'add_to_parser' first to set it"
            )

        return self._dest_identifier
