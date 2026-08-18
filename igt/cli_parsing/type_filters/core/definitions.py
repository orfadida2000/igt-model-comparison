"""Enum-based definitions for named command-line type-filter chains and choices.

`TypeFilterChainDefinition` composes validators sequentially, whereas
`TypeFilterChoiceDefinition` tries alternative parsers until one succeeds. Both are
resolved through the shared type-filter registry.
"""

from __future__ import annotations

from argparse import ArgumentTypeError
from collections.abc import Callable
from enum import Enum
from functools import partial
from types import MappingProxyType
from typing import Any, cast

from igt.typing import NonEmptyMixedTuple

type TypeFilter = Callable[[str], Any]
type GenericTypeFilter = Callable[[Any], Any]

type TypeFilterTupleDefinition = NonEmptyMixedTuple[TypeFilter, GenericTypeFilter]
type TypeFilterDefinitionValue = TypeFilter | TypeFilterTupleDefinition

type TypeFilterProvider = type[TypeFilterDefinition]

type ProviderTypeFilterRegistry = dict[TypeFilterDefinition, TypeFilter]
type ProviderTypeFilterRegistryView = MappingProxyType[TypeFilterDefinition, TypeFilter]


def _validate_definition_value(
    member: TypeFilterDefinition,
) -> TypeFilterTupleDefinition:
    """Validate and normalize one enum member's type-filter definition.

    Args:
        member: Definition member whose value should represent one filter or a
            non-empty chain of filters.

    Returns:
        The normalized non-empty tuple of callables.

    Raises:
        TypeError: If the member value is neither a callable nor a tuple of
            callables, or if any tuple element is not callable.
    """
    if not callable(member.value) and not isinstance(member.value, tuple):
        raise TypeError(
            f"The value of the member {member.name!r} of {member.__class__.__name__!r} is not a callable or a tuple of callables"
        )

    filter_definition_tup = member.value if isinstance(member.value, tuple) else (member.value,)

    if len(filter_definition_tup) == 0:
        raise TypeError(
            f"The value of the member {member.name!r} of {member.__class__.__name__!r} is an empty tuple"
        )

    for i, item in enumerate(filter_definition_tup):
        if not callable(item):
            raise TypeError(
                f"The value of the member {member.name!r} of {member.__class__.__name__!r} has a non-callable item of type {type(item).__name__!r} at index {i} in its tuple"
            )

    return filter_definition_tup


def _validate_definition_class(
    candidate: object,
    *,
    allow_memberless: bool = False,
) -> TypeFilterProvider:
    """Validate a type-filter definition provider class.

    Args:
        candidate: Object expected to be a concrete subclass of
            [TypeFilterDefinition][igt.cli_parsing.type_filters.core.definitions.TypeFilterDefinition].
        allow_memberless: Whether the class may define no enum members.

    Returns:
        The validated provider class.

    Raises:
        TypeError: If the candidate is not a valid provider class, overrides
            equality or hashing, fails to implement `get_type_filter`, has an
            invalid member definition, or is unexpectedly memberless.
    """
    if not isinstance(candidate, type):
        raise TypeError(f"A class must be of type 'type', got {type(candidate).__name__!r}.")

    if not issubclass(candidate, TypeFilterDefinition):
        raise TypeError(
            f"The class {candidate.__name__!r} must be a subclass of {TypeFilterDefinition.__name__!r}."
        )

    if candidate.__eq__ is not TypeFilterDefinition.__eq__:
        raise TypeError(
            f"The definition class {candidate.__name__!r} must not override the `__eq__` method."
        )

    if candidate.__hash__ is not TypeFilterDefinition.__hash__:
        raise TypeError(
            f"The definition class {candidate.__name__!r} must not override the `__hash__` method."
        )

    if candidate.get_type_filter is TypeFilterDefinition.get_type_filter:
        raise TypeError(
            f"The definition class {candidate.__name__!r} must override the `get_type_filter` method."
        )

    if len(candidate) == 0 and not allow_memberless:
        raise TypeError(
            f"The definition class {candidate.__name__!r} must define at least one type filter definition member."
        )

    for member in candidate:
        _validate_definition_value(member)

    return candidate


class TypeFilterDefinition(Enum):
    """Base enum for named command-line type-filter definitions.

    Provider subclasses define enum members whose values are either one
    callable or a non-empty tuple of callables. Concrete subclasses implement
    `get_type_filter` to interpret those callables, for example as a sequential
    chain or as alternative choices.

    Provider classes must preserve the base equality and hashing behavior so
    members remain stable registry keys.
    """

    def __init_subclass__(cls, *, _allow_memberless: bool = True, **kwargs: Any) -> None:
        """Validate every subclass when it is created.

        Args:
            _allow_memberless: Whether the subclass may define no members.
            **kwargs: Additional subclass-creation arguments forwarded to
                `Enum.__init_subclass__`.

        Raises:
            TypeError: If the subclass violates the definition-class
                invariants.
        """
        super().__init_subclass__(**kwargs)

        _validate_definition_class(cls, allow_memberless=_allow_memberless)

    def get_type_filter(
        self,
        *,
        is_validated: bool = False,
    ) -> TypeFilter:
        """Resolve this definition member to an argparse-compatible filter.

        Args:
            is_validated: Whether the member definition has already been
                validated by its provider class.

        Returns:
            A callable that validates and converts one command-line value.

        Raises:
            NotImplementedError: Always for the abstract base definition.
        """
        raise NotImplementedError(
            f"The method 'get_type_filter' must be implemented in a non-concrete subclass of {__class__.__name__!r}."
        )

    @classmethod
    def get_type_filter_registry(
        cls,
        *,
        validate_members: bool = False,
    ) -> ProviderTypeFilterRegistry:
        """Build the type-filter registry for all members of this provider.

        Args:
            validate_members: Whether to revalidate each member while
                resolving its callable.

        Returns:
            A mapping from definition members to their resolved callables.
        """
        return {member: member.get_type_filter(is_validated=not validate_members) for member in cls}


class TypeFilterChainDefinition(TypeFilterDefinition):
    """Definition provider whose member callables are applied as a chain.

    Each enum member resolves to one argparse-compatible callable. The raw
    command-line string enters the first callable, and each intermediate result
    is passed to the next callable.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate creation of a concrete chain-definition provider.

        Args:
            **kwargs: Subclass-creation arguments forwarded to the base
                definition class.

        Raises:
            TypeError: If the reserved `_allow_memberless` option is supplied
                or the provider does not satisfy the required invariants.
        """
        if "_allow_memberless" in kwargs:
            raise TypeError(
                f"The '_allow_memberless' keyword argument cannot be used when subclassing {__class__.__name__!r} as it is always set to False."
            )

        super().__init_subclass__(_allow_memberless=False, **kwargs)

    def get_type_filter(self, *, is_validated: bool = False) -> TypeFilter:
        """Create a filter that applies this member's callables sequentially.

        Args:
            is_validated: Whether the member definition has already been
                validated.

        Returns:
            A callable whose output from each filter becomes the input to the
            next filter in the chain.
        """
        if not is_validated:
            tuple_definition = _validate_definition_value(self)
        else:
            tuple_definition = cast(
                TypeFilterTupleDefinition,
                self.value if isinstance(self.value, tuple) else (self.value,),
            )

        def chained_type_filter(s: str) -> Any:
            """Apply the configured type filters sequentially to one value.

            Args:
                s: Raw command-line value.

            Returns:
                The output produced by the final filter in the chain.
            """
            current_value = s

            for _callable in tuple_definition:
                current_value = _callable(current_value)

            return current_value

        return chained_type_filter


class TypeFilterChoiceDefinition(TypeFilterDefinition):
    """Definition provider whose member callables are alternative parsers.

    Each enum member resolves to one argparse-compatible callable that tries
    its configured alternatives in order and returns the first successful
    conversion. If every alternative fails, their validation messages are
    combined into one `ArgumentTypeError`.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate creation of a concrete choice-definition provider.

        Args:
            **kwargs: Subclass-creation arguments forwarded to the base
                definition class.

        Raises:
            TypeError: If the reserved `_allow_memberless` option is supplied
                or the provider does not satisfy the required invariants.
        """
        if "_allow_memberless" in kwargs:
            raise TypeError(
                f"The '_allow_memberless' keyword argument cannot be used when subclassing {__class__.__name__!r} as it is always set to False."
            )

        super().__init_subclass__(_allow_memberless=False, **kwargs)

    def get_type_filter(self, *, is_validated: bool = False) -> TypeFilter:
        """Build the callable that tries this choice definition's alternative filters.

        Each candidate filter is attempted in declaration order. The first successful
        conversion is returned. If every candidate rejects a command-line value, the
        returned callable combines their validation messages into one
        `argparse.ArgumentTypeError`.

        Args:
            is_validated: Whether registry/member validation has already been completed by
                the caller.

        Returns:
            Argparse-compatible callable implementing the alternative-filter choice.
        """
        if not is_validated:
            tuple_definition = _validate_definition_value(self)
        else:
            tuple_definition = cast(
                TypeFilterTupleDefinition,
                self.value if isinstance(self.value, tuple) else (self.value,),
            )

        def _get_callable_name(type_filter: TypeFilter | GenericTypeFilter) -> str:
            """Return a readable name for a type-filter callable.

            Args:
                type_filter: Callable whose name should appear in an aggregate
                    validation error.

            Returns:
                A qualified function name, partial-function description, or
                callable-class name.
            """
            name = getattr(type_filter, "__qualname__", None)

            if isinstance(name, str):
                return name

            name = getattr(type_filter, "__name__", None)

            if isinstance(name, str):
                return name

            if isinstance(type_filter, partial):
                return f"partial({_get_callable_name(type_filter.func)})"

            return type(type_filter).__qualname__

        def choice_type_filter(value: str) -> Any:
            """Try each alternative type filter until one succeeds.

            Args:
                value: Raw command-line value.

            Returns:
                The converted value from the first successful filter.

            Raises:
                ArgumentTypeError: If every configured alternative rejects the
                    value.
            """
            failures: list[tuple[TypeFilter | GenericTypeFilter, Exception]] = []

            for type_filter in tuple_definition:
                try:
                    return type_filter(value)
                except (ArgumentTypeError, TypeError, ValueError) as e:
                    failures.append((type_filter, e))

            failure_details = "; ".join(
                (
                    f"[{index}] {_get_callable_name(type_filter)}: "
                    f"{type(error).__name__}" + (f": {error}" if str(error) else "")
                )
                for index, (type_filter, error) in enumerate(failures, start=1)
            )

            raise ArgumentTypeError(
                f"The value {value!r} did not match any allowed type filter. "
                f"Failures: {failure_details}"
            )

        return choice_type_filter
