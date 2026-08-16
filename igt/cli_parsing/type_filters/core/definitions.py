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
    """
    Base class for named type filter chains to be used in command line parsing.

    A subclass of this class must not override the `__eq__` or `__hash__` methods.

    Each subclass of this class must define at least one named type filter chain member, where each member's value must be either a callable or a non-empty tuple of callables,
    where for members with a single callable value, their value is treated as a 1 element type filters chain.
    Each member represents an equivalent type filter callable, which receives a string argument and chains the callables in the tuple together.
    Its string argument being the input to the first callable in the tuple, the output of each callable besides the last one being the input to the next callable in the tuple, and the output of the last callable being the output of the equivalent type filter itself.

    The equivalent type filter callable is meant to be used as a type filter for an argument in a command line parser, therefore it's the responsibility of programmer to ensure that for each member,its value will be equivalent to a callable that can be used as such
    (i.e. it must be able to accept a single positional string argument and return a value of the desired type, or raise an exception for invalid input where the exception's type is appropriate for command line parsing).
    """

    def __init_subclass__(cls, *, _allow_memberless: bool = True, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        _validate_definition_class(cls, allow_memberless=_allow_memberless)

    def get_type_filter(
        self,
        *,
        is_validated: bool = False,
    ) -> TypeFilter:
        raise NotImplementedError(
            f"The method 'get_type_filter' must be implemented in a non-concrete subclass of {__class__.__name__!r}."
        )

    @classmethod
    def get_type_filter_registry(
        cls,
        *,
        validate_members: bool = False,
    ) -> ProviderTypeFilterRegistry:
        return {member: member.get_type_filter(is_validated=not validate_members) for member in cls}


class TypeFilterChainDefinition(TypeFilterDefinition):
    """
    Base class for named type filter chains to be used in command line parsing.

    A subclass of this class must not override the `__eq__` or `__hash__` methods.

    Each subclass of this class must define at least one named type filter chain member, where each member's value must be either a callable or a non-empty tuple of callables,
    where for members with a single callable value, their value is treated as a 1 element type filters chain.
    Each member represents an equivalent type filter callable, which receives a string argument and chains the callables in the tuple together.
    Its string argument being the input to the first callable in the tuple, the output of each callable besides the last one being the input to the next callable in the tuple, and the output of the last callable being the output of the equivalent type filter itself.

    The equivalent type filter callable is meant to be used as a type filter for an argument in a command line parser, therefore it's the responsibility of programmer to ensure that for each member,its value will be equivalent to a callable that can be used as such
    (i.e. it must be able to accept a single positional string argument and return a value of the desired type, or raise an exception for invalid input where the exception's type is appropriate for command line parsing).
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if "_allow_memberless" in kwargs:
            raise TypeError(
                f"The '_allow_memberless' keyword argument cannot be used when subclassing {__class__.__name__!r} as it is always set to False."
            )

        super().__init_subclass__(_allow_memberless=False, **kwargs)

    def get_type_filter(self, *, is_validated: bool = False) -> TypeFilter:
        if not is_validated:
            tuple_definition = _validate_definition_value(self)
        else:
            tuple_definition = cast(
                TypeFilterTupleDefinition,
                self.value if isinstance(self.value, tuple) else (self.value,),
            )

        def chained_type_filter(s: str) -> Any:
            current_value = s

            for _callable in tuple_definition:
                current_value = _callable(current_value)

            return current_value

        return chained_type_filter


class TypeFilterChoiceDefinition(TypeFilterDefinition):
    """
    Base class for named type filter chains to be used in command line parsing.

    A subclass of this class must not override the `__eq__` or `__hash__` methods.

    Each subclass of this class must define at least one named type filter chain member, where each member's value must be either a callable or a non-empty tuple of callables,
    where for members with a single callable value, their value is treated as a 1 element type filters chain.
    Each member represents an equivalent type filter callable, which receives a string argument and chains the callables in the tuple together.
    Its string argument being the input to the first callable in the tuple, the output of each callable besides the last one being the input to the next callable in the tuple, and the output of the last callable being the output of the equivalent type filter itself.

    The equivalent type filter callable is meant to be used as a type filter for an argument in a command line parser, therefore it's the responsibility of programmer to ensure that for each member,its value will be equivalent to a callable that can be used as such
    (i.e. it must be able to accept a single positional string argument and return a value of the desired type, or raise an exception for invalid input where the exception's type is appropriate for command line parsing).
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        if "_allow_memberless" in kwargs:
            raise TypeError(
                f"The '_allow_memberless' keyword argument cannot be used when subclassing {__class__.__name__!r} as it is always set to False."
            )

        super().__init_subclass__(_allow_memberless=False, **kwargs)

    def get_type_filter(self, *, is_validated: bool = False) -> TypeFilter:
        if not is_validated:
            tuple_definition = _validate_definition_value(self)
        else:
            tuple_definition = cast(
                TypeFilterTupleDefinition,
                self.value if isinstance(self.value, tuple) else (self.value,),
            )

        def _get_callable_name(type_filter: TypeFilter | GenericTypeFilter) -> str:
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
