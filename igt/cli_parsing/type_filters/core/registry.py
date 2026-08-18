"""Process-wide registration and resolution of named command-line type filters.

The registry associates concrete `TypeFilterDefinition` subclasses with providers
that construct their argparse-compatible callables, and supports explicit provider
registration and member resolution.
"""

from types import MappingProxyType
from typing import ClassVar, Never

from igt.cli_parsing.type_filters.core.definitions import (
    ProviderTypeFilterRegistryView,
    TypeFilter,
    TypeFilterDefinition,
    TypeFilterProvider,
    _validate_definition_class,
)

type TypeFilterProviderRegistry = dict[TypeFilterProvider, ProviderTypeFilterRegistryView]


class TypeFilterRegistry:
    """Process-wide registry of resolved named type filters.

    Provider classes derived from
    [`TypeFilterDefinition`][igt.cli_parsing.type_filters.core.definitions.TypeFilterDefinition]
    are registered once and mapped to immutable views of their resolved member
    callables. The class is a static namespace: it cannot be instantiated or
    subclassed.

    Attributes:
        PROVIDER_REGISTRY: Mutable backing mapping from provider classes to their
            immutable member-to-filter registries.
    """

    PROVIDER_REGISTRY: ClassVar[TypeFilterProviderRegistry] = {}

    def __new__(cls) -> Never:
        """Prevent instantiation of the registry utility class.

        Raises:
            TypeError: Always, because the registry is static process-wide
                state.
        """
        raise TypeError(f"The class {__class__.__name__!r} cannot be instantiated.")

    def __init_subclass__(cls) -> Never:
        """Prevent subclassing of the registry utility class.

        Raises:
            TypeError: Always, because the registry is not an extension point.
        """
        raise TypeError(f"The class {__class__.__name__!r} cannot be subclassed.")

    @staticmethod
    def _validate_provider_class_get_registration_status(
        type_filter_provider: TypeFilterProvider,
    ) -> bool:
        """Validate a provider and report whether it is already registered.

        Args:
            type_filter_provider: Provider class to validate and inspect.

        Returns:
            `True` when the provider is registered consistently; otherwise
            `False`.

        Raises:
            TypeError: If the provider class is invalid.
            RuntimeError: If the stored registry no longer matches the
                provider's enum members.
        """
        _validate_definition_class(type_filter_provider, allow_memberless=False)

        if type_filter_provider not in __class__.PROVIDER_REGISTRY:
            # No need to validate if the provider class is not registered
            return False

        if set(__class__.PROVIDER_REGISTRY[type_filter_provider]) != set(type_filter_provider):
            missing_members = set(type_filter_provider) - set(
                __class__.PROVIDER_REGISTRY[type_filter_provider]
            )
            non_provider_member_values = set(
                __class__.PROVIDER_REGISTRY[type_filter_provider]
            ) - set(type_filter_provider)

            error_message = f"Inconsistency detected in the global registry, The type filters provider class {type_filter_provider.__name__!r} is detected as a registered provider, but the its matching values in the global registry are inconsistent with the members of the provider class. "
            if missing_members:
                error_message += f"The following members are missing from the registry: {tuple(member.name for member in missing_members)}. "
            if non_provider_member_values:
                # We can't assume that the non-provider member values are even Enum members
                error_message += f"The following values are present in the registry but aren't a member of the provider class: {tuple(non_provider_member_values)}. "
            raise RuntimeError(error_message)

        return True

    @staticmethod
    def register_provider(
        type_filter_provider: TypeFilterProvider,
        *,
        update: bool = False,
    ) -> None:
        """Register or refresh one type-filter provider.

        Args:
            type_filter_provider: Provider class whose members should be
                resolved and stored.
            update: Whether to rebuild an already registered provider.

        Raises:
            TypeError: If the provider class is invalid.
            RuntimeError: If an existing registration is inconsistent.
        """
        is_provider_registered = __class__._validate_provider_class_get_registration_status(
            type_filter_provider
        )

        if is_provider_registered and not update:
            return  # No-op, the provider is already registered and updates are not allowed

        provider_registry = type_filter_provider.get_type_filter_registry(validate_members=False)

        __class__.PROVIDER_REGISTRY[type_filter_provider] = MappingProxyType(
            dict(provider_registry)
        )

    @staticmethod
    def unregister_provider(type_filter_provider: TypeFilterProvider) -> None:
        """Remove one provider from the global registry if present.

        Args:
            type_filter_provider: Provider class to unregister.

        Raises:
            TypeError: If the provider class is invalid.
            RuntimeError: If an existing registration is inconsistent.
        """
        is_provider_registered = __class__._validate_provider_class_get_registration_status(
            type_filter_provider
        )

        if not is_provider_registered:
            return  # No-op, the provider is not registered

        del __class__.PROVIDER_REGISTRY[type_filter_provider]

    @staticmethod
    def resolve_type_filter(
        type_filter_member: TypeFilterDefinition,
    ) -> TypeFilter:
        """Resolve a registered definition member to its type-filter callable.

        Args:
            type_filter_member: Named member to resolve.

        Returns:
            The callable registered for the member.

        Raises:
            TypeError: If the argument is not a type-filter definition member.
            KeyError: If the member's provider class is not registered.
            RuntimeError: If the provider is registered but the requested
                member is missing from its registry.
        """
        if not isinstance(type_filter_member, TypeFilterDefinition):
            raise TypeError(
                f"The 'type_filter_member' argument must be an instance of {TypeFilterDefinition.__name__!r}, got {type(type_filter_member).__name__!r}."
            )

        provider_cls = type(type_filter_member)

        if provider_cls not in __class__.PROVIDER_REGISTRY:
            raise KeyError(
                f"The named type filter member {type_filter_member.name!r} of the provider class {provider_cls.__name__!r} could not be found in the global registry."
            )

        if type_filter_member not in __class__.PROVIDER_REGISTRY[provider_cls]:
            raise RuntimeError(
                f"The named type filter member {type_filter_member.name!r} of the provider class {provider_cls.__name__!r} is not registered in the global registry but the provider class is marked as registered. This should not happen as it indicates an inconsistency in the global registry."
            )

        return __class__.PROVIDER_REGISTRY[provider_cls][type_filter_member]
