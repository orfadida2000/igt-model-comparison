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
    """"""

    PROVIDER_REGISTRY: ClassVar[TypeFilterProviderRegistry] = {}

    def __new__(cls) -> Never:
        raise TypeError(f"The class {__class__.__name__!r} cannot be instantiated.")

    def __init_subclass__(cls) -> Never:
        raise TypeError(f"The class {__class__.__name__!r} cannot be subclassed.")

    @staticmethod
    def _validate_provider_class_get_registration_status(
        type_filter_provider: TypeFilterProvider,
    ) -> bool:
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
