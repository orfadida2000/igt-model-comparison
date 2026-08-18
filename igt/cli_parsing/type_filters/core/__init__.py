"""Core abstractions for named command-line type filters.

The core package defines enum-backed filter definitions and the process-wide
[`TypeFilterRegistry`][igt.cli_parsing.type_filters.core.registry.TypeFilterRegistry]
used to resolve those definitions into callables accepted by argparse.
"""

