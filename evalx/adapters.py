"""Adapters: how the harness calls the app under test.

An adapter is any callable taking a question string and returning an answer
string. Targets are named ``module:attr``, e.g. ``examples.demo_app:answer``.
"""

import importlib
import os
import sys
from collections.abc import Callable


class AdapterError(Exception):
    """Raised when a target cannot be resolved to a callable."""


def resolve_target(target: str) -> Callable[[str], str]:
    """Import ``module:attr`` and return it as a question -> answer callable."""
    if ":" not in target:
        raise AdapterError(f"target must look like 'module:attr', got {target!r}")
    module_name, attr = target.split(":", 1)
    # Targets live in the user's project, so import from the working
    # directory like `python -m` and pytest do. The installed `evalx` script
    # doesn't put it on sys.path by itself.
    cwd = os.getcwd()
    if cwd not in sys.path and "" not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AdapterError(f"cannot import module {module_name!r}: {exc}") from exc
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise AdapterError(f"{target!r} is not callable")
    return fn
