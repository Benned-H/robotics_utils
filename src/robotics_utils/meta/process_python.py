"""Define functions for loading or 'meta-processing' Python code."""

from __future__ import annotations

import inspect
import re
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import FunctionType


def parse_docstring_params(docstring: str) -> dict[str, str]:
    """Extract parameter semantics from a docstring.

    :param docstring: String containing the docstring of a skill function
    :return: Map from parameter names to their semantic descriptions
    """
    param_docs = {}
    param_pattern = r":param\s+(\w+):\s*([^\n]+)"

    for match in re.finditer(param_pattern, docstring):
        param_name = match.group(1)
        description = match.group(2).strip() if match.group(2) else ""
        param_docs[param_name] = description

    return param_docs


def load_class_from_module(class_name: str, module_name: str) -> type:
    """Dynamically load a class from the specified module.

    :param class_name: Name of a class to load from a module (e.g., "MyClass")
    :param module_name: String representation of the module (e.g., "my_package.module_name")
    :return: Type of the dynamically loaded class
    """
    loaded_module = import_module(module_name)
    if not hasattr(loaded_module, class_name):
        raise ImportError(f"Cannot load class '{class_name}' from module '{loaded_module}'.")

    return getattr(loaded_module, class_name)


def get_default_values(function: FunctionType) -> dict[str, Any]:
    """Extract any default argument values for the given function.

    Reference: https://stackoverflow.com/a/12627202

    :param function: Python function inspected for default arguments
    :return: Map from input variable names to their corresponding default values
    """
    signature = inspect.signature(function)
    return {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
