"""
Local Tool Registry module for registering, querying, and managing tools.
"""

from typing import Any, Dict, Optional, Type

TOOL_REGISTRY: Dict[str, Type[Any]] = {}
"""Dict[str, Type[Any]]: In-memory mapping of tool names to their implementation classes."""


def register_tool(name: str, tool_class: Type[Any]) -> None:
    """
    Register a tool class in the local in-memory registry.

    Args:
        name (str): Identifier name for the tool.
        tool_class (Type[Any]): The tool class implementing the BaseTool interface.

    Example:
        >>> from vectros_sdk.tool.core.registry import register_tool
        >>> class Calculator: pass
        >>> register_tool("calculator", Calculator)
    """
    TOOL_REGISTRY[name] = tool_class


def get_tool(name: str) -> Optional[Type[Any]]:
    """
    Retrieve a tool class by name from the local registry.

    Args:
        name (str): The name identifier of the tool to retrieve.

    Returns:
        Optional[Type[Any]]: The tool class if found, otherwise None.

    Example:
        >>> from vectros_sdk.tool.core.registry import get_tool
        >>> cls = get_tool("calculator")
    """
    return TOOL_REGISTRY.get(name)


def list_registered_tools() -> Dict[str, Type[Any]]:
    """
    Return a shallow copy of all registered tool classes currently in the registry.

    Returns:
        Dict[str, Type[Any]]: Dictionary of registered tool names mapped to their classes.

    Example:
        >>> from vectros_sdk.tool.core.registry import list_registered_tools
        >>> all_tools = list_registered_tools()
        >>> print(list(all_tools.keys()))
    """
    return dict(TOOL_REGISTRY)


def clear_registry() -> None:
    """
    Clear all registered tool classes from the registry.

    Example:
        >>> from vectros_sdk.tool.core.registry import clear_registry
        >>> clear_registry()
    """
    TOOL_REGISTRY.clear()
