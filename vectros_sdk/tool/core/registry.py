"""Tool registry for registering and discovering local tools."""
from typing import Any, Dict, Optional, Type

TOOL_REGISTRY: Dict[str, Type[Any]] = {}


def register_tool(name: str, tool_class: Type[Any]) -> None:
    """Register a tool class in the local registry."""
    TOOL_REGISTRY[name] = tool_class


def get_tool(name: str) -> Optional[Type[Any]]:
    """Retrieve a tool class from the local registry."""
    return TOOL_REGISTRY.get(name)


def list_registered_tools() -> Dict[str, Type[Any]]:
    """Return a copy of all registered tool classes in the registry."""
    return dict(TOOL_REGISTRY)


def clear_registry() -> None:
    """Clear all registered tools from the registry."""
    TOOL_REGISTRY.clear()
