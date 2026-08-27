"""Tool registry for registering and discovering local tools."""
from typing import Any, Dict, Optional, Type

TOOL_REGISTRY: Dict[str, Type[Any]] = {}


def register_tool(name: str, tool_class: Type[Any]) -> None:
    """Register a tool class in the local registry."""
    TOOL_REGISTRY[name] = tool_class


def get_tool(name: str) -> Optional[Type[Any]]:
    """Retrieve a tool class from the local registry."""
    return TOOL_REGISTRY.get(name)
