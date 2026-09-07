"""Agent registry for registering and discovering local AIOS agents."""
from typing import Any, Dict, Optional, Type

AGENT_REGISTRY: Dict[str, Type[Any]] = {}


def register_agent(name: str, agent_class: Type[Any]) -> None:
    """Register an agent class in the local registry."""
    AGENT_REGISTRY[name] = agent_class


def get_agent(name: str) -> Optional[Type[Any]]:
    """Retrieve an agent class from the local registry."""
    return AGENT_REGISTRY.get(name)


def list_registered_agents() -> Dict[str, Type[Any]]:
    """Return a copy of all registered agent classes in the registry."""
    return dict(AGENT_REGISTRY)


def clear_agent_registry() -> None:
    """Clear all registered agents from the registry."""
    AGENT_REGISTRY.clear()

