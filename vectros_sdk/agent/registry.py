"""
Local Agent Registry module for registering, querying, and discovering agent classes.
"""

from typing import Any, Dict, Optional, Type

AGENT_REGISTRY: Dict[str, Type[Any]] = {}
"""Dict[str, Type[Any]]: In-memory mapping of agent names to their implementation classes."""


def register_agent(name: str, agent_class: Type[Any]) -> None:
    """
    Register an agent class in the local in-memory registry.

    Args:
        name (str): Unique name identifier for the agent.
        agent_class (Type[Any]): The agent class implementing the BaseAgent interface.

    Example:
        >>> from vectros_sdk.agent.registry import register_agent
        >>> class MathBot: pass
        >>> register_agent("math_bot", MathBot)
    """
    AGENT_REGISTRY[name] = agent_class


def get_agent(name: str) -> Optional[Type[Any]]:
    """
    Retrieve an agent class by name from the local registry.

    Args:
        name (str): Identifier name of the agent.

    Returns:
        Optional[Type[Any]]: The agent class if registered, otherwise None.

    Example:
        >>> from vectros_sdk.agent.registry import get_agent
        >>> cls = get_agent("math_bot")
    """
    return AGENT_REGISTRY.get(name)


def list_registered_agents() -> Dict[str, Type[Any]]:
    """
    Return a shallow copy of all registered agent classes currently in the registry.

    Returns:
        Dict[str, Type[Any]]: Dictionary mapping agent names to their registered classes.

    Example:
        >>> from vectros_sdk.agent.registry import list_registered_agents
        >>> agents = list_registered_agents()
        >>> print(list(agents.keys()))
    """
    return dict(AGENT_REGISTRY)


def clear_agent_registry() -> None:
    """
    Clear all registered agent classes from the registry.

    Example:
        >>> from vectros_sdk.agent.registry import clear_agent_registry
        >>> clear_agent_registry()
    """
    AGENT_REGISTRY.clear()
