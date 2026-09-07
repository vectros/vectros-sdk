"""
Abstract Base Tool definition for Vectros SDK tool development.

Defines the `BaseTool` class contract required for building custom tools that
can be registered locally and published to the AIOS Tool Hub.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    Abstract base class for developing custom tools in the AIOS ecosystem.

    Subclasses must implement:
    1. `get_tool_call_format()`: Returning the JSON Schema specification for LLM interaction.
    2. `run(params)`: Containing the core execution logic.

    Example:
        >>> class MyTool(BaseTool):
        ...     def get_tool_call_format(self) -> Dict[str, Any]:
        ...         return {
        ...             "type": "function",
        ...             "function": {
        ...                 "name": "my_tool",
        ...                 "description": "Performs my task",
        ...                 "parameters": {"type": "object", "properties": {}}
        ...             }
        ...         }
        ...     def run(self, params: Dict[str, Any]) -> Any:
        ...         return "Success"
    """

    @abstractmethod
    def get_tool_call_format(self) -> Dict[str, Any]:
        """
        Define the schema and documentation for how language models interact with this tool.

        Returns:
            Dict[str, Any]: Function specification schema in OpenAI/AIOS tool format.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, params: Dict[str, Any]) -> Any:
        """
        Execute the tool functionality with the provided parameter values.

        Args:
            params (Dict[str, Any]): Dictionary of input arguments matching the tool schema.

        Returns:
            Any: Result of the tool execution (string, dictionary, or data object).
        """
        raise NotImplementedError
