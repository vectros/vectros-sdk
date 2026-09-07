from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Abstract base class for developing custom AIOS tools."""

    @abstractmethod
    def get_tool_call_format(self) -> Dict[str, Any]:
        """
        Defines the schema and documentation for how LLMs should interact with this tool.

        Returns:
            Dict[str, Any]: Function specification schema in OpenAI/AIOS tool call format.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, params: Dict[str, Any]) -> Any:
        """
        Execute tool functionality given parameters.

        Args:
            params: Dictionary of input parameters.

        Returns:
            Any: Result of the tool execution.
        """
        raise NotImplementedError

