"""
Data models for the Tool API in Vectros SDK.

Defines `ToolQuery` for dispatching function/tool calls to the AIOS kernel and
`ToolResponse` for encapsulating tool execution results.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import ConfigDict, Field

from vectros_sdk.core.models import Query, Response


class ToolQuery(Query):
    """
    Query model for Tool API requests dispatched to the AIOS kernel.

    Attributes:
        query_class (str): Module classification tag, always 'tool'.
        agent_name (Optional[str]): Namespace identifier for the calling agent.
        tool_calls (List[Dict[str, Union[str, Any]]]): List of tool call specifications
            containing tool name and execution arguments.

    Example:
        >>> from vectros_sdk.tool.models import ToolQuery
        >>> q = ToolQuery(
        ...     agent_name="agent_1",
        ...     tool_calls=[{"name": "calc", "parameters": {"expr": "2+2"}}]
        ... )
    """
    query_class: str = Field(default="tool", description="Module category tag.")
    agent_name: Optional[str] = Field(default=None, description="Identifier for the agent calling tools.")
    tool_calls: List[Dict[str, Union[str, Any]]] = Field(..., description="List of tool invocations to run.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class ToolResponse(Response):
    """
    Response model for Tool API operations.

    Attributes:
        response_class (str): Module tag, always 'tool'.
        response_message (Optional[str]): Result string or formatted output from tool execution.
        finished (bool): Whether the tool execution finished successfully.
        error (Optional[str]): Error message if execution encountered an issue.
        status_code (int): HTTP status code from kernel dispatcher.

    Example:
        >>> from vectros_sdk.tool.models import ToolResponse
        >>> resp = ToolResponse(response_message="4", finished=True)
        >>> print(resp["response"]["response_message"])
        '4'
    """
    response_class: str = Field(default="tool", description="Response category tag.")
    response_message: Optional[str] = Field(default=None, description="Tool execution outcome message.")
    finished: bool = Field(default=False, description="Whether the tool execution completed.")
    error: Optional[str] = Field(default=None, description="Error message if execution failed.")
    status_code: int = Field(default=200, description="HTTP response code.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
