"""
Tool API functional interface for Vectros SDK.

Provides functions for dispatching tool calls directly to the AIOS kernel.
"""

from typing import Any, Dict, List, Optional

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request
from vectros_sdk.tool.models import ToolQuery, ToolResponse


def _parse_tool_response(raw_resp: Dict[str, Any]) -> ToolResponse:
    """
    Parse raw response dictionary from kernel into a typed ToolResponse object.

    Args:
        raw_resp (Dict[str, Any]): Raw JSON response from AIOS kernel.

    Returns:
        ToolResponse: Typed ToolResponse instance with unwrapped fields.
    """
    if isinstance(raw_resp, dict):
        if "response" in raw_resp and isinstance(raw_resp["response"], dict):
            inner = dict(raw_resp["response"])
            if "status_code" not in inner and "status_code" in raw_resp:
                inner["status_code"] = raw_resp["status_code"]
            if "error" not in inner and "error" in raw_resp:
                inner["error"] = raw_resp["error"]
            if "finished" not in inner and "finished" in raw_resp:
                inner["finished"] = raw_resp["finished"]
            return ToolResponse(**inner)
        return ToolResponse(**raw_resp)
    return ToolResponse(response_message=str(raw_resp))


def call_tool(
    agent_name: str,
    tool_calls: List[Dict[str, Any]],
    base_url: str = aios_kernel_url,
) -> ToolResponse:
    """
    Dispatch explicit tool executions to the AIOS kernel for processing.

    Args:
        agent_name (str): Identifier for the agent requesting the tool execution.
        tool_calls (List[Dict[str, Any]]): List of tool invocation specifications containing
            tool name and parameters dictionary.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        ToolResponse: Response object containing tool execution output and status.

    Raises:
        AIOSKernelError: If kernel communication fails or execution encounters an error.

    Example:
        >>> from vectros_sdk.tool.api import call_tool
        >>> resp = call_tool(
        ...     agent_name="weather_agent",
        ...     tool_calls=[{
        ...         "name": "weather_service/get_forecast",
        ...         "parameters": {"location": "New York", "units": "metric"}
        ...     }]
        ... )
        >>> print(resp.response_message)
    """
    query = ToolQuery(
        agent_name=agent_name,
        tool_calls=tool_calls,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_tool_response(raw_response)
