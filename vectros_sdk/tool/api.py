from typing import Any, Dict, List, Optional

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request
from vectros_sdk.tool.models import ToolQuery, ToolResponse


def _parse_tool_response(raw_resp: Dict[str, Any]) -> ToolResponse:
    """Parse raw response dictionary from kernel into ToolResponse object."""
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
    Dispatches tool execution requests to the AIOS kernel.

    Args:
        agent_name: Identifier for the agent making the request.
        tool_calls: List of tool call specifications to execute.
        base_url: API endpoint URL (default: configured AIOS kernel URL).

    Returns:
        ToolResponse: Response object containing execution results from the kernel.
    """
    query = ToolQuery(
        agent_name=agent_name,
        tool_calls=tool_calls,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_tool_response(raw_response)

