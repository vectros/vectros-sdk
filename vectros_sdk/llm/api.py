from typing import Any, Dict, List, Optional

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request
from vectros_sdk.llm.models import LLMQuery, LLMResponse


def _parse_llm_response(raw_resp: Dict[str, Any]) -> LLMResponse:
    """Parse raw response dictionary from kernel into LLMResponse object."""
    if isinstance(raw_resp, dict):
        if "response" in raw_resp and isinstance(raw_resp["response"], dict):
            inner = dict(raw_resp["response"])
            if "status_code" not in inner and "status_code" in raw_resp:
                inner["status_code"] = raw_resp["status_code"]
            if "error" not in inner and "error" in raw_resp:
                inner["error"] = raw_resp["error"]
            return LLMResponse(**inner)
        return LLMResponse(**raw_resp)
    return LLMResponse(response_message=str(raw_resp))


def llm_chat(
    agent_name: str,
    messages: List[Dict[str, Any]],
    base_url: str = aios_kernel_url,
    llms: Optional[List[Dict[str, Any]]] = None,
) -> LLMResponse:
    """
    Basic chat interaction with the language model.

    Args:
        agent_name: Identifier for the agent making the request.
        messages: List of message dictionaries (role, content, etc.).
        base_url: API endpoint URL.
        llms: Optional list of LLM configurations to use.

    Returns:
        LLMResponse: Response object containing the model's text response.
    """
    query = LLMQuery(
        agent_name=agent_name,
        messages=messages,
        action_type="chat",
        message_return_type="text",
        llms=llms,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_llm_response(raw_response)


def llm_chat_with_json_output(
    agent_name: str,
    messages: List[Dict[str, Any]],
    base_url: str = aios_kernel_url,
    llms: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> LLMResponse:
    """
    Get structured JSON response from the language model according to a specified schema.

    Args:
        agent_name: Identifier for the agent making the request.
        messages: List of message dictionaries.
        base_url: API endpoint URL.
        llms: Optional list of LLM configurations.
        response_format: JSON schema specifying required output format.

    Returns:
        LLMResponse: Response object containing structured JSON response.
    """
    query = LLMQuery(
        agent_name=agent_name,
        messages=messages,
        action_type="chat",
        message_return_type="json",
        response_format=response_format,
        llms=llms,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_llm_response(raw_response)


def llm_chat_with_tool_call_output(
    agent_name: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    base_url: str = aios_kernel_url,
    llms: Optional[List[Dict[str, Any]]] = None,
) -> LLMResponse:
    """
    Chat with tool integration allowing LLM to decide which tools to use.

    Args:
        agent_name: Identifier for the agent making the request.
        messages: List of message dictionaries.
        tools: List of available tools and their specifications.
        base_url: API endpoint URL.
        llms: Optional list of LLM configurations.

    Returns:
        LLMResponse: Response object containing tool calls made by the model.
    """
    query = LLMQuery(
        agent_name=agent_name,
        messages=messages,
        tools=tools,
        action_type="tool_use",
        message_return_type="text",
        tool_choice="auto",
        llms=llms,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_llm_response(raw_response)


def llm_call_tool(
    agent_name: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    base_url: str = aios_kernel_url,
    llms: Optional[List[Dict[str, Any]]] = None,
) -> LLMResponse:
    """
    Direct tool invocation instructing the language model to use specified tools.

    Args:
        agent_name: Identifier for the agent making the request.
        messages: List of message dictionaries.
        tools: List of available tools and their specifications.
        base_url: API endpoint URL.
        llms: Optional list of LLM configurations.

    Returns:
        LLMResponse: Response object containing tool calls and results.
    """
    query = LLMQuery(
        agent_name=agent_name,
        messages=messages,
        tools=tools,
        action_type="tool_use",
        message_return_type="text",
        tool_choice="required",
        llms=llms,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_llm_response(raw_response)


def llm_operate_file(
    agent_name: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    base_url: str = aios_kernel_url,
    llms: Optional[List[Dict[str, Any]]] = None,
) -> LLMResponse:
    """
    Use the language model to perform file operations based on instructions.

    Args:
        agent_name: Identifier for the agent making the request.
        messages: List of message dictionaries.
        tools: Optional list of tools.
        base_url: API endpoint URL.
        llms: Optional list of LLM configurations.

    Returns:
        LLMResponse: Response object containing file operation results.
    """
    query = LLMQuery(
        agent_name=agent_name,
        messages=messages,
        tools=tools or [],
        action_type="operate_file",
        message_return_type="text",
        llms=llms,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_llm_response(raw_response)
