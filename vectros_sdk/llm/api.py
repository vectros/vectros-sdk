"""
LLM Core API functional interface for Vectros SDK.

Provides functions for multi-turn chat, structured JSON outputs, tool calling,
and file operations with language models in AIOS.
"""

from typing import Any, Dict, List, Optional

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request
from vectros_sdk.llm.models import LLMQuery, LLMResponse


def _parse_llm_response(raw_resp: Dict[str, Any]) -> LLMResponse:
    """
    Parse raw response dictionary from kernel dispatcher into an LLMResponse model.

    Args:
        raw_resp (Dict[str, Any]): Raw JSON response dictionary from AIOS kernel.

    Returns:
        LLMResponse: Typed LLMResponse instance with unwrapped fields.
    """
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
    Perform a basic text-based conversation with the language model.

    Args:
        agent_name (str): Identifier for the agent making the request.
        messages (List[Dict[str, Any]]): List of message dictionaries with 'role' and 'content'.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.
        llms (Optional[List[Dict[str, Any]]], optional): Optional list of LLM configurations for routing.

    Returns:
        LLMResponse: Response object containing the model's text response and execution status.

    Raises:
        AIOSKernelError: If kernel communication fails or returns an error.

    Example:
        >>> from vectros_sdk.llm.api import llm_chat
        >>> response = llm_chat(
        ...     "my_assistant",
        ...     messages=[
        ...         {"role": "system", "content": "You are a helpful AI assistant."},
        ...         {"role": "user", "content": "What is Python?"}
        ...     ]
        ... )
        >>> print(response.response_message)
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
    Get structured JSON responses from the language model adhering to a specified schema.

    Args:
        agent_name (str): Identifier for the agent making the request.
        messages (List[Dict[str, Any]]): List of conversation messages.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.
        llms (Optional[List[Dict[str, Any]]], optional): Optional list of LLM configurations.
        response_format (Optional[Dict[str, Any]], optional): JSON schema specifying output structure.

    Returns:
        LLMResponse: Response object containing the structured JSON string in `response_message`.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.llm.api import llm_chat_with_json_output
        >>> response = llm_chat_with_json_output(
        ...     "analyzer",
        ...     messages=[{"role": "user", "content": "Extract topics from AIOS"}],
        ...     response_format={"type": "json_object"}
        ... )
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
    Chat with tool integration, allowing the model to decide which tools to call dynamically.

    Args:
        agent_name (str): Identifier for the agent making the request.
        messages (List[Dict[str, Any]]): List of conversation messages.
        tools (List[Dict[str, Any]]): List of available tool schema specifications.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.
        llms (Optional[List[Dict[str, Any]]], optional): Optional list of LLM configurations.

    Returns:
        LLMResponse: Response object containing `tool_calls` chosen by the language model.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.llm.api import llm_chat_with_tool_call_output
        >>> response = llm_chat_with_tool_call_output(
        ...     "researcher",
        ...     messages=[{"role": "user", "content": "Find papers on transformers"}],
        ...     tools=[{"name": "search", "parameters": {"query": {"type": "string"}}}]
        ... )
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
    Direct tool invocation instructing the language model to execute specified tools with input.

    Args:
        agent_name (str): Identifier for the agent making the request.
        messages (List[Dict[str, Any]]): List of message dictionaries.
        tools (List[Dict[str, Any]]): List of tools and parameters to invoke.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.
        llms (Optional[List[Dict[str, Any]]], optional): Optional list of LLM configurations.

    Returns:
        LLMResponse: Response object containing tool calls and execution results.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.llm.api import llm_call_tool
        >>> response = llm_call_tool(
        ...     "weather_agent",
        ...     messages=[{"role": "user", "content": "Get forecast for NYC"}],
        ...     tools=[{"name": "weather_service", "parameters": {"city": "NYC"}}]
        ... )
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
    Instruct the language model to perform file operations based on conversation instructions.

    Args:
        agent_name (str): Identifier for the agent making the request.
        messages (List[Dict[str, Any]]): List of message dictionaries describing the file action.
        tools (Optional[List[Dict[str, Any]]], optional): Optional auxiliary tools. Defaults to None.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.
        llms (Optional[List[Dict[str, Any]]], optional): Optional list of LLM configurations.

    Returns:
        LLMResponse: Response object containing the file operation result message.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.llm.api import llm_operate_file
        >>> response = llm_operate_file(
        ...     "file_bot",
        ...     messages=[{"role": "user", "content": "Create notes.txt with todo items"}]
        ... )
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
