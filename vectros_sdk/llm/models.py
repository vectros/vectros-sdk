"""
Data models for LLM Core API in Vectros SDK.

Defines `LLMQuery` for requesting language model completions and operations,
and `LLMResponse` for encapsulating generated text, tool calls, and execution metadata.
"""

import json
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import ConfigDict, Field, field_validator

from vectros_sdk.core.models import Query, Response


class LLMQuery(Query):
    """
    Query model for LLM core API requests dispatched to the AIOS kernel.

    Attributes:
        query_class (str): Identifier tag, always 'llm'.
        llms (Optional[List[Dict[str, Any]]]): List of LLM backend configurations (e.g. name, backend).
        messages (List[Dict[str, Union[str, Any]]]): Conversation message history (roles: system, user, assistant).
        tools (Optional[List[Dict[str, Any]]]): Available function tools schema specifications.
        action_type (Literal['chat', 'tool_use', 'operate_file']): Mode of LLM interaction.
        message_return_type (Literal['text', 'json']): Expected format of the LLM response text.
        response_format (Optional[Dict[str, Any]]): JSON schema if message_return_type is 'json'.
        tool_choice (Optional[Union[str, Dict[str, Any]]]): Tool choice mode ('auto', 'required', etc.).

    Example:
        >>> from vectros_sdk.llm.models import LLMQuery
        >>> q = LLMQuery(
        ...     agent_name="my_bot",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     action_type="chat"
        ... )
    """
    query_class: str = Field(default="llm", description="Query class tag, defaults to 'llm'.")
    llms: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional list of LLM configurations for model routing.")
    messages: List[Dict[str, Union[str, Any]]] = Field(..., description="List of conversation messages.")
    tools: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="List of tool specifications.")
    action_type: Literal["chat", "tool_use", "operate_file"] = Field(default="chat", description="LLM interaction type.")
    message_return_type: Literal["text", "json"] = Field(default="text", description="Expected text return format.")
    response_format: Optional[Dict[str, Any]] = Field(default=None, description="Schema format for structured JSON output.")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Tool selection behavior.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class LLMResponse(Response):
    """
    Response model for LLM core API execution results.

    Attributes:
        response_class (str): Module tag, always 'llm'.
        response_message (Optional[str]): Generated response text or JSON string from the LLM.
        tool_calls (Optional[Union[List[Dict[str, Any]], str]]): Parsed or raw tool call specifications.
        finished (bool): Indicates if the LLM request finished successfully.
        error (Optional[str]): Error message if execution encountered an issue.
        status_code (int): HTTP status code from kernel dispatcher.

    Example:
        >>> from vectros_sdk.llm.models import LLMResponse
        >>> resp = LLMResponse(response_message="Hello!", finished=True)
        >>> print(resp["response"]["response_message"])
        'Hello!'
    """
    response_class: str = Field(default="llm", description="Response category tag.")
    response_message: Optional[str] = Field(default=None, description="Generated response message from LLM.")
    tool_calls: Optional[Union[List[Dict[str, Any]], str]] = Field(default=None, description="Tool invocations selected by LLM.")
    finished: bool = Field(default=False, description="Whether LLM processing completed.")
    error: Optional[str] = Field(default=None, description="Error message if execution failed.")
    status_code: int = Field(default=200, description="HTTP response code.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @field_validator("tool_calls", mode="before")
    @classmethod
    def parse_tool_calls(cls, v: Any) -> Any:
        """
        Validate and deserialize tool_calls if provided as a JSON string.

        Args:
            v (Any): Raw tool_calls value (string or List[Dict]).

        Returns:
            Any: Parsed Python list of dictionaries if valid JSON string, otherwise raw value.
        """
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, (list, dict)):
                    return parsed
            except Exception:
                return v
        return v
