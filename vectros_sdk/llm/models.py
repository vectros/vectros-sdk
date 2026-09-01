import json
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import ConfigDict, Field, field_validator

from vectros_sdk.core.models import Query, Response


class LLMQuery(Query):
    """Query model for LLM core API requests."""
    query_class: str = "llm"
    llms: Optional[List[Dict[str, Any]]] = Field(default=None)
    messages: List[Dict[str, Union[str, Any]]]
    tools: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    action_type: Literal["chat", "tool_use", "operate_file"] = Field(default="chat")
    message_return_type: Literal["text", "json"] = Field(default="text")
    response_format: Optional[Dict[str, Any]] = Field(default=None)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class LLMResponse(Response):
    """Response model for LLM core API results."""
    response_class: str = "llm"
    response_message: Optional[str] = None
    tool_calls: Optional[Union[List[Dict[str, Any]], str]] = None
    finished: bool = False
    error: Optional[str] = None
    status_code: int = 200

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @field_validator("tool_calls", mode="before")
    @classmethod
    def parse_tool_calls(cls, v: Any) -> Any:
        """Support tool_calls passed either as JSON string or parsed List[Dict]."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, (list, dict)):
                    return parsed
            except Exception:
                return v
        return v
