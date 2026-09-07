from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import ConfigDict, Field

from vectros_sdk.core.models import Query, Response


class PostQuery(Query):
    """Query model for Agent-to-Agent Communication (Post API) requests."""
    query_class: str = "post"
    agent_name: Optional[str] = None
    action_type: Literal["send", "receive", "broadcast", "publish", "subscribe"] = "send"
    recipient: Optional[str] = None
    topic: Optional[str] = None
    message: Optional[Union[str, Dict[str, Any]]] = None
    message_id: Optional[str] = None
    limit: Optional[int] = None
    mark_as_read: bool = True
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class PostResponse(Response):
    """Response model for Agent-to-Agent Communication operations."""
    response_class: str = "post"
    response_message: Optional[str] = None
    message_id: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    finished: bool = True
    error: Optional[str] = None
    status_code: int = 200

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

