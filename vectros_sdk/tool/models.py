from typing import Any, Dict, List, Optional, Union
from pydantic import ConfigDict, Field

from vectros_sdk.core.models import Query, Response


class ToolQuery(Query):
    """Query model for Tool API requests to AIOS kernel."""
    query_class: str = "tool"
    agent_name: Optional[str] = None
    tool_calls: List[Dict[str, Union[str, Any]]]

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class ToolResponse(Response):
    """Response model for Tool API operations."""
    response_class: str = "tool"
    response_message: Optional[str] = None
    finished: bool = False
    error: Optional[str] = None
    status_code: int = 200

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

