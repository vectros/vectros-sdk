from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import ConfigDict, Field

from vectros_sdk.core.models import Query, Response


class MemoryQuery(Query):
    """Query model for Memory API requests to AIOS kernel."""
    query_class: str = "memory"
    agent_name: str
    action_type: Literal["create", "get", "update", "delete", "search", "create_agentic"] = "create"
    memory_id: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    query: Optional[str] = None
    k: Optional[int] = None
    params: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class MemoryResponse(Response):
    """Unified Response model for Memory CRUD and Semantic Search operations."""
    response_class: str = "memory"
    success: bool = True
    memory_id: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    finished: bool = True
    error: Optional[str] = None
    status_code: int = 200

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
