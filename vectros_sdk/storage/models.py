from typing import Any, Dict, List, Optional, Union
from pydantic import ConfigDict, Field

from vectros_sdk.core.models import Query, Response


class StorageQuery(Query):
    """Query model for Storage API requests to AIOS kernel."""
    query_class: str = "storage"
    agent_name: Optional[str] = None
    operation_type: str = Field(default="text")
    params: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class StorageResponse(Response):
    """Response model for Storage API operations."""
    response_class: str = "storage"
    response_message: Optional[str] = None
    finished: bool = False
    error: Optional[str] = None
    status_code: int = 200

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

