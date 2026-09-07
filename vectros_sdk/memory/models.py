"""
Data models for the Memory API in Vectros SDK.

Defines `MemoryQuery` for requesting memory CRUD/search operations and
`MemoryResponse` for structured memory results and semantic search scores.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import ConfigDict, Field

from vectros_sdk.core.models import Query, Response


class MemoryQuery(Query):
    """
    Query model for Memory API requests dispatched to the AIOS kernel.

    Attributes:
        query_class (str): Module tag, always 'memory'.
        agent_name (str): Namespace identifier for agent memory isolation.
        action_type (Literal['create', 'get', 'update', 'delete', 'search', 'create_agentic']): Operation type.
        memory_id (Optional[str]): Unique memory ID required for get, update, and delete actions.
        content (Optional[str]): Text content to store or update.
        metadata (Optional[Dict[str, Any]]): Key-value tags and attributes.
        query (Optional[str]): Search phrase for semantic queries.
        k (Optional[int]): Number of ranked search results to return.
        params (Optional[Dict[str, Any]]): Additional parameters.

    Example:
        >>> from vectros_sdk.memory.models import MemoryQuery
        >>> q = MemoryQuery(
        ...     agent_name="my_bot",
        ...     action_type="create",
        ...     content="Remember this meeting note."
        ... )
    """
    query_class: str = Field(default="memory", description="Module category tag.")
    agent_name: str = Field(..., description="Namespace identifier for the agent.")
    action_type: Literal["create", "get", "update", "delete", "search", "create_agentic"] = Field(
        default="create", description="Memory operation to perform."
    )
    memory_id: Optional[str] = Field(default=None, description="Memory ID for targeted CRUD operations.")
    content: Optional[str] = Field(default=None, description="Text body of the memory.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata key-value pairs.")
    query: Optional[str] = Field(default=None, description="Semantic search phrase.")
    k: Optional[int] = Field(default=None, description="Maximum number of search results to return.")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Extra operation parameters.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class MemoryResponse(Response):
    """
    Response model for Memory CRUD and Semantic Search operations.

    Attributes:
        response_class (str): Module tag, always 'memory'.
        success (bool): Whether the memory operation succeeded.
        memory_id (Optional[str]): Assigned or queried memory ID.
        content (Optional[str]): Retrieved memory text content.
        metadata (Optional[Dict[str, Any]]): Retrieved metadata dictionary.
        search_results (Optional[List[Dict[str, Any]]]): Ranked list of matching memory objects.
        finished (bool): Operation completion status.
        error (Optional[str]): Error message if execution failed.
        status_code (int): HTTP status code from kernel dispatcher.

    Example:
        >>> from vectros_sdk.memory.models import MemoryResponse
        >>> resp = MemoryResponse(success=True, memory_id="mem_123")
        >>> print(resp["memory_id"])
        'mem_123'
    """
    response_class: str = Field(default="memory", description="Response category tag.")
    success: bool = Field(default=True, description="Indicates if operation was successful.")
    memory_id: Optional[str] = Field(default=None, description="ID of the created/modified/retrieved memory.")
    content: Optional[str] = Field(default=None, description="Text content of the retrieved memory.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata associated with the memory.")
    search_results: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Ranked list of search results for semantic queries."
    )
    finished: bool = Field(default=True, description="Whether the operation finished.")
    error: Optional[str] = Field(default=None, description="Error message if operation failed.")
    status_code: int = Field(default=200, description="HTTP status code from kernel dispatcher.")

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
