from typing import Any, Dict, Optional

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request
from vectros_sdk.memory.models import MemoryQuery, MemoryResponse


def _parse_memory_response(raw_resp: Dict[str, Any]) -> MemoryResponse:
    """Parse raw response dictionary from kernel into MemoryResponse object."""
    if isinstance(raw_resp, dict):
        if "response" in raw_resp and isinstance(raw_resp["response"], dict):
            inner = dict(raw_resp["response"])
            if "status_code" not in inner and "status_code" in raw_resp:
                inner["status_code"] = raw_resp["status_code"]
            if "error" not in inner and "error" in raw_resp:
                inner["error"] = raw_resp["error"]
            if "success" not in inner and "success" in raw_resp:
                inner["success"] = raw_resp["success"]
            return MemoryResponse(**inner)
        return MemoryResponse(**raw_resp)
    return MemoryResponse(response_message=str(raw_resp))


def create_memory(
    agent_name: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    base_url: str = aios_kernel_url,
) -> MemoryResponse:
    """
    Store new memory with optional metadata.

    Args:
        agent_name: Namespace identifier for the agent (required).
        content: Memory text content (required).
        metadata: Optional key-value pairs (tags, priority, context, etc.).
        base_url: API endpoint URL.

    Returns:
        MemoryResponse: Response object containing memory_id and operation status.
    """
    query = MemoryQuery(
        agent_name=agent_name,
        action_type="create",
        content=content,
        metadata=metadata or {},
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_memory_response(raw_response)


def get_memory(
    agent_name: str,
    memory_id: str,
    base_url: str = aios_kernel_url,
) -> MemoryResponse:
    """
    Fetch memory contents by ID.

    Args:
        agent_name: Namespace identifier for the agent.
        memory_id: Target memory ID.
        base_url: API endpoint URL.

    Returns:
        MemoryResponse: Response object containing content and metadata.
    """
    query = MemoryQuery(
        agent_name=agent_name,
        action_type="get",
        memory_id=memory_id,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_memory_response(raw_response)


def update_memory(
    agent_name: str,
    memory_id: str,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    base_url: str = aios_kernel_url,
) -> MemoryResponse:
    """
    Modify existing memory content and/or metadata.

    Args:
        agent_name: Namespace identifier for the agent.
        memory_id: Target memory ID.
        content: New text content (optional).
        metadata: New/updated metadata fields (optional).
        base_url: API endpoint URL.

    Returns:
        MemoryResponse: Response object containing update status.
    """
    query = MemoryQuery(
        agent_name=agent_name,
        action_type="update",
        memory_id=memory_id,
        content=content,
        metadata=metadata,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_memory_response(raw_response)


def delete_memory(
    agent_name: str,
    memory_id: str,
    base_url: str = aios_kernel_url,
) -> MemoryResponse:
    """
    Remove memory by ID.

    Args:
        agent_name: Namespace identifier for the agent.
        memory_id: Target memory ID.
        base_url: API endpoint URL.

    Returns:
        MemoryResponse: Response object containing deletion status.
    """
    query = MemoryQuery(
        agent_name=agent_name,
        action_type="delete",
        memory_id=memory_id,
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_memory_response(raw_response)


def search_memories(
    agent_name: str,
    query: str,
    k: int = 5,
    base_url: str = aios_kernel_url,
) -> MemoryResponse:
    """
    Find relevant memories using natural language semantic search.

    Args:
        agent_name: Namespace identifier for the agent.
        query: Search phrase or question.
        k: Maximum number of ranked results to return (default: 5).
        base_url: API endpoint URL.

    Returns:
        MemoryResponse: Response object containing ranked search_results list.
    """
    query_obj = MemoryQuery(
        agent_name=agent_name,
        action_type="search",
        query=query,
        k=k,
    )
    raw_response = send_request(query_obj, base_url=base_url)
    return _parse_memory_response(raw_response)


def create_agentic_memory(
    agent_name: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    base_url: str = aios_kernel_url,
) -> MemoryResponse:
    """
    Store memory with autonomous cognitive organization capabilities (A-mem).

    Args:
        agent_name: Namespace identifier for the agent.
        content: Memory text with contextual signals.
        metadata: Structured attributes.
        base_url: API endpoint URL.

    Returns:
        MemoryResponse: Response object containing memory_id and status.
    """
    query = MemoryQuery(
        agent_name=agent_name,
        action_type="create_agentic",
        content=content,
        metadata=metadata or {},
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_memory_response(raw_response)
