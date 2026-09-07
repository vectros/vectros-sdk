"""
Memory API functional interface for Vectros SDK.

Provides functions for CRUD operations, semantic search, and autonomous
agentic memory (A-mem) organization in the AIOS kernel.
"""

from typing import Any, Dict, Optional

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.client.send_request import send_request
from vectros_sdk.memory.models import MemoryQuery, MemoryResponse


def _parse_memory_response(raw_resp: Dict[str, Any]) -> MemoryResponse:
    """
    Parse raw response dictionary from kernel into a typed MemoryResponse object.

    Args:
        raw_resp (Dict[str, Any]): Raw JSON response from AIOS kernel.

    Returns:
        MemoryResponse: Typed MemoryResponse model.
    """
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
    Store new memory content with optional structured metadata in an agent's namespace.

    Args:
        agent_name (str): Namespace identifier for the agent (required).
        content (str): Memory text content to store (required).
        metadata (Optional[Dict[str, Any]], optional): Key-value pairs (e.g. tags, priority, context). Defaults to None.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        MemoryResponse: Response object containing assigned `memory_id` and operation status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.memory.api import create_memory
        >>> resp = create_memory(
        ...     agent_name="project_bot",
        ...     content="Accelerate rollout by 2 weeks",
        ...     metadata={"priority": "high"}
        ... )
        >>> print(resp.memory_id)
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
    Fetch memory content and metadata by its unique memory ID.

    Args:
        agent_name (str): Namespace identifier for the agent.
        memory_id (str): Target unique memory ID.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        MemoryResponse: Response object containing `content` and `metadata`.

    Raises:
        AIOSKernelError: If kernel communication fails or memory is not found.

    Example:
        >>> from vectros_sdk.memory.api import get_memory
        >>> resp = get_memory("project_bot", "mem_123")
        >>> print(resp.content)
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
    Modify an existing memory's text content and/or metadata attributes.

    Args:
        agent_name (str): Namespace identifier for the agent.
        memory_id (str): Target unique memory ID.
        content (Optional[str], optional): Updated text content. Defaults to None.
        metadata (Optional[Dict[str, Any]], optional): Updated metadata fields. Defaults to None.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        MemoryResponse: Response object containing update status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.memory.api import update_memory
        >>> resp = update_memory(
        ...     agent_name="project_bot",
        ...     memory_id="mem_123",
        ...     content="Accelerate rollout by 3 weeks"
        ... )
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
    Remove a memory item permanently by ID from the agent's namespace.

    Args:
        agent_name (str): Namespace identifier for the agent.
        memory_id (str): Target memory ID to delete.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        MemoryResponse: Response object containing deletion status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.memory.api import delete_memory
        >>> resp = delete_memory("project_bot", "mem_123")
        >>> print(resp.success)
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
    Perform natural language semantic similarity search across an agent's stored memories.

    Args:
        agent_name (str): Namespace identifier for the agent.
        query (str): Search phrase or query string.
        k (int, optional): Maximum number of ranked results to return. Defaults to 5.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        MemoryResponse: Response object containing `search_results` with memory IDs and similarity scores.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.memory.api import search_memories
        >>> resp = search_memories("project_bot", query="timeline decisions", k=3)
        >>> for item in resp.search_results:
        ...     print(item["content"], item["score"])
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

    Enables dynamic linking, auto-clustering, and cognitive tagging in the AIOS kernel.

    Args:
        agent_name (str): Namespace identifier for the agent.
        content (str): Memory text containing contextual signals.
        metadata (Optional[Dict[str, Any]], optional): Structured cognitive attributes. Defaults to None.
        base_url (str, optional): API endpoint URL. Defaults to configured `aios_kernel_url`.

    Returns:
        MemoryResponse: Response object containing assigned memory_id and status.

    Raises:
        AIOSKernelError: If kernel communication fails.

    Example:
        >>> from vectros_sdk.memory.api import create_agentic_memory
        >>> resp = create_agentic_memory(
        ...     agent_name="research_bot",
        ...     content="Breakthrough in transformer attention scaling.",
        ...     metadata={"field": "ai_research"}
        ... )
    """
    query = MemoryQuery(
        agent_name=agent_name,
        action_type="create_agentic",
        content=content,
        metadata=metadata or {},
    )
    raw_response = send_request(query, base_url=base_url)
    return _parse_memory_response(raw_response)
