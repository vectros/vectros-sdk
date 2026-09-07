"""
Unified AIOS Client (CerebrumClient) module for Vectros SDK.

Provides high-level client interfaces (`AIOSClient` / `CerebrumClient`) and
specialized sub-clients for LLM, Memory, Storage, Tool, Post, and Agent subsystems.
"""

from typing import Any, Dict, List, Optional, Type, Union

from vectros_sdk.agent.registry import (
    clear_agent_registry,
    get_agent,
    list_registered_agents,
    register_agent,
)
from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.llm.api import (
    llm_call_tool,
    llm_chat,
    llm_chat_with_json_output,
    llm_chat_with_tool_call_output,
    llm_operate_file,
)
from vectros_sdk.llm.models import LLMResponse
from vectros_sdk.memory.api import (
    create_agentic_memory,
    create_memory,
    delete_memory,
    get_memory,
    search_memories,
    update_memory,
)
from vectros_sdk.memory.models import MemoryResponse
from vectros_sdk.post.api import (
    broadcast_post,
    publish_to_topic,
    receive_posts,
    send_post,
    subscribe_topic,
)
from vectros_sdk.post.models import PostResponse
from vectros_sdk.storage.api import (
    create_dir,
    create_file,
    mount,
    retrieve_file,
    rollback_file,
    share_file,
    write_file,
)
from vectros_sdk.storage.models import StorageResponse
from vectros_sdk.tool.api import call_tool
from vectros_sdk.tool.core.registry import (
    clear_registry as clear_tool_registry,
    get_tool,
    list_registered_tools,
    register_tool,
)
from vectros_sdk.tool.models import ToolResponse


class LLMClient:
    """
    Sub-client managing LLM Core API operations for a scoped AIOSClient session.
    """

    def __init__(self, client: "AIOSClient"):
        self._c = client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        llms: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Send a conversation message list to the language model.

        Args:
            messages (List[Dict[str, Any]]): Conversation history dictionaries.
            llms (Optional[List[Dict[str, Any]]], optional): LLM model routing override. Defaults to None.

        Returns:
            LLMResponse: Response containing generated text.
        """
        return llm_chat(
            agent_name=self._c.agent_name or "default_agent",
            messages=messages,
            base_url=self._c.base_url,
            llms=llms or self._c.default_llms,
        )

    def chat_json(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
        llms: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Request structured JSON output from the language model conforming to schema.

        Args:
            messages (List[Dict[str, Any]]): Conversation messages.
            response_format (Optional[Dict[str, Any]], optional): JSON schema. Defaults to None.
            llms (Optional[List[Dict[str, Any]]], optional): LLM model override. Defaults to None.

        Returns:
            LLMResponse: Response containing parsed or raw JSON string.
        """
        return llm_chat_with_json_output(
            agent_name=self._c.agent_name or "default_agent",
            messages=messages,
            base_url=self._c.base_url,
            llms=llms or self._c.default_llms,
            response_format=response_format,
        )

    def chat_tool(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        llms: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Chat with tool integration allowing dynamic function calling.

        Args:
            messages (List[Dict[str, Any]]): Conversation messages.
            tools (List[Dict[str, Any]]): List of tool specifications.
            llms (Optional[List[Dict[str, Any]]], optional): LLM model override. Defaults to None.

        Returns:
            LLMResponse: Response containing model decision and tool calls.
        """
        return llm_chat_with_tool_call_output(
            agent_name=self._c.agent_name or "default_agent",
            messages=messages,
            tools=tools,
            base_url=self._c.base_url,
            llms=llms or self._c.default_llms,
        )

    def call_tool(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        llms: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Explicitly invoke tools with user messages via LLM core.

        Args:
            messages (List[Dict[str, Any]]): Message history.
            tools (List[Dict[str, Any]]): Tools to invoke.
            llms (Optional[List[Dict[str, Any]]], optional): LLM model override. Defaults to None.

        Returns:
            LLMResponse: Response containing execution result.
        """
        return llm_call_tool(
            agent_name=self._c.agent_name or "default_agent",
            messages=messages,
            tools=tools,
            base_url=self._c.base_url,
            llms=llms or self._c.default_llms,
        )

    def operate_file(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        llms: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Perform file operations through language model instructions.

        Args:
            messages (List[Dict[str, Any]]): Conversation messages.
            tools (Optional[List[Dict[str, Any]]], optional): Optional tools. Defaults to None.
            llms (Optional[List[Dict[str, Any]]], optional): LLM model override. Defaults to None.

        Returns:
            LLMResponse: Response containing file operation results.
        """
        return llm_operate_file(
            agent_name=self._c.agent_name or "default_agent",
            messages=messages,
            tools=tools,
            base_url=self._c.base_url,
            llms=llms or self._c.default_llms,
        )


class MemoryClient:
    """
    Sub-client managing Memory API CRUD and semantic search operations.
    """

    def __init__(self, client: "AIOSClient"):
        self._c = client

    def create(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryResponse:
        """
        Store a new memory item in the client's agent namespace.

        Args:
            content (str): Memory text content.
            metadata (Optional[Dict[str, Any]], optional): Metadata attributes. Defaults to None.

        Returns:
            MemoryResponse: Response with assigned memory_id.
        """
        return create_memory(
            agent_name=self._c.agent_name or "default_agent",
            content=content,
            metadata=metadata,
            base_url=self._c.base_url,
        )

    def get(self, memory_id: str) -> MemoryResponse:
        """
        Retrieve memory content and metadata by ID.

        Args:
            memory_id (str): Target memory ID.

        Returns:
            MemoryResponse: Response with content and metadata.
        """
        return get_memory(
            agent_name=self._c.agent_name or "default_agent",
            memory_id=memory_id,
            base_url=self._c.base_url,
        )

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryResponse:
        """
        Update an existing memory's content or metadata.

        Args:
            memory_id (str): Target memory ID.
            content (Optional[str], optional): New content text. Defaults to None.
            metadata (Optional[Dict[str, Any]], optional): New metadata. Defaults to None.

        Returns:
            MemoryResponse: Response with update status.
        """
        return update_memory(
            agent_name=self._c.agent_name or "default_agent",
            memory_id=memory_id,
            content=content,
            metadata=metadata,
            base_url=self._c.base_url,
        )

    def delete(self, memory_id: str) -> MemoryResponse:
        """
        Delete a memory item by ID.

        Args:
            memory_id (str): Target memory ID to delete.

        Returns:
            MemoryResponse: Response with deletion status.
        """
        return delete_memory(
            agent_name=self._c.agent_name or "default_agent",
            memory_id=memory_id,
            base_url=self._c.base_url,
        )

    def search(self, query: str, k: int = 5) -> MemoryResponse:
        """
        Search memories semantically by query phrase.

        Args:
            query (str): Search phrase.
            k (int, optional): Maximum results. Defaults to 5.

        Returns:
            MemoryResponse: Response containing ranked search_results.
        """
        return search_memories(
            agent_name=self._c.agent_name or "default_agent",
            query=query,
            k=k,
            base_url=self._c.base_url,
        )

    def create_agentic(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryResponse:
        """
        Create an autonomous cognitive memory (A-mem) with dynamic clustering.

        Args:
            content (str): Memory text.
            metadata (Optional[Dict[str, Any]], optional): Structured metadata. Defaults to None.

        Returns:
            MemoryResponse: Response with memory ID.
        """
        return create_agentic_memory(
            agent_name=self._c.agent_name or "default_agent",
            content=content,
            metadata=metadata,
            base_url=self._c.base_url,
        )


class StorageClient:
    """
    Sub-client managing Storage API filesystem and file sharing operations.
    """

    def __init__(self, client: "AIOSClient"):
        self._c = client

    def mount(self, root_dir: str) -> StorageResponse:
        """Mount a root storage directory for the client session."""
        return mount(
            agent_name=self._c.agent_name or "default_agent",
            root_dir=root_dir,
            base_url=self._c.base_url,
        )

    def create_file(self, file_path: str) -> StorageResponse:
        """Create an empty file at the relative path."""
        return create_file(
            agent_name=self._c.agent_name or "default_agent",
            file_path=file_path,
            base_url=self._c.base_url,
        )

    def create_dir(self, dir_path: str) -> StorageResponse:
        """Create a directory structure at the relative path."""
        return create_dir(
            agent_name=self._c.agent_name or "default_agent",
            dir_path=dir_path,
            base_url=self._c.base_url,
        )

    def write_file(self, file_path: str, content: str) -> StorageResponse:
        """Write text content to a file, creating it if necessary."""
        return write_file(
            agent_name=self._c.agent_name or "default_agent",
            file_path=file_path,
            content=content,
            base_url=self._c.base_url,
        )

    def retrieve_file(
        self,
        query_text: str,
        n: int,
        keywords: Optional[List[str]] = None,
    ) -> StorageResponse:
        """Search and retrieve files matching query criteria."""
        return retrieve_file(
            agent_name=self._c.agent_name or "default_agent",
            query_text=query_text,
            n=n,
            keywords=keywords,
            base_url=self._c.base_url,
        )

    def rollback_file(self, file_path: str, n: int) -> StorageResponse:
        """Revert a file by n versions."""
        return rollback_file(
            agent_name=self._c.agent_name or "default_agent",
            file_path=file_path,
            n=n,
            base_url=self._c.base_url,
        )

    def share_file(self, file_path: str) -> StorageResponse:
        """Share a file with other agents across AIOS."""
        return share_file(
            agent_name=self._c.agent_name or "default_agent",
            file_path=file_path,
            base_url=self._c.base_url,
        )


class ToolClient:
    """
    Sub-client managing tool execution and local tool registry.
    """

    def __init__(self, client: "AIOSClient"):
        self._c = client

    def call(self, tool_calls: List[Dict[str, Any]]) -> ToolResponse:
        """Dispatch tool calls to AIOS kernel."""
        return call_tool(
            agent_name=self._c.agent_name or "default_agent",
            tool_calls=tool_calls,
            base_url=self._c.base_url,
        )

    def register(self, name: str, tool_class: Type[Any]) -> None:
        """Register a tool class in the local registry."""
        register_tool(name, tool_class)

    def get(self, name: str) -> Optional[Type[Any]]:
        """Retrieve a tool class from the registry."""
        return get_tool(name)

    def list(self) -> Dict[str, Type[Any]]:
        """List all registered tool classes."""
        return list_registered_tools()


class PostClient:
    """
    Sub-client managing Agent-to-Agent Communication, messaging, and pub/sub.
    """

    def __init__(self, client: "AIOSClient"):
        self._c = client

    def send(
        self,
        recipient: str,
        message: Union[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PostResponse:
        """Send a direct message to a recipient agent."""
        return send_post(
            sender=self._c.agent_name or "default_agent",
            recipient=recipient,
            message=message,
            metadata=metadata,
            base_url=self._c.base_url,
        )

    def receive(
        self,
        limit: int = 10,
        mark_as_read: bool = True,
    ) -> PostResponse:
        """Fetch pending messages from the agent's mailbox."""
        return receive_posts(
            agent_name=self._c.agent_name or "default_agent",
            limit=limit,
            mark_as_read=mark_as_read,
            base_url=self._c.base_url,
        )

    def broadcast(
        self,
        message: Union[str, Dict[str, Any]],
        topic: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PostResponse:
        """Broadcast a message to all agents or a specific topic."""
        return broadcast_post(
            sender=self._c.agent_name or "default_agent",
            message=message,
            topic=topic,
            metadata=metadata,
            base_url=self._c.base_url,
        )

    def publish(
        self,
        topic: str,
        message: Union[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PostResponse:
        """Publish a message to a pub/sub channel."""
        return publish_to_topic(
            sender=self._c.agent_name or "default_agent",
            topic=topic,
            message=message,
            metadata=metadata,
            base_url=self._c.base_url,
        )

    def subscribe(self, topic: str) -> PostResponse:
        """Subscribe to a pub/sub channel."""
        return subscribe_topic(
            agent_name=self._c.agent_name or "default_agent",
            topic=topic,
            base_url=self._c.base_url,
        )


class AgentClient:
    """
    Sub-client managing local agent registration and discovery.
    """

    def __init__(self, client: "AIOSClient"):
        self._c = client

    def register(self, name: str, agent_class: Type[Any]) -> None:
        """Register an agent class in the local registry."""
        register_agent(name, agent_class)

    def get(self, name: str) -> Optional[Type[Any]]:
        """Retrieve an agent class from the registry."""
        return get_agent(name)

    def list(self) -> Dict[str, Type[Any]]:
        """List all registered agent classes."""
        return list_registered_agents()


class AIOSClient:
    """
    Unified client orchestrator for interacting with the AIOS Kernel.

    Provides scoped access to LLM, Memory, Storage, Tool, Post, and Agent subsystems.

    Attributes:
        base_url (str): AIOS kernel base endpoint URL.
        agent_name (Optional[str]): Active agent namespace identifier.
        default_llms (Optional[List[Dict[str, Any]]]): Default LLM configurations for routing.
        llm (LLMClient): Sub-client for language model operations.
        memory (MemoryClient): Sub-client for memory storage and retrieval.
        storage (StorageClient): Sub-client for filesystem operations.
        tool (ToolClient): Sub-client for tool calling and registry.
        post (PostClient): Sub-client for agent-to-agent messaging.
        agent (AgentClient): Sub-client for agent registration.

    Example:
        >>> from vectros_sdk import AIOSClient
        >>> client = AIOSClient(base_url="http://localhost:8000", agent_name="assistant")
        >>> resp = client.chat("What is AIOS?")
        >>> client.remember("AIOS is an AI operating system.")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        agent_name: Optional[str] = None,
        default_llms: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the AIOS unified client.

        Args:
            base_url (Optional[str], optional): Kernel URL. Defaults to `aios_kernel_url`.
            agent_name (Optional[str], optional): Default agent namespace for requests.
            default_llms (Optional[List[Dict[str, Any]]], optional): Default LLM configuration list.
        """
        self.base_url = base_url or aios_kernel_url
        self.agent_name = agent_name
        self.default_llms = default_llms

        # Subsystems
        self.llm = LLMClient(self)
        self.memory = MemoryClient(self)
        self.storage = StorageClient(self)
        self.tool = ToolClient(self)
        self.post = PostClient(self)
        self.agent = AgentClient(self)

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        llms: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        High-level convenience method to chat with the LLM.

        Args:
            prompt (str): User prompt text.
            system_prompt (Optional[str], optional): System prompt framing. Defaults to None.
            llms (Optional[List[Dict[str, Any]]], optional): LLM model override. Defaults to None.

        Returns:
            LLMResponse: Generated model response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.llm.chat(messages, llms=llms)

    def remember(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryResponse:
        """
        High-level convenience method to store a memory item.

        Args:
            content (str): Text memory content.
            metadata (Optional[Dict[str, Any]], optional): Structured metadata attributes.

        Returns:
            MemoryResponse: Response with assigned memory_id.
        """
        return self.memory.create(content, metadata=metadata)

    def recall(
        self,
        query: str,
        k: int = 5,
    ) -> MemoryResponse:
        """
        High-level convenience method to search memories semantically.

        Args:
            query (str): Semantic search phrase.
            k (int, optional): Maximum ranked results. Defaults to 5.

        Returns:
            MemoryResponse: Search results matching the query.
        """
        return self.memory.search(query, k=k)

    def send_message(
        self,
        recipient: str,
        message: Union[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PostResponse:
        """
        High-level convenience method to send a direct message to another agent.

        Args:
            recipient (str): Target recipient agent name.
            message (Union[str, Dict[str, Any]]): Text or structured payload.
            metadata (Optional[Dict[str, Any]], optional): Metadata headers.

        Returns:
            PostResponse: Delivery confirmation.
        """
        return self.post.send(recipient=recipient, message=message, metadata=metadata)


# Alias
CerebrumClient = AIOSClient
"""Alias of AIOSClient for Cerebrum architecture compatibility."""
