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
    def __init__(self, client: "AIOSClient"):
        self._c = client

    def chat(self, messages: List[Dict[str, Any]], llms: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        return llm_chat(
            agent_name=self._c.agent_name or "default_agent",
            messages=messages,
            base_url=self._c.base_url,
            llms=llms or self._c.default_llms,
        )

    def chat_json(self, messages: List[Dict[str, Any]], response_format: Optional[Dict[str, Any]] = None, llms: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        return llm_chat_with_json_output(
            agent_name=self._c.agent_name or "default_agent",
            messages=messages,
            base_url=self._c.base_url,
            llms=llms or self._c.default_llms,
            response_format=response_format,
        )

    def chat_tool(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], llms: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        return llm_chat_with_tool_call_output(
            agent_name=self._c.agent_name or "default_agent",
            messages=messages,
            tools=tools,
            base_url=self._c.base_url,
            llms=llms or self._c.default_llms,
        )

    def call_tool(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], llms: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        return llm_call_tool(
            agent_name=self._c.agent_name or "default_agent",
            messages=messages,
            tools=tools,
            base_url=self._c.base_url,
            llms=llms or self._c.default_llms,
        )

    def operate_file(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, llms: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        return llm_operate_file(
            agent_name=self._c.agent_name or "default_agent",
            messages=messages,
            tools=tools,
            base_url=self._c.base_url,
            llms=llms or self._c.default_llms,
        )


class MemoryClient:
    def __init__(self, client: "AIOSClient"):
        self._c = client

    def create(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryResponse:
        return create_memory(
            agent_name=self._c.agent_name or "default_agent",
            content=content,
            metadata=metadata,
            base_url=self._c.base_url,
        )

    def get(self, memory_id: str) -> MemoryResponse:
        return get_memory(
            agent_name=self._c.agent_name or "default_agent",
            memory_id=memory_id,
            base_url=self._c.base_url,
        )

    def update(self, memory_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> MemoryResponse:
        return update_memory(
            agent_name=self._c.agent_name or "default_agent",
            memory_id=memory_id,
            content=content,
            metadata=metadata,
            base_url=self._c.base_url,
        )

    def delete(self, memory_id: str) -> MemoryResponse:
        return delete_memory(
            agent_name=self._c.agent_name or "default_agent",
            memory_id=memory_id,
            base_url=self._c.base_url,
        )

    def search(self, query: str, k: int = 5) -> MemoryResponse:
        return search_memories(
            agent_name=self._c.agent_name or "default_agent",
            query=query,
            k=k,
            base_url=self._c.base_url,
        )

    def create_agentic(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryResponse:
        return create_agentic_memory(
            agent_name=self._c.agent_name or "default_agent",
            content=content,
            metadata=metadata,
            base_url=self._c.base_url,
        )


class StorageClient:
    def __init__(self, client: "AIOSClient"):
        self._c = client

    def mount(self, root_dir: str) -> StorageResponse:
        return mount(
            agent_name=self._c.agent_name or "default_agent",
            root_dir=root_dir,
            base_url=self._c.base_url,
        )

    def create_file(self, file_path: str) -> StorageResponse:
        return create_file(
            agent_name=self._c.agent_name or "default_agent",
            file_path=file_path,
            base_url=self._c.base_url,
        )

    def create_dir(self, dir_path: str) -> StorageResponse:
        return create_dir(
            agent_name=self._c.agent_name or "default_agent",
            dir_path=dir_path,
            base_url=self._c.base_url,
        )

    def write_file(self, file_path: str, content: str) -> StorageResponse:
        return write_file(
            agent_name=self._c.agent_name or "default_agent",
            file_path=file_path,
            content=content,
            base_url=self._c.base_url,
        )

    def retrieve_file(self, query_text: str, n: int, keywords: Optional[List[str]] = None) -> StorageResponse:
        return retrieve_file(
            agent_name=self._c.agent_name or "default_agent",
            query_text=query_text,
            n=n,
            keywords=keywords,
            base_url=self._c.base_url,
        )

    def rollback_file(self, file_path: str, n: int) -> StorageResponse:
        return rollback_file(
            agent_name=self._c.agent_name or "default_agent",
            file_path=file_path,
            n=n,
            base_url=self._c.base_url,
        )

    def share_file(self, file_path: str) -> StorageResponse:
        return share_file(
            agent_name=self._c.agent_name or "default_agent",
            file_path=file_path,
            base_url=self._c.base_url,
        )


class ToolClient:
    def __init__(self, client: "AIOSClient"):
        self._c = client

    def call(self, tool_calls: List[Dict[str, Any]]) -> ToolResponse:
        return call_tool(
            agent_name=self._c.agent_name or "default_agent",
            tool_calls=tool_calls,
            base_url=self._c.base_url,
        )

    def register(self, name: str, tool_class: Type[Any]) -> None:
        register_tool(name, tool_class)

    def get(self, name: str) -> Optional[Type[Any]]:
        return get_tool(name)

    def list(self) -> Dict[str, Type[Any]]:
        return list_registered_tools()


class PostClient:
    def __init__(self, client: "AIOSClient"):
        self._c = client

    def send(self, recipient: str, message: Union[str, Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> PostResponse:
        return send_post(
            sender=self._c.agent_name or "default_agent",
            recipient=recipient,
            message=message,
            metadata=metadata,
            base_url=self._c.base_url,
        )

    def receive(self, limit: int = 10, mark_as_read: bool = True) -> PostResponse:
        return receive_posts(
            agent_name=self._c.agent_name or "default_agent",
            limit=limit,
            mark_as_read=mark_as_read,
            base_url=self._c.base_url,
        )

    def broadcast(self, message: Union[str, Dict[str, Any]], topic: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> PostResponse:
        return broadcast_post(
            sender=self._c.agent_name or "default_agent",
            message=message,
            topic=topic,
            metadata=metadata,
            base_url=self._c.base_url,
        )

    def publish(self, topic: str, message: Union[str, Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> PostResponse:
        return publish_to_topic(
            sender=self._c.agent_name or "default_agent",
            topic=topic,
            message=message,
            metadata=metadata,
            base_url=self._c.base_url,
        )

    def subscribe(self, topic: str) -> PostResponse:
        return subscribe_topic(
            agent_name=self._c.agent_name or "default_agent",
            topic=topic,
            base_url=self._c.base_url,
        )


class AgentClient:
    def __init__(self, client: "AIOSClient"):
        self._c = client

    def register(self, name: str, agent_class: Type[Any]) -> None:
        register_agent(name, agent_class)

    def get(self, name: str) -> Optional[Type[Any]]:
        return get_agent(name)

    def list(self) -> Dict[str, Type[Any]]:
        return list_registered_agents()


class AIOSClient:
    """
    Unified client orchestrator for interacting with the AIOS Kernel.

    Provides scoped access to LLM, Memory, Storage, Tool, Post, and Agent subsystems.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        agent_name: Optional[str] = None,
        default_llms: Optional[List[Dict[str, Any]]] = None,
    ):
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

    def chat(self, prompt: str, system_prompt: Optional[str] = None, llms: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.llm.chat(messages, llms=llms)

    def remember(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryResponse:
        return self.memory.create(content, metadata=metadata)

    def recall(self, query: str, k: int = 5) -> MemoryResponse:
        return self.memory.search(query, k=k)

    def send_message(self, recipient: str, message: Union[str, Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> PostResponse:
        return self.post.send(recipient=recipient, message=message, metadata=metadata)


# Alias
CerebrumClient = AIOSClient

