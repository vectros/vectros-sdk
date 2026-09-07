from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from vectros_sdk.client.config import aios_kernel_url
from vectros_sdk.llm.api import llm_chat
from vectros_sdk.llm.models import LLMResponse
from vectros_sdk.memory.api import create_memory, search_memories
from vectros_sdk.memory.models import MemoryResponse


class BaseAgent(ABC):
    """
    Abstract base class for developing AIOS device-side agents.

    Provides core lifecycle parameters and integrated helpers for interacting
    with AIOS Kernel modules (LLM, Memory, Storage, Tools).
    """

    def __init__(
        self,
        agent_name: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        llm_config: Optional[List[Dict[str, Any]]] = None,
        base_url: str = aios_kernel_url,
    ):
        self.agent_name = agent_name
        self.system_prompt = system_prompt or f"You are an AI assistant named {agent_name}."
        self.tools = tools or []
        self.llm_config = llm_config
        self.base_url = base_url

    @abstractmethod
    def run(self, task: Union[str, Dict[str, Any]]) -> Any:
        """
        Main execution loop / entry point for the agent.

        Args:
            task: Task description string or structured task dict.

        Returns:
            Any: Result of the agent task execution.
        """
        raise NotImplementedError

    def chat(
        self,
        prompt: str,
        llms: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        Convenience helper to chat with the LLM using this agent's configuration.

        Args:
            prompt: User message prompt text.
            llms: Optional LLM routing configuration override.

        Returns:
            LLMResponse: Response from the LLM.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return llm_chat(
            agent_name=self.agent_name,
            messages=messages,
            base_url=self.base_url,
            llms=llms or self.llm_config,
        )

    def remember(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryResponse:
        """
        Convenience helper to persist a memory item in this agent's namespace.

        Args:
            content: Text memory content.
            metadata: Optional key-value metadata.

        Returns:
            MemoryResponse: Response containing memory_id.
        """
        return create_memory(
            agent_name=self.agent_name,
            content=content,
            metadata=metadata,
            base_url=self.base_url,
        )

    def recall(
        self,
        query: str,
        k: int = 5,
    ) -> MemoryResponse:
        """
        Convenience helper to search relevant memories from this agent's namespace.

        Args:
            query: Semantic search query string.
            k: Maximum number of ranked results.

        Returns:
            MemoryResponse: Response containing ranked memory results.
        """
        return search_memories(
            agent_name=self.agent_name,
            query=query,
            k=k,
            base_url=self.base_url,
        )

