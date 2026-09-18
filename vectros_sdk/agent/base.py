"""
Base Agent abstraction for Vectros SDK agent development.

Defines the `BaseAgent` class providing standardized agent lifecycle management,
abstract task execution, and convenience helpers for LLM chat, memory storage,
and semantic memory recall.
"""

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

    Provides core lifecycle attributes and integrated helpers for interacting
    with AIOS Kernel modules (LLM, Memory, Storage, Tools, Post).

    Attributes:
        agent_name (str): Unique namespace identifier for this agent.
        system_prompt (str): Default system prompt framing the agent's behavior.
        tools (List[Dict[str, Any]]): List of tools available to this agent.
        llm_config (Optional[List[Dict[str, Any]]]): LLM routing and backend configuration.
        base_url (str): AIOS kernel endpoint URL.

    Example:
        >>> class MathBot(BaseAgent):
        ...     def run(self, task):
        ...         res = self.chat(f"Solve: {task}")
        ...         self.remember(f"Solved: {task}")
        ...         return res.response_message
    """

    def __init__(
        self,
        agent_name: str,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        llm_config: Optional[List[Dict[str, Any]]] = None,
        base_url: str = aios_kernel_url,
        socket_path: Optional[str] = None,
    ):
        """
        Initialize the agent with name, system prompt, tools, and kernel configuration.

        Args:
            agent_name (str): Identifier name for the agent.
            system_prompt (Optional[str], optional): System prompt. If None, defaults to generic assistant prompt.
            tools (Optional[List[Dict[str, Any]]], optional): Available tools list. Defaults to empty list.
            llm_config (Optional[List[Dict[str, Any]]], optional): LLM configuration list for model routing. Defaults to None.
            base_url (str, optional): AIOS kernel URL. Defaults to configured `aios_kernel_url`.
            socket_path (Optional[str], optional): Path to a standing `aiosctl serve-kernel`
                Unix socket. When given, `chat`/`remember`/`recall` route to the real kernel
                instead of `base_url`'s dead HTTP mock endpoint (SDK.3's `real_kernel.py` path).
                Defaults to None (HTTP mock path, unchanged).
        """
        self.agent_name = agent_name
        self.system_prompt = system_prompt or f"You are an AI assistant named {agent_name}."
        self.tools = tools or []
        self.llm_config = llm_config
        self.base_url = base_url
        self.socket_path = socket_path

    @abstractmethod
    def run(self, task: Union[str, Dict[str, Any]]) -> Any:
        """
        Main execution loop and entry point for the agent.

        Args:
            task (Union[str, Dict[str, Any]]): Task description string or structured task dictionary.

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
        Convenience helper to chat with the LLM using this agent's configuration and system prompt.

        Args:
            prompt (str): User message prompt text.
            llms (Optional[List[Dict[str, Any]]], optional): Optional LLM routing configuration override.

        Returns:
            LLMResponse: Response object from the LLM.

        Raises:
            AIOSKernelError: If kernel request fails.
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
            socket_path=self.socket_path,
        )

    def remember(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryResponse:
        """
        Convenience helper to persist a memory item in this agent's namespace.

        Args:
            content (str): Text memory content to store.
            metadata (Optional[Dict[str, Any]], optional): Structured metadata tags. Defaults to None.

        Returns:
            MemoryResponse: Response containing assigned memory ID.

        Raises:
            AIOSKernelError: If kernel request fails.
        """
        return create_memory(
            agent_name=self.agent_name,
            content=content,
            metadata=metadata,
            base_url=self.base_url,
            socket_path=self.socket_path,
        )

    def recall(
        self,
        query: str,
        k: int = 5,
    ) -> MemoryResponse:
        """
        Convenience helper to search relevant memories from this agent's namespace.

        Args:
            query (str): Semantic search query string.
            k (int, optional): Maximum number of ranked results to return. Defaults to 5.

        Returns:
            MemoryResponse: Response containing ranked matching memories.

        Raises:
            MemoryFeatureUnimplemented: Always -- `search_memories` has no real AIOS
                kernel equivalent (ARCH.6/MEM.6: no vector/semantic memory) and is
                intentionally unimplemented rather than faked, in both the real-kernel
                and HTTP-mock paths. `BaseAgent` subclasses (including this class's own
                docstring example) must not build `run()` around `recall()` until a
                real semantic memory backend exists.
        """
        return search_memories(
            agent_name=self.agent_name,
            query=query,
            k=k,
            base_url=self.base_url,
            socket_path=self.socket_path,
        )
