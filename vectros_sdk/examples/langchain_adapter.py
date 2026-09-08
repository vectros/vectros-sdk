"""
LangChain adapter for AIOS kernel.

Provides AIOSLLM (BaseLLM) and AIOSTool (BaseTool) that submit Operations
to the real aiosd daemon via KernelClient and stream real events.

When Ollama is running and the daemon wires the Infer loop, AIOSLLM picks up
the artifact_id from the OperationSucceeded watch event. The artifact bytes
would then be fetched via a separate GetArtifact call (tracked as ARTIFACT-001).
Until that endpoint exists, the LLM returns the artifact_id so the caller knows
where the result is stored.
"""
from __future__ import annotations
import json
import time
import uuid
from typing import Any, Iterator, List, Optional

try:
    from langchain_core.language_models import BaseLLM
    from langchain_core.tools import BaseTool
    from langchain_core.outputs import LLMResult, Generation
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseLLM = object  # type: ignore[assignment,misc]
    BaseTool = object  # type: ignore[assignment,misc]
    LLMResult = dict  # type: ignore[assignment,misc]
    Generation = dict  # type: ignore[assignment,misc]
    _LANGCHAIN_AVAILABLE = False

from vectros_sdk.transport.kernel_client import KernelClient
from vectros_sdk.domain import OperationKind

_TERMINAL = frozenset(
    {"OperationSucceeded", "OperationFailed", "OperationCancelled", "OperationOutcomeUnknown"}
)


class AIOSLLM(BaseLLM):  # type: ignore[misc]
    """
    LangChain LLM that submits Infer Operations to the real AIOS daemon.

    The daemon dispatches admitted Infer ops to LocalOllamaAdapter and persists
    the output as an Artifact. The watch stream returns an artifact_id on success.

    Usage::

        llm = AIOSLLM(agent_id="agent:my-agent")
        result = llm.invoke("What is AIOS?")
        print(result)
    """
    agent_id: str = "agent:langchain"
    kernel_client: Any = None
    watch_timeout: float = 60.0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.kernel_client is None:
            self.kernel_client = KernelClient()

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:  # LLMResult
        generations: list[list[Any]] = []

        for prompt in prompts:
            instance_id = f"inst-{uuid.uuid4().hex[:8]}"
            run_id = f"run-{uuid.uuid4().hex[:8]}"
            op_id = f"op-{uuid.uuid4().hex[:8]}"

            # Submit the Infer operation with the real prompt as payload
            self.kernel_client.submit(
                agent_id=self.agent_id,
                instance_id=instance_id,
                run_id=run_id,
                operation_id=op_id,
                kind_int=int(OperationKind.INFER),
                payload=json.dumps({"prompt": prompt}),
            )

            # Watch for terminal event and capture artifact_id (FIX #12)
            response_text, artifact_id = self._await_result(run_id)
            if artifact_id:
                try:
                    response_text = self.kernel_client.get_artifact(artifact_id).decode("utf-8")
                except Exception as e:
                    response_text = f"[artifact fetch failed: {e}] {response_text}"

            if _LANGCHAIN_AVAILABLE:
                generations.append([Generation(text=response_text)])
            else:
                generations.append([{"text": response_text}])

        if _LANGCHAIN_AVAILABLE:
            return LLMResult(generations=generations)
        return {"generations": generations}

    def _await_result(self, run_id: str) -> tuple[str, Optional[str]]:
        """Stream watch events until terminal state. Return (text, artifact_id)."""
        start = time.monotonic()
        for event in self.kernel_client.watch(self.agent_id, run_id):
            event_type = event.get("event_type", "")
            artifact_id = event.get("artifact_id")

            if event_type == "OperationSucceeded":
                return "Operation completed successfully.", artifact_id
            if event_type == "OperationFailed":
                return "Operation failed in the daemon.", None
            if event_type in ("OperationCancelled", "OperationOutcomeUnknown"):
                return f"Operation ended with: {event_type}", None
            if event.get("error"):
                return f"Watch error: {event['error']}", None

            if time.monotonic() - start > self.watch_timeout:
                return "Watch timed out waiting for operation result.", None

        return "Watch stream ended without terminal event.", None

    @property
    def _llm_type(self) -> str:
        return "aios-kernel"


class AIOSTool(BaseTool):  # type: ignore[misc]
    """LangChain Tool that submits Operations to the AIOS daemon."""
    agent_id: str = "agent:langchain"
    name: str = "aios_tool"
    description: str = "Executes a tool operation via the AIOS kernel"
    kernel_client: Any = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.kernel_client is None:
            self.kernel_client = KernelClient()

    def _run(self, query: str, run_manager: Optional[Any] = None, **kwargs: Any) -> str:
        instance_id = f"inst-{uuid.uuid4().hex[:8]}"
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        op_id = f"op-{uuid.uuid4().hex[:8]}"

        self.kernel_client.submit(
            agent_id=self.agent_id,
            instance_id=instance_id,
            run_id=run_id,
            operation_id=op_id,
            kind_int=int(OperationKind.INVOKE_TOOL),
            payload=json.dumps({"query": query}),
        )
        start = time.monotonic()
        for event in self.kernel_client.watch(self.agent_id, run_id):
            if event.get("state") == "OperationSucceeded":
                artifact_id = event.get("artifact_id")
                if artifact_id:
                    return self.kernel_client.get_artifact(artifact_id).decode("utf-8")
                return "Tool succeeded but no artifact returned"
            if event.get("state") in ("OperationFailed", "OperationCancelled", "OperationOutcomeUnknown"):
                return f"Tool failed: {event.get('event_type')}"
            if time.monotonic() - start > 15.0:
                return "Tool timed out"
        return "Watch stream ended"

    async def _arun(self, query: str, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run; async dispatch is tracked as ASYNC-001")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("=== AIOSLLM demo ===")
    llm = AIOSLLM(agent_id="agent:langchain-demo")
    result = llm.invoke("What is the AIOS kernel?")
    print("Result:", result)

    print("\n=== AIOSTool demo ===")
    tool = AIOSTool(agent_id="agent:langchain-demo")
    out = tool.run("search for agent lifecycle")
    print("Tool output:", out)
