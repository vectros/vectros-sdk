"""
SDK.8 -- pure, deterministic tests for `AgentQuery` composition mechanics
(step ordering, `.then()`/`|` associativity, history threading, and the two
structural refusals). No backend, no live model: this only proves the
composition math is correct, the same way `test_terminal_dispatch.py`
proves `classify()` is correct without a live model.

A minimal stub stands in for `AIOSClient` here -- not `unittest.mock` --
because what is under test is `AgentQuery`'s own sequencing logic, not
whether any backend call succeeds; the real backend path is proven
end-to-end, no stub, in `test_agent_query_real_backend.py`.
"""

import os
import sys
import unittest
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vectros_sdk.core.agent_query import (  # noqa: E402
    AgentQuery,
    AgentQueryError,
    llm_step,
    memory_create_step,
    storage_write_step,
    tool_call_step,
)


class _StubSubClient:
    def __init__(self, calls: List[Any], make_response: Any) -> None:
        self._calls = calls
        self._make_response = make_response

    def _record(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self._calls.append((name, args, kwargs))
        return self._make_response(name, args, kwargs)


class StubClient:
    """A minimal stand-in exposing exactly the surface `AgentQuery` steps
    call: `.llm.chat`, `.memory.create`, `.storage.write_file`,
    `.tool.call`, plus `socket_path`. Every call is recorded verbatim and
    answered by a caller-supplied function, so tests assert on exactly what
    was dispatched and in what order."""

    def __init__(self, make_response=lambda name, args, kwargs: {"name": name}) -> None:
        self.socket_path: Optional[str] = "/tmp/does-not-need-to-exist-for-this-test.sock"
        self.calls: List[Any] = []
        sub = _StubSubClient(self.calls, make_response)

        class _LLM:
            def chat(_self, messages, llms=None):
                return sub._record("llm.chat", messages=messages, llms=llms)

        class _Memory:
            def create(_self, content, metadata=None):
                return sub._record("memory.create", content=content, metadata=metadata)

        class _Storage:
            def write_file(_self, file_path, content):
                return sub._record("storage.write_file", file_path=file_path, content=content)

        class _Tool:
            def call(_self, tool_calls):
                return sub._record("tool.call", tool_calls=tool_calls)

        self.llm = _LLM()
        self.memory = _Memory()
        self.storage = _Storage()
        self.tool = _Tool()


class TestAgentQueryComposition(unittest.TestCase):
    def test_then_and_pipe_produce_the_same_ordered_chain(self) -> None:
        step_a = llm_step("ask", [{"role": "user", "content": "hi"}])
        step_b = memory_create_step("remember", "static content")

        via_then = step_a.then(step_b)
        via_pipe = step_a | step_b

        self.assertEqual([s.name for s in via_then.steps], ["ask", "remember"])
        self.assertEqual([s.name for s in via_pipe.steps], ["ask", "remember"])

    def test_chaining_three_steps_left_to_right_preserves_order(self) -> None:
        chain = (
            llm_step("ask", [{"role": "user", "content": "hi"}])
            | memory_create_step("remember", "x")
            | storage_write_step("save", "notes.txt", "y")
        )
        self.assertEqual([s.name for s in chain.steps], ["ask", "remember", "save"])

    def test_composing_two_agent_queries_flattens_into_one_chain(self) -> None:
        left = AgentQuery([llm_step("ask", [{"role": "user", "content": "hi"}])])
        right = AgentQuery([memory_create_step("remember", "x")])
        combined = left | right
        self.assertEqual([s.name for s in combined.steps], ["ask", "remember"])

    def test_run_dispatches_steps_in_order_through_the_client(self) -> None:
        client = StubClient()
        chain = llm_step("ask", [{"role": "user", "content": "hi"}]) | memory_create_step(
            "remember", "static content"
        )
        result = chain.run(client)
        self.assertEqual([name for name, _, _ in client.calls], ["llm.chat", "memory.create"])
        self.assertEqual(result.step_names, ["ask", "remember"])
        self.assertEqual(len(result), 2)

    def test_a_later_step_can_thread_the_real_response_of_an_earlier_one(self) -> None:
        class FakeLLMResponse:
            def __init__(self, text: str) -> None:
                self.response_message = text

        def make_response(name: str, args: Any, kwargs: Dict[str, Any]) -> Any:
            if name == "llm.chat":
                return FakeLLMResponse("the model's real answer")
            return {"name": name, "kwargs": kwargs}

        client = StubClient(make_response=make_response)
        chain = llm_step("ask", [{"role": "user", "content": "hi"}]) | memory_create_step(
            "remember", lambda history: history[-1].response_message
        )
        chain.run(client)

        recorded_memory_call = next(c for c in client.calls if c[0] == "memory.create")
        self.assertEqual(recorded_memory_call[2]["content"], "the model's real answer")

    def test_tool_call_step_can_thread_history_into_its_arguments(self) -> None:
        def make_response(name: str, args: Any, kwargs: Dict[str, Any]) -> Any:
            return {"name": name}

        client = StubClient(make_response=make_response)
        chain = memory_create_step("remember", "seed") | tool_call_step(
            "invoke",
            lambda history: [{"name": "echo", "parameters": {"seen": len(history)}}],
        )
        chain.run(client)

        recorded_tool_call = next(c for c in client.calls if c[0] == "tool.call")
        self.assertEqual(
            recorded_tool_call[2]["tool_calls"], [{"name": "echo", "parameters": {"seen": 1}}]
        )

    def test_running_an_empty_agent_query_refuses_rather_than_doing_nothing_silently(
        self,
    ) -> None:
        with self.assertRaises(AgentQueryError):
            AgentQuery().run(StubClient())

    def test_running_without_socket_path_refuses_since_there_is_no_http_mock_equivalent(
        self,
    ) -> None:
        client = StubClient()
        client.socket_path = None
        chain = memory_create_step("remember", "x")
        with self.assertRaises(AgentQueryError):
            chain.run(client)

    def test_result_last_raises_on_an_empty_result_rather_than_indexing_a_missing_step(
        self,
    ) -> None:
        from vectros_sdk.core.agent_query import AgentQueryResult

        with self.assertRaises(AgentQueryError):
            AgentQueryResult(step_names=[], responses=[]).last


if __name__ == "__main__":
    unittest.main()
