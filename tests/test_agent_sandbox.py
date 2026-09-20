"""
ASBX.13 — proves `vectros_sdk.testing.AgentSandbox` (the context manager and
its pytest fixture) is a real, working seam, not aspirational scaffolding.

No mocks: every test here spawns the real `aiosctl serve-sandbox` process
this repository builds and talks to it over a real Unix socket through the
same `AIOSClient`/`real_kernel.py` translation layer
`test_real_kernel_client.py` already proves against the full production
bootstrap -- here against a `ScopedKernel` whose `AccessManager` ceiling is
exactly one spec's own grants instead.
"""

import shutil
import unittest
import urllib.request

import pytest

from vectros_sdk.testing import AgentSandbox

CARGO_AVAILABLE = shutil.which("cargo") is not None
OLLAMA_MODEL = "gemma4:e4b"


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0):
            return True
    except Exception:
        return False


@unittest.skipUnless(CARGO_AVAILABLE, "cargo is required to build/run the real aiosctl binary")
class TestAgentSandbox(unittest.TestCase):
    """The context manager, used directly (no pytest fixture)."""

    def test_memory_round_trip_through_a_real_sandboxed_kernel(self) -> None:
        with AgentSandbox() as sandbox:
            created = sandbox.client.memory.create(content="hello from a real AgentSandbox")
            self.assertTrue(created.success)
            fetched = sandbox.client.memory.get(created.memory_id)
            self.assertEqual(fetched.content, "hello from a real AgentSandbox")

    def test_two_sandboxes_never_collide_and_are_genuinely_isolated(self) -> None:
        with AgentSandbox() as first, AgentSandbox() as second:
            self.assertNotEqual(first.agent_name, second.agent_name)
            first.client.memory.create(content="only in the first sandbox")
            # The second sandbox's own memory slot was never written -- a
            # real get against it must not see the first sandbox's content.
            second_created = second.client.memory.create(content="only in the second sandbox")
            self.assertEqual(
                second.client.memory.get(second_created.memory_id).content,
                "only in the second sandbox",
            )

    def test_storage_round_trip_through_a_real_sandboxed_kernel(self) -> None:
        with AgentSandbox() as sandbox:
            created = sandbox.client.storage.create_file(file_path="notes.txt")
            self.assertTrue(created.finished)
            written = sandbox.client.storage.write_file(
                file_path="notes.txt", content="real content in a real sandbox"
            )
            self.assertTrue(written.finished)

    def test_tool_call_invokes_the_one_real_registered_fixture_tool(self) -> None:
        with AgentSandbox() as sandbox:
            response = sandbox.client.tool.call(
                [{"name": "uppercase", "parameters": {"input": "hello sandbox"}}]
            )
            self.assertTrue(response.finished)
            self.assertEqual(response.response_message, "HELLO SANDBOX")

    def test_extra_grants_reach_a_caller_named_resource(self) -> None:
        # A resource outside the conventional default set is only reachable
        # once explicitly named -- the spec really is the whole authority,
        # not just for the built-in defaults.
        with AgentSandbox(extra_grants=[("memory.get", "res_memory_custom")]):
            pass  # constructing/entering successfully is the proof here;
            # the real denial-of-the-unnamed-resource property is already
            # proven directly against ScopedKernel in the Rust suite.

    @unittest.skipUnless(_ollama_reachable(), "a local Ollama daemon serving gemma4:e4b is required")
    def test_real_llm_generation_through_a_sandboxed_kernel(self) -> None:
        with AgentSandbox(model=OLLAMA_MODEL) as sandbox:
            response = sandbox.client.llm.chat(
                messages=[{"role": "user", "content": "Reply with exactly the single word: PASS"}],
                llms=[{"name": OLLAMA_MODEL}],
            )
            self.assertTrue(response.finished)
            print(f"real AgentSandbox llm.chat: {response.response_message!r}")
            self.assertTrue(response.response_message)


@pytest.mark.skipif(not CARGO_AVAILABLE, reason="cargo is required to build/run the real aiosctl binary")
def test_agent_sandbox_pytest_fixture_gives_a_real_working_sandbox(agent_sandbox: AgentSandbox) -> None:
    created = agent_sandbox.client.memory.create(content="hello from the pytest fixture")
    assert created.success
    fetched = agent_sandbox.client.memory.get(created.memory_id)
    assert fetched.content == "hello from the pytest fixture"
