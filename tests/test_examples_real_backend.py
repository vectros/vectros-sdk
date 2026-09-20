"""
ASBX.16 -- real, sandboxed-kernel proof for the SDK example agents
(`vectros_sdk.examples.agents.*`).

`test_examples.py` exercises these same agents, but only against
`FakeAIOSKernelServer` -- a self-contained, in-process fake (see
`vectros_sdk.examples.server.mock_kernel_server`'s own docstring) that never
reaches the real kernel. This file uses `vectros_sdk.testing.AgentSandbox`
(ASBX.13) instead: a real `aiosctl serve-sandbox` process, talked to over a
real Unix socket through the same `AIOSClient`/`real_kernel.py` translation
layer `test_real_kernel_client.py` already proves end to end. No mocks.

**Honest scope, found by driving each agent against the real backend rather
than assumed from reading the code:**

- `ResearchAnalystAgent.conduct_research` is mostly real: its LLM synthesis
  (`client.llm.chat_json`) and its plain key/value memory save+read-back
  (`client.memory.create`/`.get`) both have real kernel equivalents and are
  exercised for real here. Its tool-call step is not: `client.tool.call`
  against a real kernel only ever runs the one reviewed fixture tool
  (`"uppercase"`) -- `real_kernel.py`'s own `_KNOWN_TOOLS` set -- so calling
  it with `"math_evaluator"` (a pure local Python class, never registered
  with any kernel) genuinely raises `RealBackendUnsupported`, proven below
  rather than silently skipped.
- `DataArchivistAgent.setup_project_storage` is almost entirely unreal: only
  its `create_file`/`write_file` steps have a real equivalent (a single
  flat (collection, object) slot per agent); `mount`, `create_dir`,
  `retrieve_file`, `rollback_file`, and `share_file` all raise
  `RealBackendUnsupported` (the real Storage Manager has no directory tree,
  content search, versioning, or sharing at all) -- so the method as a
  whole cannot complete against a real backend. Proven below: the two real
  steps round-trip correctly in isolation, and the full method fails loudly
  at its very first unreal step (`mount`) rather than partially succeeding.
- `TaskCoordinatorAgent.orchestrate_pipeline` is entirely unreal: every one
  of its six steps is a Post API call, and Post has no real kernel
  equivalent at all (the real IPC primitive is point-to-point between two
  known agents with an existing grant, not this API's pub/sub shape) --
  proven below to fail loudly at its very first step.

A real, previously-unreachable gap was found and fixed while writing this
file: `vectros_sdk.post.api`'s five functions (`send_post`/`receive_posts`/
`broadcast_post`/`publish_to_topic`/`subscribe_topic`) and `PostClient`'s
matching methods never accepted or forwarded `socket_path` at all, so a
caller holding a real `AIOSClient(socket_path=...)` calling `client.post.*`
silently fell through to the dead HTTP mock endpoint (a connection error)
instead of `real_kernel.py`'s own clean, documented `RealBackendUnsupported`
-- the same class of gap SDK.10 fixed for `BaseAgent.chat/remember/recall`.
"""

import shutil
import unittest
import urllib.request

from vectros_sdk.client.real_kernel import RealBackendUnsupported
from vectros_sdk.examples.agents.archivist_agent import DataArchivistAgent
from vectros_sdk.examples.agents.coordinator_agent import TaskCoordinatorAgent
from vectros_sdk.examples.agents.research_agent import ResearchAnalystAgent
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
class TestResearchAnalystAgentRealBackend(unittest.TestCase):
    @unittest.skipUnless(
        _ollama_reachable(), "a local Ollama daemon serving gemma4:e4b is required"
    )
    def test_conduct_research_reaches_real_llm_and_memory_through_a_sandboxed_kernel(self) -> None:
        # Observed live, repeatedly: this occasionally hits a genuine
        # transient Ollama "operation_failed" (the same class of real,
        # infrastructure-level flakiness ASBX.10's own evidence documents,
        # reproducing clean immediately after) -- surfaced as a real test
        # failure when it happens, never retried/hidden.
        with AgentSandbox(model=OLLAMA_MODEL) as sandbox:
            agent = ResearchAnalystAgent(name="real_research_analyst", client=sandbox.client)
            # numbers omitted: the tool-call step has no real kernel
            # equivalent for math_evaluator -- see the refusal test below.
            result = agent.conduct_research(topic="Real sandboxed kernel proof")
            self.assertEqual(result["agent"], "real_research_analyst")
            self.assertTrue(result["synthesis"])
            self.assertIsNotNone(result["saved_memory_id"])
            self.assertTrue(result["saved_memory_verified"])

    def test_math_evaluator_tool_call_has_no_real_kernel_equivalent(self) -> None:
        # This is exactly the call conduct_research(numbers=[...]) makes --
        # proving it fails loudly against a real backend, not silently.
        with AgentSandbox() as sandbox:
            with self.assertRaises(RealBackendUnsupported):
                sandbox.client.tool.call(
                    tool_calls=[
                        {
                            "name": "math_evaluator",
                            "parameters": {"operation": "average", "numbers": [1, 2, 3]},
                        }
                    ]
                )


@unittest.skipUnless(CARGO_AVAILABLE, "cargo is required to build/run the real aiosctl binary")
class TestDataArchivistAgentRealBackend(unittest.TestCase):
    def test_the_two_real_backed_storage_steps_round_trip_through_a_sandboxed_kernel(self) -> None:
        with AgentSandbox() as sandbox:
            created = sandbox.client.storage.create_file(file_path="summary.txt")
            self.assertTrue(created.finished)
            written = sandbox.client.storage.write_file(
                file_path="summary.txt", content="real content in a real sandbox"
            )
            self.assertTrue(written.finished)

    def test_setup_project_storage_fails_loudly_at_its_first_real_gap(self) -> None:
        # setup_project_storage()'s very first call is client.storage.mount()
        # -- proving the workflow, as written, cannot silently succeed
        # against a real backend; it fails exactly where the real gap is,
        # not partway through with a confusing error.
        with AgentSandbox() as sandbox:
            agent = DataArchivistAgent(name="real_data_archivist", client=sandbox.client)
            with self.assertRaises(RealBackendUnsupported):
                agent.setup_project_storage(project_name="alpha", initial_content="content")


@unittest.skipUnless(CARGO_AVAILABLE, "cargo is required to build/run the real aiosctl binary")
class TestTaskCoordinatorAgentRealBackend(unittest.TestCase):
    def test_orchestrate_pipeline_fails_loudly_post_has_no_real_equivalent(self) -> None:
        # Every one of orchestrate_pipeline's six steps is a Post call;
        # this proves the very first one (subscribe) refuses immediately
        # and honestly, rather than the client silently falling through to
        # the dead HTTP mock endpoint (the gap fixed in post/api.py above).
        with AgentSandbox() as sandbox:
            agent = TaskCoordinatorAgent(name="real_task_coordinator", client=sandbox.client)
            with self.assertRaises(RealBackendUnsupported):
                agent.orchestrate_pipeline(mission="Real sandboxed kernel proof", worker_names=["w"])
