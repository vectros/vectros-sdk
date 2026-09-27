"""
VAL.10 -- the canonical acceptance workload: the SDK's three example agents
(`ResearchAnalystAgent`, `DataArchivistAgent`, `TaskCoordinatorAgent`)
running **concurrently**, each in its own real sandboxed kernel
(`vectros_sdk.testing.AgentSandbox` -> a real `aiosctl serve-sandbox`
process), against the same real local model. No mocks, no
`FakeAIOSKernelServer`.

**Honest scope (user-approved, not a silent narrowing):** each agent runs
only its real-capable subset. ASBX.16 (`test_examples_real_backend.py`)
established what is real:

- `ResearchAnalystAgent.conduct_research` -- real LLM synthesis plus a real
  memory write and read-back. Run without `numbers`, because its
  `math_evaluator` tool has no real kernel equivalent (the kernel registers
  only the reviewed `uppercase` fixture).
- `DataArchivistAgent.archive_summary` (new) -- the real flat storage
  create+write. `setup_project_storage` needs mounts, directories, content
  search, version rollback and sharing that the real Storage Manager does
  not have (**known limitation: no directory tree / versioning / sharing**).
- `TaskCoordinatorAgent.plan_and_record` (new) -- a real model call plus a
  real memory write and read-back. `orchestrate_pipeline` is built entirely
  on Post pub/sub, which has **no real kernel equivalent at all** (**known
  limitation: no pub/sub IPC**).

Isolation is proven the way MA.4 proves it: three independent kernels share
no state (separate processes and sockets), every agent's result routes back
to itself (each carries its own marker end to end), and per-agent
authorization holds -- from its own kernel socket, the archivist tries to
write the research agent's memory resource, which its grants do not cover,
and the kernel refuses it (`permission_denied`) while the other agents' real
work proceeds at the same time.
"""

import shutil
import threading
import unittest
import urllib.request

from vectros_sdk.examples.agents.archivist_agent import DataArchivistAgent
from vectros_sdk.examples.agents.coordinator_agent import TaskCoordinatorAgent
from vectros_sdk.examples.agents.research_agent import ResearchAnalystAgent
from vectros_sdk.testing import AgentSandbox
from vectros_sdk.transport.execution_protocol import (
    ExecutionProtocolClient,
    ExecutionProtocolError,
    memory_put,
)

CARGO_AVAILABLE = shutil.which("cargo") is not None
OLLAMA_MODEL = "gemma4:e4b"


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1.0):
            return True
    except Exception:
        return False


@unittest.skipUnless(CARGO_AVAILABLE, "cargo is required to build/run the real aiosctl binary")
@unittest.skipUnless(_ollama_reachable(), "a local Ollama daemon serving gemma4:e4b is required")
class TestCanonicalAcceptanceWorkload(unittest.TestCase):
    def test_three_example_agents_run_concurrently_in_isolated_real_sandboxes(self) -> None:
        barrier = threading.Barrier(3)
        results: dict = {}
        errors: dict = {}

        def research() -> None:
            with AgentSandbox(agent_name="val10research", model=OLLAMA_MODEL) as sandbox:
                agent = ResearchAnalystAgent(name="val10_research", client=sandbox.client)
                barrier.wait()
                results["research"] = agent.conduct_research(topic="VAL10-RESEARCH lighthouses")

        def archivist() -> None:
            # Deliberately no model grant: this agent's authorization differs.
            with AgentSandbox(agent_name="val10archive") as sandbox:
                agent = DataArchivistAgent(name="val10_archive", client=sandbox.client)
                barrier.wait()
                results["archive"] = agent.archive_summary(
                    file_path="val10.txt", content="VAL10-ARCHIVE summary"
                )
                # Per-agent authorization: another agent's resource, on this
                # agent's own real kernel socket.
                with ExecutionProtocolClient(sandbox.client.socket_path) as raw:
                    try:
                        raw.execute(
                            "req_val10_cross",
                            memory_put("res_memory_val10research", "must be refused"),
                        )
                        results["archive_cross"] = "ALLOWED"
                    except ExecutionProtocolError as error:
                        results["archive_cross"] = error.code

        def coordinator() -> None:
            with AgentSandbox(agent_name="val10coord", model=OLLAMA_MODEL) as sandbox:
                agent = TaskCoordinatorAgent(name="val10_coord", client=sandbox.client)
                barrier.wait()
                results["coord"] = agent.plan_and_record(mission="VAL10-COORD mission")

        def run(name, target) -> threading.Thread:
            def wrapped() -> None:
                try:
                    target()
                except Exception as error:  # surfaced below, never swallowed
                    errors[name] = error
                    barrier.abort()

            thread = threading.Thread(target=wrapped, name=name)
            thread.start()
            return thread

        threads = [run("research", research), run("archive", archivist), run("coord", coordinator)]
        for thread in threads:
            thread.join(timeout=300)
        self.assertFalse(errors, f"agent failures: {errors!r}")
        print("VAL10 results:", results)

        research_result = results["research"]
        self.assertTrue(research_result["synthesis"])
        self.assertTrue(research_result["saved_memory_verified"])
        self.assertEqual(research_result["saved_memory_id"], "res_memory_val10research")

        archive_result = results["archive"]
        self.assertTrue(archive_result["created"] and archive_result["written"])
        self.assertEqual(
            results["archive_cross"],
            "permission_denied",
            "per-agent authorization: the archivist must not write another agent's memory",
        )

        coord_result = results["coord"]
        self.assertTrue(coord_result["plan"])
        self.assertEqual(coord_result["saved_memory_id"], "res_memory_val10coord")
        self.assertEqual(
            coord_result["recalled"],
            coord_result["plan"],
            "the coordinator's own memory must return exactly its own plan",
        )


if __name__ == "__main__":
    unittest.main()
