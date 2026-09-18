"""
SDK.10 -- proves `BaseAgent` (`vectros_sdk/agent/base.py`) can actually reach
a real, standing kernel server, not just the dead HTTP mock endpoint.

Real gap found and fixed by this task, not just a test gap: `BaseAgent.chat`/
`.remember`/`.recall` never accepted or threaded a `socket_path` at all, so
every `BaseAgent` subclass -- the framework class this SDK's own docstring
tells real agent authors to build on -- could only ever speak to
`http://localhost:8000/query`, which nothing in this workspace serves. That
made `BaseAgent` unusable against the real kernel in any configuration,
silently, with only mock-based tests (`test_agent_api.py`,
`test_brutal_agent.py`) standing in as if it worked. Fixed in
`agent/base.py` by adding a `socket_path` constructor parameter threaded
into every helper's underlying `llm_chat`/`create_memory`/`search_memories`
call, matching the pattern `AIOSClient`'s sub-clients already use.

No mocks: starts the actual `aiosctl serve-kernel` process built from this
repository and talks to it over a real Unix socket, mirroring
`test_real_kernel_client.py`'s own harness rather than importing it.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vectros_sdk import BaseAgent  # noqa: E402
from vectros_sdk.memory.api import MemoryFeatureUnimplemented, delete_memory  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1].parent
CARGO_AVAILABLE = shutil.which("cargo") is not None


class RealMathAgent(BaseAgent):
    """A concrete agent whose `run()` only calls the two helpers that have a
    real backend today (`chat`, `remember`) -- `recall` is proven separately
    below to raise for real, per MEM.6/ARCH.6, rather than built into a
    pipeline that could never complete."""

    def run(self, task: Union[str, Dict[str, Any]]) -> Any:
        prompt = task if isinstance(task, str) else task.get("prompt", "")
        llm_res = self.chat(prompt)
        self.remember(f"Solved {prompt}: {llm_res.response_message}")
        return llm_res.response_message


def _wait_for_socket(path: Path, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.5)
                    probe.connect(str(path))
                return
            except OSError:
                pass
        time.sleep(0.1)
    raise TimeoutError(f"kernel server never bound {path}")


@unittest.skipUnless(CARGO_AVAILABLE, "cargo is required to build/run the real kernel server")
class TestBaseAgentRealBackend(unittest.TestCase):
    """Every test here talks to a real `aiosd` kernel, not a mock."""

    OLLAMA_MODEL = "gemma4:e4b"

    @classmethod
    def setUpClass(cls) -> None:
        pid = os.getpid()
        cls.agent_id = f"agent_sdk10_{pid}"
        cls.socket_path = Path(f"/tmp/aios-py-sdk10-{pid}.sock")
        cls.telemetry_dir = Path(f"/tmp/aios-py-sdk10-{pid}-telemetry")
        cls.socket_path.unlink(missing_ok=True)
        shutil.rmtree(cls.telemetry_dir, ignore_errors=True)
        cls.telemetry_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["AIOS_AGENT_ID"] = cls.agent_id
        env["AIOS_RUN_ID"] = f"run_sdk10_{pid}"
        cls.server = subprocess.Popen(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "aiosctl",
                "--",
                "serve-kernel",
                str(cls.socket_path),
                str(cls.telemetry_dir),
                "--ollama-model",
                cls.OLLAMA_MODEL,
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_socket(cls.socket_path)
        except TimeoutError:
            cls.server.terminate()
            output = cls.server.stdout.read() if cls.server.stdout else ""
            cls.server.wait(timeout=5)
            raise AssertionError(f"server never started; output:\n{output}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.terminate()
        try:
            cls.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.server.kill()
        cls.socket_path.unlink(missing_ok=True)
        shutil.rmtree(cls.telemetry_dir, ignore_errors=True)

    def _agent_suffix(self) -> str:
        return self.agent_id[len("agent_") :]

    def _agent(self) -> RealMathAgent:
        return RealMathAgent(
            agent_name=self._agent_suffix(),
            system_prompt="You are terse.",
            llm_config=[{"name": self.OLLAMA_MODEL}],
            socket_path=str(self.socket_path),
        )

    # Numbered rather than left to unittest's default alphabetical ordering:
    # the bootstrapped tenant grants exactly one Memory key per agent
    # (ARCH.46), and every test here shares one class-level agent_id, so a
    # test that leaves a memory behind must run before the one test that
    # depends on that slot being free again.

    def test_01_chat_reaches_the_real_model_through_baseagent(self) -> None:
        agent = self._agent()
        response = agent.chat("Reply with exactly the single word: PASS")
        self.assertTrue(response.finished)
        self.assertTrue(response.response_message)

    def test_02_recall_still_has_no_real_equivalent_and_is_refused_not_faked(self) -> None:
        agent = self._agent()
        with self.assertRaises(MemoryFeatureUnimplemented):
            agent.recall("anything")

    def test_03_remember_really_stores_a_memory_through_baseagent(self) -> None:
        agent = self._agent()
        response = agent.remember("BaseAgent really wrote this", metadata={"via": "sdk10"})
        self.assertTrue(response.success)
        self.assertTrue(response.memory_id)

        # Free the agent's one Memory slot (ARCH.46) so test_04 can create
        # its own memory as part of a real run() pipeline.
        deleted = delete_memory(
            agent_name=self._agent_suffix(),
            memory_id=response.memory_id,
            socket_path=str(self.socket_path),
        )
        self.assertTrue(deleted.success)

    def test_04_run_completes_a_real_chat_and_remember_pipeline_end_to_end(self) -> None:
        agent = self._agent()
        result = agent.run("Reply with exactly the single word: PASS")
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
