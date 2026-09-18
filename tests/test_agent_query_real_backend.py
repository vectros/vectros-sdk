"""
SDK.8 -- proves `AgentQuery` really drives two different backends
(a live Ollama model, then real kernel-backed memory storage) through one
composed chain, against a real, standing `aiosctl serve-kernel` and a real
Ollama daemon. No mocks: mirrors `test_real_kernel_client.py`'s own harness
rather than importing it, matching this suite's existing convention of
self-contained test files.

This is the strongest "real, no mock" proof available for SDK.8 given the
protocol gap documented in `vectros_sdk/core/agent_query.py`'s module
docstring: there is no wire-level Composite/AgentQuery submission variant,
so what is proven here is that the SDK's own linear pipeline dispatches
each already-existing atomic Query for real, in order, with a real
response from one step threaded into the next.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vectros_sdk.client.client import AIOSClient  # noqa: E402
from vectros_sdk.core.agent_query import (  # noqa: E402
    AgentQueryError,
    llm_step,
    memory_create_step,
)

REPO_ROOT = Path(__file__).resolve().parents[1].parent
CARGO_AVAILABLE = shutil.which("cargo") is not None
OLLAMA_MODEL = "gemma4:e4b"


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
class TestAgentQueryRealBackend(unittest.TestCase):
    """Every test here talks to a real `aiosd` kernel and a real Ollama
    daemon, not a mock."""

    @classmethod
    def setUpClass(cls) -> None:
        pid = os.getpid()
        cls.agent_id = f"agent_sdk8_{pid}"
        cls.socket_path = Path(f"/tmp/aios-py-sdk8-{pid}.sock")
        cls.telemetry_dir = Path(f"/tmp/aios-py-sdk8-{pid}-telemetry")
        cls.socket_path.unlink(missing_ok=True)
        shutil.rmtree(cls.telemetry_dir, ignore_errors=True)
        cls.telemetry_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["AIOS_AGENT_ID"] = cls.agent_id
        env["AIOS_RUN_ID"] = f"run_sdk8_{pid}"
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
                OLLAMA_MODEL,
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
        # tenant_permissions() namespaces every grant by
        # agent_id.trim_start_matches("agent_") -- the client must address
        # resources under that same suffix.
        return self.agent_id[len("agent_") :]

    def test_ask_the_real_model_then_really_store_its_answer_in_one_composed_chain(
        self,
    ) -> None:
        client = AIOSClient(agent_name=self._agent_suffix(), socket_path=str(self.socket_path))

        chain = llm_step(
            "ask",
            [{"role": "user", "content": "Reply with exactly the single word: PASS"}],
            llms=[{"name": OLLAMA_MODEL}],
        ) | memory_create_step(
            "remember",
            lambda history: history[-1].response_message,
            metadata={"source": "agent_query_test"},
        )

        result = chain.run(client)

        self.assertEqual(result.step_names, ["ask", "remember"])
        llm_response = result[0]
        self.assertTrue(llm_response.finished)
        self.assertTrue(llm_response.response_message)

        memory_response = result.last
        self.assertTrue(memory_response.success)
        self.assertTrue(memory_response.memory_id)

        # Prove the second step really did thread the first step's real
        # output, not a placeholder: fetch it back from the real kernel.
        fetched = client.memory.get(memory_response.memory_id)
        self.assertEqual(fetched.content, llm_response.response_message)

    def test_running_without_socket_path_refuses_rather_than_silently_faking_composite_execution(
        self,
    ) -> None:
        client = AIOSClient(agent_name=self._agent_suffix())
        chain = memory_create_step("remember", "unused")
        with self.assertRaises(AgentQueryError):
            chain.run(client)


if __name__ == "__main__":
    unittest.main()
