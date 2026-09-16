"""
M2/SDK.3 — proves the high-level `AIOSClient` (and the module-level
`llm`/`memory`/`storage`/`tool` API functions it wraps) can drive a real,
standing kernel server (`aiosctl serve-kernel`) end to end, via
`vectros_sdk.client.real_kernel`, instead of the dead HTTP mock path.

No mocks: starts the actual `aiosctl serve-kernel` process built from this
repository and talks to it over a real Unix socket, exactly like
`test_execution_protocol_real_server.py` (this file mirrors that harness
rather than importing it, matching this suite's existing convention of
self-contained test files).
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
from vectros_sdk.client.real_kernel import RealBackendUnsupported  # noqa: E402
from vectros_sdk.memory.api import create_memory, delete_memory, get_memory, update_memory  # noqa: E402
from vectros_sdk.storage.api import create_file, write_file  # noqa: E402
from vectros_sdk.tool.api import call_tool  # noqa: E402

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
class TestRealKernelClient(unittest.TestCase):
    """Every test here talks to a real `aiosd` kernel, not a mock."""

    @classmethod
    def setUpClass(cls) -> None:
        pid = os.getpid()
        cls.agent_id = f"agent_sdk3_{pid}"
        cls.socket_path = Path(f"/tmp/aios-py-sdk3-{pid}.sock")
        cls.telemetry_dir = Path(f"/tmp/aios-py-sdk3-{pid}-telemetry")
        cls.socket_path.unlink(missing_ok=True)
        shutil.rmtree(cls.telemetry_dir, ignore_errors=True)
        cls.telemetry_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["AIOS_AGENT_ID"] = cls.agent_id
        env["AIOS_RUN_ID"] = f"run_sdk3_{pid}"
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
        # agent_id.trim_start_matches("agent_") — the client must address
        # resources under that same suffix.
        return self.agent_id[len("agent_"):]

    def test_memory_create_get_update_delete_round_trip_through_the_real_kernel(self) -> None:
        agent = self._agent_suffix()
        created = create_memory(
            agent_name=agent, content="hello real memory", socket_path=str(self.socket_path)
        )
        self.assertTrue(created.success)
        memory_id = created.memory_id
        self.assertTrue(memory_id)

        fetched = get_memory(
            agent_name=agent, memory_id=memory_id, socket_path=str(self.socket_path)
        )
        self.assertEqual(fetched.content, "hello real memory")

        updated = update_memory(
            agent_name=agent,
            memory_id=memory_id,
            content="updated real memory",
            socket_path=str(self.socket_path),
        )
        self.assertTrue(updated.success)

        refetched = get_memory(
            agent_name=agent, memory_id=memory_id, socket_path=str(self.socket_path)
        )
        self.assertEqual(refetched.content, "updated real memory")

        deleted = delete_memory(
            agent_name=agent, memory_id=memory_id, socket_path=str(self.socket_path)
        )
        self.assertTrue(deleted.success)

    def test_memory_search_has_no_real_equivalent_and_is_refused_not_faked(self) -> None:
        client = AIOSClient(agent_name=self._agent_suffix(), socket_path=str(self.socket_path))
        with self.assertRaises(RealBackendUnsupported):
            client.memory.search("anything")

    def test_storage_create_file_and_write_file_round_trip_through_the_real_kernel(self) -> None:
        agent = self._agent_suffix()
        created = create_file(
            agent_name=agent, file_path="notes.txt", socket_path=str(self.socket_path)
        )
        self.assertTrue(created.finished)

        written = write_file(
            agent_name=agent,
            file_path="notes.txt",
            content="real content on a real filesystem-shaped store",
            socket_path=str(self.socket_path),
        )
        self.assertTrue(written.finished)

    def test_storage_retrieve_file_has_no_real_equivalent_and_is_refused_not_faked(self) -> None:
        client = AIOSClient(agent_name=self._agent_suffix(), socket_path=str(self.socket_path))
        with self.assertRaises(RealBackendUnsupported):
            client.storage.retrieve_file("anything", n=1)

    def test_tool_call_invokes_the_one_real_registered_fixture_tool(self) -> None:
        agent = self._agent_suffix()
        response = call_tool(
            agent_name=agent,
            tool_calls=[{"name": "uppercase", "parameters": {"input": "hello real tool"}}],
            socket_path=str(self.socket_path),
        )
        self.assertTrue(response.finished)
        self.assertEqual(response.response_message, "HELLO REAL TOOL")

    def test_tool_call_for_an_unregistered_tool_is_refused_not_faked(self) -> None:
        client = AIOSClient(agent_name=self._agent_suffix(), socket_path=str(self.socket_path))
        with self.assertRaises(RealBackendUnsupported):
            client.tool.call([{"name": "not_a_real_tool", "parameters": {}}])

    def test_llm_chat_completes_a_real_generation_end_to_end_through_the_client(self) -> None:
        client = AIOSClient(agent_name=self._agent_suffix(), socket_path=str(self.socket_path))
        response = client.llm.chat(
            messages=[
                {"role": "user", "content": "Reply with exactly the single word: PASS"}
            ],
            llms=[{"name": OLLAMA_MODEL}],
        )
        self.assertTrue(response.finished)
        print(f"real end-to-end llm.chat via AIOSClient(socket_path=...): {response.response_message!r}")
        self.assertTrue(response.response_message)

    def test_llm_tool_use_has_no_real_equivalent_yet_and_is_refused_not_faked(self) -> None:
        client = AIOSClient(agent_name=self._agent_suffix(), socket_path=str(self.socket_path))
        with self.assertRaises(RealBackendUnsupported):
            client.llm.chat_tool(
                messages=[{"role": "user", "content": "call a tool"}],
                tools=[{"name": "uppercase"}],
                llms=[{"name": OLLAMA_MODEL}],
            )

    def test_post_has_no_real_equivalent_and_is_refused_not_faked(self) -> None:
        client = AIOSClient(agent_name=self._agent_suffix(), socket_path=str(self.socket_path))
        with self.assertRaises(RealBackendUnsupported):
            from vectros_sdk.client.real_kernel import execute_real
            from vectros_sdk.post.models import PostQuery

            execute_real(
                PostQuery(agent_name=self._agent_suffix(), action_type="send"),
                str(self.socket_path),
            )


if __name__ == "__main__":
    unittest.main()
