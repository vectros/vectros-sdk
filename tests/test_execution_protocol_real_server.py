"""
M2 — proves the Python SDK can speak the *real* AIOS protocol against a real,
standing kernel server. No mocks: this starts the actual `aiosctl serve-kernel`
process (built from this repository) and talks to it over a real Unix socket
with `ExecutionProtocolClient`.

This is deliberately not a `unittest.mock`-based test — see
docs/architecture.md SDK.10 for why the rest of this test suite's mock-only
coverage is a tracked gap, not a pattern to keep extending.
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

from vectros_sdk.transport.execution_protocol import (  # noqa: E402
    ExecutionProtocolClient,
    ExecutionProtocolError,
    context_append,
    context_create,
    context_prepare,
    memory_get,
    memory_put,
    model_generate,
)

REPO_ROOT = Path(__file__).resolve().parents[1].parent
CARGO_AVAILABLE = shutil.which("cargo") is not None


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
class TestRealExecutionProtocol(unittest.TestCase):
    """Every test here talks to a real `aiosd` kernel, not a mock."""

    def setUp(self) -> None:
        pid = os.getpid()
        self.socket_path = Path(f"/tmp/aios-py-m2-{pid}.sock")
        self.telemetry_dir = Path(f"/tmp/aios-py-m2-{pid}-telemetry")
        self.socket_path.unlink(missing_ok=True)
        shutil.rmtree(self.telemetry_dir, ignore_errors=True)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        self.server = subprocess.Popen(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "aiosctl",
                "--",
                "serve-kernel",
                str(self.socket_path),
                str(self.telemetry_dir),
                "--ollama-model",
                "gemma4:e4b",
            ],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_socket(self.socket_path)
        except TimeoutError:
            self.server.terminate()
            output = self.server.stdout.read() if self.server.stdout else ""
            self.server.wait(timeout=5)
            raise AssertionError(f"server never started; output:\n{output}")

    def tearDown(self) -> None:
        self.server.terminate()
        try:
            self.server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.server.kill()
        self.socket_path.unlink(missing_ok=True)
        shutil.rmtree(self.telemetry_dir, ignore_errors=True)

    def test_real_client_round_trips_memory_through_a_real_kernel(self) -> None:
        with ExecutionProtocolClient(str(self.socket_path)) as client:
            put_value = client.execute(
                "req_py_put",
                memory_put("res_memory_default", "hello from real python client"),
            )
            self.assertIn("Data", put_value)

            get_value = client.execute("req_py_get", memory_get("res_memory_default"))
            entry = get_value["Data"]["MemoryEntry"]["entry"]
            self.assertEqual(entry["value"], {"Text": "hello from real python client"})

    def test_real_client_receives_a_typed_rejection_for_an_unknown_resource(self) -> None:
        with ExecutionProtocolClient(str(self.socket_path)) as client:
            with self.assertRaises(ExecutionProtocolError):
                client.execute(
                    "req_py_bad",
                    memory_get("res_memory_does_not_exist_and_is_not_granted"),
                )

    def test_real_client_completes_a_real_model_generation_end_to_end(self) -> None:
        with ExecutionProtocolClient(str(self.socket_path), timeout=60.0) as client:
            context = "context_context_default"
            response = client.execute("req_py_ctx", context_create(context))
            self.assertEqual(response, {"Revision": 0})

            response = client.execute(
                "req_py_append",
                context_append(
                    context, 0, "User", "Reply with exactly the single word: PASS"
                ),
            )
            revision = response["Revision"]

            model = "res_model_gemma4-e4b"
            response = client.execute(
                "req_py_prepare",
                context_prepare(context, revision, model, 2048, 4096),
            )
            input_bytes = response["Prepared"]["input_bytes"]

            response = client.execute(
                "req_py_gen",
                model_generate("req_py_prepare", model, input_bytes, 64),
            )
            generated = response["Generated"]
            print(f"real end-to-end Python generation: {generated['tokens']}")
            self.assertFalse(generated["fixture"])
            self.assertTrue(generated["tokens"])


if __name__ == "__main__":
    unittest.main()
